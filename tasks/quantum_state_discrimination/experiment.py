from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .bounds import helstrom_measurement
from .models import effective_povm, povm_diagnostics
from .states import BinaryStateEnsemble


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 300
    learning_rate: float = 0.05
    leakage_penalty: float = 0.1
    grad_clip: float = 10.0
    log_every: int = 25


def _weighted_success(outputs: dict[str, torch.Tensor], priors: tuple[float, float]) -> torch.Tensor:
    probabilities = outputs["probabilities"]
    weights = torch.as_tensor(priors, dtype=probabilities.dtype, device=probabilities.device)
    correct = torch.stack((probabilities[0, 0], probabilities[1, 1]))
    return torch.sum(weights * correct)


def _weighted_leakage(outputs: dict[str, torch.Tensor], priors: tuple[float, float]) -> torch.Tensor:
    weights = torch.as_tensor(
        priors,
        dtype=outputs["leakage"].dtype,
        device=outputs["leakage"].device,
    )
    return torch.sum(weights * outputs["leakage"])


def evaluate_discriminator(
    model: nn.Module,
    ensemble: BinaryStateEnsemble,
) -> dict[str, Any]:
    model.eval()
    parameter = next(model.parameters())
    local_ensemble = ensemble.to(parameter.device)
    with torch.no_grad():
        outputs = model(local_ensemble.states)
        success = float(_weighted_success(outputs, local_ensemble.priors))
        leakage = float(_weighted_leakage(outputs, local_ensemble.priors))
        probabilities = outputs["probabilities"].detach().cpu()
        effects = effective_povm(model)
        diagnostics = povm_diagnostics(effects)
    bound = helstrom_measurement(local_ensemble).success
    return {
        "success": success,
        "helstrom_success": bound,
        "helstrom_gap": max(bound - success, 0.0),
        "weighted_leakage": leakage,
        "conditional_probabilities": probabilities.tolist(),
        "povm_diagnostics": diagnostics,
        "effects": effects.detach().cpu(),
    }


def train_discriminator(
    model: nn.Module,
    ensemble: BinaryStateEnsemble,
    config: TrainConfig = TrainConfig(),
) -> dict[str, Any]:
    if config.epochs < 1:
        raise ValueError("epochs must be positive.")
    parameter = next(model.parameters())
    local_ensemble = ensemble.to(parameter.device)
    states = local_ensemble.states
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_success = -float("inf")
    best_state = None
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        outputs = model(states)
        success = _weighted_success(outputs, local_ensemble.priors)
        leakage = _weighted_leakage(outputs, local_ensemble.priors)
        regularization = (
            model.regularization()
            if hasattr(model, "regularization")
            else torch.zeros((), dtype=success.dtype, device=success.device)
        )
        loss = -success + config.leakage_penalty * leakage + regularization
        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        with torch.no_grad():
            updated_outputs = model(states)
            updated_success = _weighted_success(updated_outputs, local_ensemble.priors)
            updated_leakage = _weighted_leakage(updated_outputs, local_ensemble.priors)
        success_value = float(updated_success)
        if success_value > best_success:
            best_success = success_value
            best_state = copy.deepcopy(model.state_dict())
        if (
            epoch == 0
            or epoch + 1 == config.epochs
            or (config.log_every > 0 and (epoch + 1) % config.log_every == 0)
        ):
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(loss.detach()),
                    "success": success_value,
                    "leakage": float(updated_leakage),
                }
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate_discriminator(model, local_ensemble)
    metrics.update(
        {
            "training_seconds": time.perf_counter() - started,
            "training": asdict(config),
            "history": history,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    return metrics


def simulate_shot_success(
    conditional_probabilities: list[list[float]],
    priors: tuple[float, float],
    shots: int,
    seed: int,
) -> float:
    """Sample output0/output1/leakage; leakage is a failed decision."""

    if shots < 1:
        raise ValueError("shots must be positive.")
    rng = np.random.default_rng(seed)
    successes = []
    for label in (0, 1):
        p0, p1 = conditional_probabilities[label]
        leakage = max(1.0 - p0 - p1, 0.0)
        probabilities = np.asarray((p0, p1, leakage), dtype=np.float64)
        probabilities = np.clip(probabilities, 0.0, None)
        probabilities /= probabilities.sum()
        counts = rng.multinomial(shots, probabilities)
        successes.append(counts[label] / shots)
    return float(priors[0] * successes[0] + priors[1] * successes[1])
