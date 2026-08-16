"""Dependency-light reference backend for the one-qubit Helstrom experiment.

The production implementation uses PyTorch autograd.  This module evaluates
the same coherent-then-dissipative model with NumPy and optimizes its seven
parameters using finite-difference Adam.  It is intentionally limited to one
qubit and exists both as an independent numerical oracle and as a fallback for
minimal environments where PyTorch is unavailable.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np


IDENTITY = np.eye(2, dtype=np.complex128)
PAULI_X = np.asarray(((0, 1), (1, 0)), dtype=np.complex128)
PAULI_Y = np.asarray(((0, -1j), (1j, 0)), dtype=np.complex128)
PAULI_Z = np.asarray(((1, 0), (0, -1)), dtype=np.complex128)
PAULIS = (PAULI_X, PAULI_Y, PAULI_Z)


@dataclass(frozen=True)
class NumpyBinaryEnsemble:
    rho0: np.ndarray
    rho1: np.ndarray
    priors: tuple[float, float]
    metadata: dict[str, Any]

    @property
    def states(self) -> np.ndarray:
        return np.stack((self.rho0, self.rho1), axis=0)


@dataclass(frozen=True)
class NumpyTrainConfig:
    epochs: int = 300
    learning_rate: float = 0.05
    leakage_penalty: float = 0.1
    grad_clip: float = 10.0
    log_every: int = 25
    finite_difference_step: float = 1e-5


def _normalize_priors(priors: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(priors, dtype=np.float64)
    if values.shape != (2,) or np.any(values < 0.0) or values.sum() <= 0.0:
        raise ValueError(f"Invalid binary priors: {priors}")
    values /= values.sum()
    return float(values[0]), float(values[1])


def _density(ket: np.ndarray) -> np.ndarray:
    ket = np.asarray(ket, dtype=np.complex128).reshape(2)
    ket = ket / np.linalg.norm(ket)
    return np.outer(ket, ket.conj())


def make_ensemble(
    separation_degrees: float,
    phase_degrees: float = 0.0,
    noise_model: str = "none",
    noise_strength: float = 0.0,
    priors: Sequence[float] = (0.5, 0.5),
) -> NumpyBinaryEnsemble:
    if not 0.0 <= separation_degrees <= 180.0:
        raise ValueError("separation_degrees must be in [0, 180].")
    if not 0.0 <= noise_strength <= 1.0:
        raise ValueError("noise_strength must be in [0, 1].")
    theta = np.deg2rad(separation_degrees)
    phase = np.deg2rad(phase_degrees)
    rho0 = _density(np.asarray((1.0, 0.0)))
    rho1 = _density(
        np.asarray((np.cos(theta / 2.0), np.exp(1j * phase) * np.sin(theta / 2.0)))
    )
    normalized_model = noise_model.strip().lower().replace("-", "_")
    if normalized_model in ("none", "identity"):
        if noise_strength != 0.0:
            raise ValueError("noise_strength must be zero for noise_model='none'.")
    elif normalized_model in ("depolarizing", "depolarising"):
        rho0 = (1.0 - noise_strength) * rho0 + noise_strength * IDENTITY / 2.0
        rho1 = (1.0 - noise_strength) * rho1 + noise_strength * IDENTITY / 2.0
    elif normalized_model in ("amplitude_damping", "amplitude"):
        k0 = np.asarray(((1.0, 0.0), (0.0, np.sqrt(1.0 - noise_strength))))
        k1 = np.asarray(((0.0, np.sqrt(noise_strength)), (0.0, 0.0)))
        rho0 = k0 @ rho0 @ k0.conj().T + k1 @ rho0 @ k1.conj().T
        rho1 = k0 @ rho1 @ k0.conj().T + k1 @ rho1 @ k1.conj().T
    else:
        raise ValueError(f"Unsupported noise model: {noise_model}")
    return NumpyBinaryEnsemble(
        rho0=rho0,
        rho1=rho1,
        priors=_normalize_priors(priors),
        metadata={
            "separation_degrees": float(separation_degrees),
            "phase_degrees": float(phase_degrees),
            "noise_model": normalized_model,
            "noise_strength": float(noise_strength),
        },
    )


def helstrom(ensemble: NumpyBinaryEnsemble) -> dict[str, Any]:
    eta0, eta1 = ensemble.priors
    decision = eta0 * ensemble.rho0 - eta1 * ensemble.rho1
    eigenvalues, eigenvectors = np.linalg.eigh(decision)
    positive = eigenvalues > 1e-12
    effect0 = (
        eigenvectors[:, positive] @ eigenvectors[:, positive].conj().T
        if np.any(positive)
        else np.zeros((2, 2), dtype=np.complex128)
    )
    effect1 = IDENTITY - effect0
    return {
        "success": 0.5 * (1.0 + np.abs(eigenvalues).sum()),
        "effect0": effect0,
        "effect1": effect1,
        "decision_operator_eigenvalues": eigenvalues,
    }


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0) + np.log1p(np.exp(-np.abs(values)))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0.0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _unitary(parameters: np.ndarray, duration: float) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=np.float64)
    radius = np.linalg.norm(parameters)
    if radius < 1e-14:
        return IDENTITY.copy()
    hamiltonian = sum(value * pauli for value, pauli in zip(parameters, PAULIS))
    return (
        np.cos(radius * duration) * IDENTITY
        - 1j * np.sin(radius * duration) * hamiltonian / radius
    )


def initial_qsnn_parameters(
    rng: np.random.Generator,
    dissipative_time: float,
    target_initial_output_mass: float,
) -> np.ndarray:
    rate = -math.log(1.0 - target_initial_output_mass) / dissipative_time
    return np.concatenate(
        (
            0.05 * rng.normal(size=3),
            np.full(2, _inverse_softplus(rate)),
            np.asarray((-2.0, 2.0)),
        )
    )


def qsnn_forward(
    parameters: np.ndarray,
    states: np.ndarray,
    coherent_time: float = 1.0,
    dissipative_time: float = 1.0,
) -> dict[str, np.ndarray]:
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape != (7,):
        raise ValueError(f"QSNN reference expects seven parameters, got {parameters.shape}")
    unitary = _unitary(parameters[:3], coherent_time)
    evolved = unitary[None, :, :] @ states @ unitary.conj().T[None, :, :]
    input_population = np.diagonal(evolved, axis1=-2, axis2=-1).real
    total_rates = _softplus(parameters[3:5]) + 1e-9
    class1_branches = _sigmoid(parameters[5:7])
    transferred = 1.0 - np.exp(-total_rates * dissipative_time)
    class1 = np.sum(input_population * transferred * class1_branches, axis=1)
    class0 = np.sum(input_population * transferred * (1.0 - class1_branches), axis=1)
    probabilities = np.stack((class0, class1), axis=1)
    output_mass = probabilities.sum(axis=1)
    return {
        "probabilities": probabilities,
        "output_mass": output_mass,
        "leakage": 1.0 - output_mass,
        "input_density_after_coherent_stage": evolved,
        "total_rates": total_rates,
        "class1_branches": class1_branches,
    }


def unitary_forward(
    parameters: np.ndarray,
    states: np.ndarray,
    coherent_time: float = 1.0,
) -> dict[str, np.ndarray]:
    unitary = _unitary(parameters, coherent_time)
    evolved = unitary[None, :, :] @ states @ unitary.conj().T[None, :, :]
    probabilities = np.diagonal(evolved, axis1=-2, axis2=-1).real
    return {
        "probabilities": probabilities,
        "output_mass": np.ones(states.shape[0]),
        "leakage": np.zeros(states.shape[0]),
    }


def _objective(
    parameters: np.ndarray,
    ensemble: NumpyBinaryEnsemble,
    forward: Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]],
    leakage_penalty: float,
    regularize_qsnn: bool,
) -> tuple[float, float, float, dict[str, np.ndarray]]:
    outputs = forward(parameters, ensemble.states)
    priors = np.asarray(ensemble.priors)
    success = float(
        priors[0] * outputs["probabilities"][0, 0]
        + priors[1] * outputs["probabilities"][1, 1]
    )
    leakage = float(np.dot(priors, outputs["leakage"]))
    regularization = 0.0
    if regularize_qsnn:
        regularization = 1e-5 * np.mean(outputs["total_rates"] ** 2)
        regularization += 1e-6 * np.mean(parameters[:3] ** 2)
    else:
        regularization = 1e-6 * np.mean(parameters**2)
    loss = -success + leakage_penalty * leakage + regularization
    return float(loss), success, leakage, outputs


def _finite_difference_gradient(
    parameters: np.ndarray,
    loss_function: Callable[[np.ndarray], float],
    step: float,
) -> np.ndarray:
    gradient = np.zeros_like(parameters)
    for index in range(parameters.size):
        offset = np.zeros_like(parameters)
        offset[index] = step
        gradient[index] = (
            loss_function(parameters + offset) - loss_function(parameters - offset)
        ) / (2.0 * step)
    return gradient


def train(
    initial_parameters: np.ndarray,
    ensemble: NumpyBinaryEnsemble,
    forward: Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]],
    config: NumpyTrainConfig,
    *,
    regularize_qsnn: bool,
) -> dict[str, Any]:
    parameters = np.asarray(initial_parameters, dtype=np.float64).copy()
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    beta1, beta2 = 0.9, 0.999
    best_success = -np.inf
    best_parameters = parameters.copy()
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    def scalar_loss(candidate: np.ndarray) -> float:
        return _objective(
            candidate,
            ensemble,
            forward,
            config.leakage_penalty,
            regularize_qsnn,
        )[0]

    for epoch in range(config.epochs):
        gradient = _finite_difference_gradient(
            parameters, scalar_loss, config.finite_difference_step
        )
        norm = np.linalg.norm(gradient)
        if config.grad_clip > 0.0 and norm > config.grad_clip:
            gradient *= config.grad_clip / norm
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
        corrected_first = first_moment / (1.0 - beta1 ** (epoch + 1))
        corrected_second = second_moment / (1.0 - beta2 ** (epoch + 1))
        parameters -= config.learning_rate * corrected_first / (
            np.sqrt(corrected_second) + 1e-8
        )
        loss, success, leakage, _ = _objective(
            parameters,
            ensemble,
            forward,
            config.leakage_penalty,
            regularize_qsnn,
        )
        if success > best_success:
            best_success = success
            best_parameters = parameters.copy()
        if (
            epoch == 0
            or epoch + 1 == config.epochs
            or (config.log_every > 0 and (epoch + 1) % config.log_every == 0)
        ):
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": loss,
                    "success": success,
                    "leakage": leakage,
                }
            )

    parameters = best_parameters
    loss, success, leakage, outputs = _objective(
        parameters,
        ensemble,
        forward,
        config.leakage_penalty,
        regularize_qsnn,
    )
    bound = float(helstrom(ensemble)["success"])
    return {
        "parameters": parameters,
        "loss": loss,
        "success": success,
        "weighted_leakage": leakage,
        "helstrom_success": bound,
        "helstrom_gap": max(bound - success, 0.0),
        "conditional_probabilities": outputs["probabilities"],
        "history": history,
        "training_seconds": time.perf_counter() - started,
        "training": asdict(config),
    }


def effective_povm(
    parameters: np.ndarray,
    forward: Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]],
) -> np.ndarray:
    mixed = IDENTITY / 2.0
    identity_coefficients = forward(parameters, mixed[None, :, :])["probabilities"][0]
    vector_coefficients = []
    for pauli in PAULIS:
        plus = (IDENTITY + pauli) / 2.0
        minus = (IDENTITY - pauli) / 2.0
        values = forward(parameters, np.stack((plus, minus)))["probabilities"]
        vector_coefficients.append(0.5 * (values[0] - values[1]))
    effects = []
    for output in range(2):
        effect = identity_coefficients[output] * IDENTITY
        for coefficient, pauli in zip(vector_coefficients, PAULIS):
            effect = effect + coefficient[output] * pauli
        effects.append(0.5 * (effect + effect.conj().T))
    return np.stack(effects)


def povm_diagnostics(effects: np.ndarray) -> dict[str, float]:
    leakage = IDENTITY - effects.sum(axis=0)
    eigenvalues = [np.linalg.eigvalsh(effect) for effect in (*effects, leakage)]
    return {
        "effect0_min_eigenvalue": float(eigenvalues[0].min()),
        "effect1_min_eigenvalue": float(eigenvalues[1].min()),
        "leakage_effect_min_eigenvalue": float(eigenvalues[2].min()),
        "minimum_effect_eigenvalue": float(min(values.min() for values in eigenvalues)),
    }


def best_fixed_pauli_success(ensemble: NumpyBinaryEnsemble) -> tuple[float, str]:
    best, name = max(ensemble.priors), "majority"
    for axis_name, pauli in zip(("X", "Y", "Z"), PAULIS):
        plus, minus = (IDENTITY + pauli) / 2.0, (IDENTITY - pauli) / 2.0
        for mapping_name, effect0, effect1 in (
            (f"{axis_name}:+->0", plus, minus),
            (f"{axis_name}:+->1", minus, plus),
        ):
            success = float(
                ensemble.priors[0] * np.trace(effect0 @ ensemble.rho0).real
                + ensemble.priors[1] * np.trace(effect1 @ ensemble.rho1).real
            )
            if success > best:
                best, name = success, mapping_name
    return best, name


def simulate_shots(
    conditional_probabilities: np.ndarray,
    priors: tuple[float, float],
    shots: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    class_success = []
    for label in (0, 1):
        p0, p1 = conditional_probabilities[label]
        values = np.clip(np.asarray((p0, p1, 1.0 - p0 - p1)), 0.0, None)
        values /= values.sum()
        counts = rng.multinomial(shots, values)
        class_success.append(counts[label] / shots)
    return float(priors[0] * class_success[0] + priors[1] * class_success[1])


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return copy.deepcopy(value)
