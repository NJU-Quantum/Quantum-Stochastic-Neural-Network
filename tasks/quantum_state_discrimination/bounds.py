from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .states import BinaryStateEnsemble


@dataclass(frozen=True)
class HelstromResult:
    success: float
    effect0: torch.Tensor
    effect1: torch.Tensor
    decision_operator_eigenvalues: torch.Tensor


def measurement_success(
    rho0: torch.Tensor,
    rho1: torch.Tensor,
    effect0: torch.Tensor,
    effect1: torch.Tensor,
    priors: Sequence[float] = (0.5, 0.5),
) -> float:
    p0, p1 = float(priors[0]), float(priors[1])
    total = p0 + p1
    p0, p1 = p0 / total, p1 / total
    value = p0 * torch.trace(effect0 @ rho0).real + p1 * torch.trace(effect1 @ rho1).real
    return float(value)


def helstrom_measurement(
    ensemble_or_rho0: BinaryStateEnsemble | torch.Tensor,
    rho1: torch.Tensor | None = None,
    priors: Sequence[float] = (0.5, 0.5),
    *,
    eig_tolerance: float = 1e-12,
) -> HelstromResult:
    """Return the minimum-error binary POVM and Holevo-Helstrom success bound."""

    if isinstance(ensemble_or_rho0, BinaryStateEnsemble):
        ensemble = ensemble_or_rho0
        rho0, rho1 = ensemble.rho0, ensemble.rho1
        priors = ensemble.priors
    else:
        rho0 = ensemble_or_rho0
        if rho1 is None:
            raise ValueError("rho1 is required when no BinaryStateEnsemble is supplied.")

    eta0, eta1 = float(priors[0]), float(priors[1])
    total = eta0 + eta1
    eta0, eta1 = eta0 / total, eta1 / total
    decision = eta0 * rho0 - eta1 * rho1
    eigenvalues, eigenvectors = torch.linalg.eigh(decision)
    positive = eigenvalues > eig_tolerance
    if bool(positive.any()):
        vectors = eigenvectors[:, positive]
        effect0 = vectors @ vectors.mH
    else:
        effect0 = torch.zeros_like(rho0)
    identity = torch.eye(2, dtype=rho0.dtype, device=rho0.device)
    effect1 = identity - effect0
    success = 0.5 * (1.0 + float(torch.linalg.vector_norm(eigenvalues, ord=1)))
    return HelstromResult(success, effect0, effect1, eigenvalues)


def best_fixed_pauli_success(ensemble: BinaryStateEnsemble) -> tuple[float, str]:
    """Best deterministic classifier made from one fixed X/Y/Z measurement."""

    dtype, device = ensemble.rho0.dtype, ensemble.rho0.device
    identity = torch.eye(2, dtype=dtype, device=device)
    paulis = {
        "X": torch.tensor(((0, 1), (1, 0)), dtype=dtype, device=device),
        "Y": torch.tensor(((0, -1j), (1j, 0)), dtype=dtype, device=device),
        "Z": torch.tensor(((1, 0), (0, -1)), dtype=dtype, device=device),
    }
    best = max(ensemble.priors)
    best_name = "majority"
    for name, pauli in paulis.items():
        plus = 0.5 * (identity + pauli)
        minus = identity - plus
        direct = measurement_success(
            ensemble.rho0, ensemble.rho1, plus, minus, ensemble.priors
        )
        swapped = measurement_success(
            ensemble.rho0, ensemble.rho1, minus, plus, ensemble.priors
        )
        if direct > best:
            best, best_name = direct, f"{name}:+->0"
        if swapped > best:
            best, best_name = swapped, f"{name}:+->1"
    return best, best_name
