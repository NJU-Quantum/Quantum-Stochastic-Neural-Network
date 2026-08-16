from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch


def _check_probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _normalize_priors(priors: Sequence[float]) -> tuple[float, float]:
    if len(priors) != 2:
        raise ValueError("Binary state discrimination requires exactly two priors.")
    p0, p1 = (float(priors[0]), float(priors[1]))
    if p0 < 0.0 or p1 < 0.0 or p0 + p1 <= 0.0:
        raise ValueError(f"Invalid priors: {priors}")
    total = p0 + p1
    return p0 / total, p1 / total


def validate_density_matrix(rho: torch.Tensor, atol: float = 1e-9) -> None:
    if rho.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 qubit density matrix, got {tuple(rho.shape)}")
    if not torch.isfinite(rho).all():
        raise ValueError("Density matrix contains a non-finite value.")
    if not torch.allclose(rho, rho.mH, atol=atol, rtol=0.0):
        raise ValueError("Density matrix must be Hermitian.")
    trace = torch.trace(rho).real
    if not torch.allclose(trace, torch.ones_like(trace), atol=atol, rtol=0.0):
        raise ValueError(f"Density matrix must have trace one, got {float(trace)}")
    if float(torch.linalg.eigvalsh(rho).min().real) < -atol:
        raise ValueError("Density matrix must be positive semidefinite.")


@dataclass(frozen=True)
class BinaryStateEnsemble:
    """Two quantum states with prior probabilities and reproducibility metadata."""

    rho0: torch.Tensor
    rho1: torch.Tensor
    priors: tuple[float, float] = (0.5, 0.5)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_density_matrix(self.rho0)
        validate_density_matrix(self.rho1)
        object.__setattr__(self, "priors", _normalize_priors(self.priors))

    @property
    def states(self) -> torch.Tensor:
        return torch.stack((self.rho0, self.rho1), dim=0)

    def to(self, device: str | torch.device) -> "BinaryStateEnsemble":
        return BinaryStateEnsemble(
            self.rho0.to(device),
            self.rho1.to(device),
            self.priors,
            dict(self.metadata),
        )


def pure_density(ket: torch.Tensor) -> torch.Tensor:
    ket = ket.reshape(-1, 1)
    norm = torch.linalg.vector_norm(ket)
    if float(norm) <= 0.0:
        raise ValueError("A quantum state vector cannot be zero.")
    ket = ket / norm
    return ket @ ket.mH


def depolarize(rho: torch.Tensor, strength: float) -> torch.Tensor:
    strength = _check_probability(strength, "depolarizing strength")
    identity = torch.eye(2, dtype=rho.dtype, device=rho.device) / 2.0
    return (1.0 - strength) * rho + strength * identity


def amplitude_damping(rho: torch.Tensor, strength: float) -> torch.Tensor:
    strength = _check_probability(strength, "amplitude-damping strength")
    real_dtype = rho.real.dtype
    one = torch.ones((), dtype=real_dtype, device=rho.device)
    gamma = torch.as_tensor(strength, dtype=real_dtype, device=rho.device)
    zero = torch.zeros((), dtype=rho.dtype, device=rho.device)
    k0 = torch.stack(
        (
            torch.stack((one.to(rho.dtype), zero)),
            torch.stack((zero, torch.sqrt(one - gamma).to(rho.dtype))),
        )
    )
    k1 = torch.stack(
        (
            torch.stack((zero, torch.sqrt(gamma).to(rho.dtype))),
            torch.stack((zero, zero)),
        )
    )
    return k0 @ rho @ k0.mH + k1 @ rho @ k1.mH


def make_nonorthogonal_qubit_ensemble(
    separation_degrees: float,
    phase_degrees: float = 0.0,
    noise_model: str = "none",
    noise_strength: float = 0.0,
    priors: Sequence[float] = (0.5, 0.5),
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.complex128,
) -> BinaryStateEnsemble:
    """Construct |0> and a non-orthogonal state separated on the Bloch sphere.

    ``separation_degrees`` is the Bloch-sphere angle.  The pure-state overlap is
    therefore cos(separation/2), rather than cos(separation).
    """

    if not 0.0 <= float(separation_degrees) <= 180.0:
        raise ValueError("separation_degrees must be in [0, 180].")
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    theta = torch.deg2rad(
        torch.as_tensor(separation_degrees, dtype=real_dtype, device=device)
    )
    phase = torch.deg2rad(torch.as_tensor(phase_degrees, dtype=real_dtype, device=device))
    ket0 = torch.tensor((1.0, 0.0), dtype=dtype, device=device)
    ket1 = torch.stack(
        (
            torch.cos(theta / 2.0).to(dtype),
            (torch.exp(1j * phase) * torch.sin(theta / 2.0)).to(dtype),
        )
    )
    rho0 = pure_density(ket0)
    rho1 = pure_density(ket1)

    normalized_model = str(noise_model).strip().lower().replace("-", "_")
    if normalized_model in ("none", "identity"):
        if float(noise_strength) != 0.0:
            raise ValueError("noise_strength must be zero when noise_model='none'.")
    elif normalized_model in ("depolarizing", "depolarising"):
        rho0 = depolarize(rho0, noise_strength)
        rho1 = depolarize(rho1, noise_strength)
    elif normalized_model in ("amplitude_damping", "amplitude"):
        rho0 = amplitude_damping(rho0, noise_strength)
        rho1 = amplitude_damping(rho1, noise_strength)
    else:
        raise ValueError(f"Unsupported noise model: {noise_model}")

    return BinaryStateEnsemble(
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
