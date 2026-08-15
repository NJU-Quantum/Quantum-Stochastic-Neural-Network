"""Adversarial entanglement-witness utilities for two-qubit benchmarks.

The discriminator score is required to be a linear functional of the input
density matrix.  A trained score ``Tr(O rho)`` is calibrated against the
largest value attainable by a separable state, producing the certified
witness ``W = c_sep I - O``.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


class SeparableMixtureGenerator(nn.Module):
    """Differentiable mixture of two-qubit pure product states.

    Every output is separable by construction:

    ``rho = sum_k softmax(logits)_k |a_k b_k><a_k b_k|``.
    """

    def __init__(self, components: int = 16, real_dtype: torch.dtype = torch.float64):
        super().__init__()
        if components <= 0:
            raise ValueError("components must be positive")
        self.components = int(components)
        angles = torch.empty(self.components, 4, dtype=real_dtype)
        angles[:, 0].uniform_(0.0, math.pi)
        angles[:, 1].uniform_(-math.pi, math.pi)
        angles[:, 2].uniform_(0.0, math.pi)
        angles[:, 3].uniform_(-math.pi, math.pi)
        self.angles = nn.Parameter(angles)
        self.logits = nn.Parameter(torch.zeros(self.components, dtype=real_dtype))

    @property
    def complex_dtype(self) -> torch.dtype:
        return torch.complex128 if self.angles.dtype == torch.float64 else torch.complex64

    @staticmethod
    def _qubit_states(theta: torch.Tensor, phi: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack(
            [
                torch.cos(0.5 * theta).to(dtype),
                torch.exp(1j * phi.to(dtype)) * torch.sin(0.5 * theta).to(dtype),
            ],
            dim=-1,
        )

    def component_states(self) -> torch.Tensor:
        first = self._qubit_states(self.angles[:, 0], self.angles[:, 1], self.complex_dtype)
        second = self._qubit_states(self.angles[:, 2], self.angles[:, 3], self.complex_dtype)
        return torch.einsum("ki,kj->kij", first, second).reshape(self.components, 4)

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def forward(self) -> torch.Tensor:
        states = self.component_states()
        projectors = states[:, :, None] @ states.conj()[:, None, :]
        return torch.einsum("k,kij->ij", self.weights().to(projectors.dtype), projectors)


def _score(discriminator: nn.Module, rho: torch.Tensor) -> torch.Tensor:
    return discriminator(rho)["z_expectation"].real


@torch.no_grad()
def effective_observable(discriminator: nn.Module, input_dim: int = 4) -> torch.Tensor:
    """Reconstruct the Hermitian input observable behind a linear score.

    Only physical pure-state probes are used.  The method therefore also
    works with a hardware-like discriminator interface that accepts density
    matrices but not arbitrary operator-basis elements.
    """

    parameter = next(discriminator.parameters())
    complex_dtype = torch.complex128 if parameter.dtype == torch.float64 else torch.complex64
    device = parameter.device
    basis = torch.eye(input_dim, dtype=complex_dtype, device=device)
    diagonal = []
    for index in range(input_dim):
        state = basis[index]
        diagonal.append(_score(discriminator, state[:, None] @ state.conj()[None, :]))
    diagonal_tensor = torch.stack(diagonal)
    observable = torch.diag(diagonal_tensor.to(complex_dtype))
    for left in range(input_dim):
        for right in range(left + 1, input_dim):
            real_state = (basis[left] + basis[right]) / math.sqrt(2.0)
            imag_state = (basis[left] + 1j * basis[right]) / math.sqrt(2.0)
            real_score = _score(
                discriminator, real_state[:, None] @ real_state.conj()[None, :]
            )
            imag_score = _score(
                discriminator, imag_state[:, None] @ imag_state.conj()[None, :]
            )
            diagonal_mean = 0.5 * (diagonal_tensor[left] + diagonal_tensor[right])
            real_part = real_score - diagonal_mean
            imag_part = diagonal_mean - imag_score
            value = real_part.to(complex_dtype) + 1j * imag_part.to(complex_dtype)
            observable[left, right] = value
            observable[right, left] = value.conj()
    return 0.5 * (observable + observable.mH)


def observable_score(observable: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Evaluate ``Tr(O rho)`` with optional density-matrix batching."""

    return torch.einsum("ij,...ji->...", observable, rho).real


def pauli_basis(dtype: torch.dtype = torch.complex128, device=None) -> dict[str, torch.Tensor]:
    zero = torch.tensor(0.0, dtype=dtype, device=device)
    one = torch.tensor(1.0, dtype=dtype, device=device)
    return {
        "I": torch.eye(2, dtype=dtype, device=device),
        "X": torch.stack([torch.stack([zero, one]), torch.stack([one, zero])]),
        "Y": torch.stack(
            [torch.stack([zero, -1j * one]), torch.stack([1j * one, zero])]
        ),
        "Z": torch.diag(torch.stack([one, -one])),
    }


def pauli_coefficients(observable: torch.Tensor) -> dict[str, float]:
    """Return coefficients in ``O = sum_ab c_ab (a tensor b)``."""

    paulis = pauli_basis(dtype=observable.dtype, device=observable.device)
    coefficients = {}
    for left_name, left in paulis.items():
        for right_name, right in paulis.items():
            operator = torch.kron(left.contiguous(), right.contiguous())
            value = 0.25 * torch.trace(observable @ operator).real
            coefficients[f"{left_name}{right_name}"] = float(value.detach().cpu())
    return coefficients


def _bloch_parameters(observable: torch.Tensor):
    coefficients = pauli_coefficients(observable)
    labels = ("X", "Y", "Z")
    alpha = coefficients["II"]
    first = np.array([coefficients[f"{label}I"] for label in labels], dtype=np.float64)
    second = np.array([coefficients[f"I{label}"] for label in labels], dtype=np.float64)
    correlations = np.array(
        [[coefficients[f"{left}{right}"] for right in labels] for left in labels],
        dtype=np.float64,
    )
    return alpha, first, second, correlations


@dataclass(frozen=True)
class SeparableScoreBound:
    lower: float
    upper: float
    gap: float
    evaluations: int
    converged: bool
    first_bloch: tuple[float, float, float]
    second_bloch: tuple[float, float, float]


def certified_separable_score_bound(
    observable: torch.Tensor,
    tolerance: float = 5e-4,
    max_cells: int = 200_000,
    initial_theta_cells: int = 12,
    initial_phi_cells: int = 24,
) -> SeparableScoreBound:
    """Bound the maximum score over all two-qubit separable states.

    A linear functional reaches its separable maximum on a pure product
    state.  Maximizing analytically over the second Bloch vector leaves

    ``g(x) = alpha + a.x + ||b + T^T x||``.

    The remaining sphere search uses a Lipschitz branch-and-bound.  The
    returned ``upper`` remains a valid global upper bound even when the cell
    budget is reached; ``gap`` reports its numerical conservatism.
    """

    if observable.shape != (4, 4):
        raise ValueError("observable must be a two-qubit 4x4 matrix")
    if tolerance <= 0 or max_cells <= 0:
        raise ValueError("tolerance and max_cells must be positive")
    alpha, first, second, correlations = _bloch_parameters(observable)
    lipschitz = float(np.linalg.norm(first) + np.linalg.norm(correlations, ord=2))
    correlation_square = correlations @ correlations.T
    global_norm_square_upper = (
        float(second @ second)
        + 2.0 * float(np.linalg.norm(correlations @ second))
        + float(np.linalg.eigvalsh(correlation_square).max())
    )
    global_upper = float(
        alpha + np.linalg.norm(first) + math.sqrt(max(0.0, global_norm_square_upper))
    )

    def bloch(theta: float, phi: float) -> np.ndarray:
        sin_theta = math.sin(theta)
        return np.array(
            [sin_theta * math.cos(phi), sin_theta * math.sin(phi), math.cos(theta)],
            dtype=np.float64,
        )

    def value(theta: float, phi: float):
        x = bloch(theta, phi)
        effective_second = second + correlations.T @ x
        return float(alpha + first @ x + np.linalg.norm(effective_second)), x, effective_second

    if lipschitz <= 1e-15:
        score = float(alpha + np.linalg.norm(second))
        y = second / np.linalg.norm(second) if np.linalg.norm(second) > 0 else np.array([0.0, 0.0, 1.0])
        return SeparableScoreBound(score, score, 0.0, 1, True, (0.0, 0.0, 1.0), tuple(y))

    best_value = -math.inf
    best_x = np.array([0.0, 0.0, 1.0])
    best_effective_second = second.copy()
    evaluations = 0
    heap: list[tuple[float, int, float, float, float, float]] = []
    serial = 0

    def sin_cap(theta_low: float, theta_high: float) -> float:
        if theta_low <= math.pi / 2.0 <= theta_high:
            return 1.0
        return max(math.sin(theta_low), math.sin(theta_high))

    def add_cell(theta_low: float, theta_high: float, phi_low: float, phi_high: float):
        nonlocal best_value, best_x, best_effective_second, evaluations, serial
        theta_mid = 0.5 * (theta_low + theta_high)
        phi_mid = 0.5 * (phi_low + phi_high)
        score, x, effective_second = value(theta_mid, phi_mid)
        evaluations += 1
        if score > best_value:
            best_value = score
            best_x = x
            best_effective_second = effective_second
        # A meridian-then-parallel path gives a safe angular-distance bound.
        radius = 0.5 * (theta_high - theta_low) + sin_cap(theta_low, theta_high) * 0.5 * (phi_high - phi_low)
        radius = min(2.0, radius)
        upper = min(global_upper, score + lipschitz * radius)
        heapq.heappush(heap, (-upper, serial, theta_low, theta_high, phi_low, phi_high))
        serial += 1

    theta_edges = np.linspace(0.0, math.pi, initial_theta_cells + 1)
    phi_edges = np.linspace(0.0, 2.0 * math.pi, initial_phi_cells + 1)
    for theta_index in range(initial_theta_cells):
        for phi_index in range(initial_phi_cells):
            add_cell(
                float(theta_edges[theta_index]),
                float(theta_edges[theta_index + 1]),
                float(phi_edges[phi_index]),
                float(phi_edges[phi_index + 1]),
            )

    # Improve the feasible lower bound before branching.
    best_theta = math.acos(float(np.clip(best_x[2], -1.0, 1.0)))
    best_phi = math.atan2(float(best_x[1]), float(best_x[0])) % (2.0 * math.pi)
    optimum = minimize(
        lambda point: -value(float(point[0]), float(point[1]))[0],
        np.array([best_theta, best_phi]),
        method="L-BFGS-B",
        bounds=((0.0, math.pi), (0.0, 2.0 * math.pi)),
    )
    if optimum.success:
        score, x, effective_second = value(float(optimum.x[0]), float(optimum.x[1]))
        evaluations += int(optimum.nfev)
        if score > best_value:
            best_value = score
            best_x = x
            best_effective_second = effective_second

    while heap and -heap[0][0] - best_value > tolerance and len(heap) < max_cells:
        _negative_upper, _serial, theta_low, theta_high, phi_low, phi_high = heapq.heappop(heap)
        theta_width = theta_high - theta_low
        phi_width = phi_high - phi_low
        if theta_width >= sin_cap(theta_low, theta_high) * phi_width:
            midpoint = 0.5 * (theta_low + theta_high)
            add_cell(theta_low, midpoint, phi_low, phi_high)
            add_cell(midpoint, theta_high, phi_low, phi_high)
        else:
            midpoint = 0.5 * (phi_low + phi_high)
            add_cell(theta_low, theta_high, phi_low, midpoint)
            add_cell(theta_low, theta_high, midpoint, phi_high)

    upper = max(best_value, -heap[0][0] if heap else best_value)
    second_norm = np.linalg.norm(best_effective_second)
    best_y = (
        best_effective_second / second_norm
        if second_norm > 1e-15
        else np.array([0.0, 0.0, 1.0])
    )
    return SeparableScoreBound(
        lower=float(best_value),
        upper=float(upper + 1e-12),
        gap=float(max(0.0, upper - best_value)),
        evaluations=evaluations,
        converged=bool(upper - best_value <= tolerance),
        first_bloch=tuple(float(value) for value in best_x),
        second_bloch=tuple(float(value) for value in best_y),
    )


def calibrated_witness(observable: torch.Tensor, separable_upper: float) -> torch.Tensor:
    identity = torch.eye(4, dtype=observable.dtype, device=observable.device)
    return separable_upper * identity - observable


def werner_psi_plus_witness(
    dtype: torch.dtype = torch.complex128, device=None
) -> torch.Tensor:
    state = torch.zeros(4, dtype=dtype, device=device)
    state[1] = 2.0**-0.5
    state[2] = 2.0**-0.5
    projector = state[:, None] @ state.conj()[None, :]
    return 0.5 * torch.eye(4, dtype=dtype, device=device) - projector
