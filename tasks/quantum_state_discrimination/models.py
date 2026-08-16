from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


def _paulis(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, ...]:
    identity = torch.eye(2, dtype=dtype, device=device)
    x = torch.tensor(((0, 1), (1, 0)), dtype=dtype, device=device)
    y = torch.tensor(((0, -1j), (1j, 0)), dtype=dtype, device=device)
    z = torch.tensor(((1, 0), (0, -1)), dtype=dtype, device=device)
    return identity, x, y, z


def _bloch_hamiltonian(parameters: torch.Tensor, complex_dtype: torch.dtype) -> torch.Tensor:
    _, x, y, z = _paulis(complex_dtype, parameters.device)
    return parameters[0] * x + parameters[1] * y + parameters[2] * z


def _embed_qubit_density(rho: torch.Tensor) -> torch.Tensor:
    if rho.dim() == 2:
        rho = rho.unsqueeze(0)
        squeeze_back = True
    elif rho.dim() == 3:
        squeeze_back = False
    else:
        raise ValueError(f"rho must have shape (2,2) or (B,2,2), got {tuple(rho.shape)}")
    if rho.shape[-2:] != (2, 2):
        raise ValueError(f"Expected qubit states, got trailing shape {tuple(rho.shape[-2:])}")
    embedded = torch.zeros((rho.shape[0], 4, 4), dtype=rho.dtype, device=rho.device)
    embedded[:, :2, :2] = rho
    return embedded[0] if squeeze_back else embedded


def _unitary_evolve(rho: torch.Tensor, hamiltonian: torch.Tensor, duration: float) -> torch.Tensor:
    unitary = torch.matrix_exp((-1j) * hamiltonian * duration)
    if rho.dim() == 2:
        return unitary @ rho @ unitary.mH
    return unitary.unsqueeze(0) @ rho @ unitary.mH.unsqueeze(0)


def _structured_dissipative_step(
    rho: torch.Tensor,
    gamma: torch.Tensor,
    duration: float,
) -> torch.Tensor:
    """Exact evolution for jumps gamma[c,j] |out_c><input_j|."""

    if rho.dim() == 2:
        rho_b = rho.unsqueeze(0)
        squeeze_back = True
    else:
        rho_b = rho
        squeeze_back = False
    real_dtype = rho_b.real.dtype
    rates = gamma.abs().square().to(real_dtype)
    total_rates = rates.sum(dim=0)
    damp = torch.zeros(4, dtype=real_dtype, device=rho_b.device)
    damp[:2] = total_rates
    decay = torch.exp(
        -0.5 * duration * (damp.view(4, 1) + damp.view(1, 4))
    ).to(rho_b.dtype)
    out = rho_b * decay.view(1, 4, 4)

    input_population = torch.diagonal(rho_b[:, :2, :2], dim1=-2, dim2=-1).real
    transfer_time = torch.where(
        total_rates > 1e-12,
        (1.0 - torch.exp(-total_rates * duration)) / total_rates,
        torch.full_like(total_rates, duration),
    )
    for output in range(2):
        gain = (input_population * (rates[output] * transfer_time).view(1, 2)).sum(dim=1)
        out[:, 2 + output, 2 + output] = (
            out[:, 2 + output, 2 + output] + gain.to(rho_b.dtype)
        )
    return out[0] if squeeze_back else out


class QubitHelstromQSNN(nn.Module):
    """Two-input/two-sink QSNN that learns a binary quantum measurement.

    The model receives density matrices directly.  A trainable coherent
    rotation acts on the qubit input subspace, followed by structured Lindblad
    jumps into two absorbing class nodes.  Residual input population is exposed
    as leakage and is always counted as a failed discrimination event.
    """

    def __init__(
        self,
        coherent_time: float = 1.0,
        dissipative_time: float = 1.0,
        target_initial_output_mass: float = 0.995,
        coherent_during_dissipation: bool = False,
        dissipative_steps: int = 8,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if not 0.0 < target_initial_output_mass < 1.0:
            raise ValueError("target_initial_output_mass must be strictly between zero and one.")
        if coherent_time < 0.0 or dissipative_time <= 0.0:
            raise ValueError("Invalid coherent or dissipative duration.")
        if dissipative_steps < 1:
            raise ValueError("dissipative_steps must be positive.")
        self.coherent_time = float(coherent_time)
        self.dissipative_time = float(dissipative_time)
        self.coherent_during_dissipation = bool(coherent_during_dissipation)
        self.dissipative_steps = int(dissipative_steps)
        self.real_dtype = dtype
        self.complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.device_name = str(device)

        self.hamiltonian_parameters = nn.Parameter(
            0.05 * torch.randn(3, dtype=dtype, device=device)
        )
        initial_rate = -math.log(1.0 - target_initial_output_mass) / self.dissipative_time
        self.raw_total_rates = nn.Parameter(
            torch.full((2,), _inverse_softplus(initial_rate), dtype=dtype, device=device)
        )
        # Initially send computational |0> mainly to class 0 and |1> to class 1.
        self.class1_branch_logits = nn.Parameter(
            torch.tensor((-2.0, 2.0), dtype=dtype, device=device)
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def input_hamiltonian(self) -> torch.Tensor:
        return _bloch_hamiltonian(self.hamiltonian_parameters, self.complex_dtype)

    def full_hamiltonian(self) -> torch.Tensor:
        full = torch.zeros(
            (4, 4),
            dtype=self.complex_dtype,
            device=self.hamiltonian_parameters.device,
        )
        full[:2, :2] = self.input_hamiltonian()
        return full

    def rates_and_gamma(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_rates = F.softplus(self.raw_total_rates) + 1e-9
        class1 = torch.sigmoid(self.class1_branch_logits)
        squared_gamma = torch.stack(
            (total_rates * (1.0 - class1), total_rates * class1), dim=0
        )
        return total_rates, class1, torch.sqrt(squared_gamma).to(self.complex_dtype)

    def regularization(self) -> torch.Tensor:
        total_rates, _, _ = self.rates_and_gamma()
        return 1e-5 * total_rates.square().mean() + 1e-6 * self.hamiltonian_parameters.square().mean()

    def forward(self, rho: torch.Tensor) -> dict[str, torch.Tensor]:
        rho = rho.to(
            device=self.hamiltonian_parameters.device,
            dtype=self.complex_dtype,
        )
        embedded = _embed_qubit_density(rho)
        hamiltonian = self.full_hamiltonian()
        evolved = _unitary_evolve(embedded, hamiltonian, self.coherent_time)
        _, _, gamma = self.rates_and_gamma()

        if self.coherent_during_dissipation:
            dt = self.dissipative_time / self.dissipative_steps
            for _ in range(self.dissipative_steps):
                evolved = _unitary_evolve(evolved, hamiltonian, 0.5 * dt)
                evolved = _structured_dissipative_step(evolved, gamma, dt)
                evolved = _unitary_evolve(evolved, hamiltonian, 0.5 * dt)
        else:
            evolved = _structured_dissipative_step(
                evolved, gamma, self.dissipative_time
            )

        diagonal = torch.diagonal(evolved, dim1=-2, dim2=-1).real
        probabilities = diagonal[..., 2:4]
        output_mass = probabilities.sum(dim=-1)
        leakage = 1.0 - output_mass
        return {
            "probabilities": probabilities,
            "output_mass": output_mass,
            "leakage": leakage,
            "rho_out": evolved,
        }


class UnitaryQubitDiscriminator(nn.Module):
    """Trainable projective-measurement baseline with no output leakage."""

    def __init__(
        self,
        coherent_time: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.coherent_time = float(coherent_time)
        self.real_dtype = dtype
        self.complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.hamiltonian_parameters = nn.Parameter(
            0.05 * torch.randn(3, dtype=dtype, device=device)
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def regularization(self) -> torch.Tensor:
        return 1e-6 * self.hamiltonian_parameters.square().mean()

    def forward(self, rho: torch.Tensor) -> dict[str, torch.Tensor]:
        rho = rho.to(
            device=self.hamiltonian_parameters.device,
            dtype=self.complex_dtype,
        )
        hamiltonian = _bloch_hamiltonian(
            self.hamiltonian_parameters, self.complex_dtype
        )
        evolved = _unitary_evolve(rho, hamiltonian, self.coherent_time)
        probabilities = torch.diagonal(evolved, dim1=-2, dim2=-1).real
        output_mass = probabilities.sum(dim=-1)
        return {
            "probabilities": probabilities,
            "output_mass": output_mass,
            "leakage": 1.0 - output_mass,
            "rho_out": evolved,
        }


@torch.no_grad()
def effective_povm(model: nn.Module) -> torch.Tensor:
    """Reconstruct E_0,E_1 such that p(c|rho)=Tr(E_c rho)."""

    parameter = next(model.parameters())
    complex_dtype = torch.complex128 if parameter.dtype == torch.float64 else torch.complex64
    identity, x, y, z = _paulis(complex_dtype, parameter.device)
    maximally_mixed = identity / 2.0
    e0 = model(maximally_mixed)["probabilities"]
    coefficients = []
    for pauli in (x, y, z):
        plus = (identity + pauli) / 2.0
        minus = (identity - pauli) / 2.0
        coefficient = 0.5 * (
            model(plus)["probabilities"] - model(minus)["probabilities"]
        )
        coefficients.append(coefficient)

    effects = []
    for output in range(2):
        effect = e0[output] * identity
        effect = effect + coefficients[0][output] * x
        effect = effect + coefficients[1][output] * y
        effect = effect + coefficients[2][output] * z
        effects.append(0.5 * (effect + effect.mH))
    return torch.stack(effects, dim=0)


@torch.no_grad()
def povm_diagnostics(effects: torch.Tensor) -> dict[str, Any]:
    identity = torch.eye(2, dtype=effects.dtype, device=effects.device)
    leakage_effect = identity - effects.sum(dim=0)
    eig0 = torch.linalg.eigvalsh(effects[0]).real
    eig1 = torch.linalg.eigvalsh(effects[1]).real
    eig_leak = torch.linalg.eigvalsh(leakage_effect).real
    return {
        "effect0_min_eigenvalue": float(eig0.min()),
        "effect0_max_eigenvalue": float(eig0.max()),
        "effect1_min_eigenvalue": float(eig1.min()),
        "effect1_max_eigenvalue": float(eig1.max()),
        "leakage_effect_min_eigenvalue": float(eig_leak.min()),
        "leakage_effect_max_eigenvalue": float(eig_leak.max()),
        "completeness_error_with_leakage": float(
            torch.linalg.matrix_norm(effects.sum(dim=0) + leakage_effect - identity)
        ),
    }
