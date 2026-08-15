"""Small, parameter-matched discriminators for fixed mixed-state benchmarks."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import qsw
from .generators import _cnot_permutation, _ry, _rz
from .objectives import output_statistics, partition_output_statistics


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _exact_layer_dissipation(
    rho: torch.Tensor,
    rates: torch.Tensor,
    source_start: int,
    target_start: int,
    duration: float,
) -> torch.Tensor:
    """Exact dissipative step for jumps |target><source| between disjoint layers."""
    if duration == 0:
        return rho
    squeeze = rho.dim() == 2
    state = rho.unsqueeze(0) if squeeze else rho
    if rates.dim() == 2:
        rates = rates.unsqueeze(0).expand(state.shape[0], -1, -1)
    if rates.dim() != 3 or rates.shape[0] != state.shape[0]:
        raise ValueError("rates must have shape (targets, sources) or (batch, targets, sources)")
    source_count = rates.shape[2]
    target_count = rates.shape[1]
    total_rates = rates.sum(dim=1)
    damping = torch.ones(
        state.shape[0], state.shape[-1], dtype=state.real.dtype, device=state.device
    )
    damping[:, source_start : source_start + source_count] = torch.exp(
        -0.5 * total_rates * duration
    )
    evolved = state * damping[:, :, None].to(state.dtype) * damping[:, None, :].to(state.dtype)

    source_diagonal = torch.diagonal(
        state[
            :,
            source_start : source_start + source_count,
            source_start : source_start + source_count,
        ],
        dim1=-2,
        dim2=-1,
    ).real
    transferred_fraction = 1.0 - torch.exp(-total_rates * duration)
    branching = rates / total_rates.clamp_min(1e-12).unsqueeze(1)
    gains = torch.einsum(
        "bs,bts->bt",
        (source_diagonal * transferred_fraction).to(state.dtype),
        branching.to(state.dtype),
    )
    for target in range(target_count):
        index = target_start + target
        evolved[:, index, index] = evolved[:, index, index] + gains[:, target]
    return evolved[0] if squeeze else evolved


class LayeredQSNNDiscriminator(nn.Module):
    """A 4-4-2-style coherent/dissipative QSNN discriminator.

    For the default dimensions the model has 22 coherent edge parameters and
    24 positive Lindblad-rate parameters, matching the parameter count quoted
    for the paper's two-qubit entanglement network.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 4,
        coherent_time: float = 1.0,
        input_hidden_time: float = 1.0,
        hidden_output_time: float = 1.0,
        chebyshev_order: int = 64,
        chebyshev_tol: float = 1e-10,
        init_h: float = 0.05,
        target_layer_mass: float = 0.9,
        real_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0 < target_layer_mass < 1:
            raise ValueError("target_layer_mass must be in (0, 1)")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = 2
        self.total_dim = self.input_dim + self.hidden_dim + self.output_dim
        self.hidden_start = self.input_dim
        self.real_index = self.input_dim + self.hidden_dim
        self.fake_index = self.real_index + 1
        self.coherent_time = float(coherent_time)
        self.input_hidden_time = float(input_hidden_time)
        self.hidden_output_time = float(hidden_output_time)
        self.chebyshev_order = int(chebyshev_order)
        self.chebyshev_tol = float(chebyshev_tol)

        # All input-hidden edges plus the complete input subgraph: 16+6=22
        # parameters in the default 4-4-2 network.
        edges = []
        for source in range(self.input_dim):
            for target in range(self.hidden_start, self.hidden_start + self.hidden_dim):
                edges.append((source, target))
        for left in range(self.input_dim):
            for right in range(left + 1, self.input_dim):
                edges.append((left, right))
        self.register_buffer("edge_indices", torch.tensor(edges, dtype=torch.long))
        self.h_raw = nn.Parameter(init_h * torch.randn(len(edges), dtype=real_dtype))

        ih_total = -math.log1p(-target_layer_mass) / max(self.input_hidden_time, 1e-12)
        ho_total = -math.log1p(-target_layer_mass) / max(self.hidden_output_time, 1e-12)
        self.ih_rate_raw = nn.Parameter(
            torch.full(
                (self.hidden_dim, self.input_dim),
                _inverse_softplus(ih_total / self.hidden_dim),
                dtype=real_dtype,
            )
        )
        self.ho_rate_raw = nn.Parameter(
            torch.full(
                (2, self.hidden_dim),
                _inverse_softplus(ho_total / 2.0),
                dtype=real_dtype,
            )
        )

    @property
    def complex_dtype(self):
        return torch.complex128 if self.h_raw.dtype == torch.float64 else torch.complex64

    def hamiltonian(self) -> torch.Tensor:
        h = torch.zeros(
            self.total_dim,
            self.total_dim,
            dtype=self.h_raw.dtype,
            device=self.h_raw.device,
        )
        left, right = self.edge_indices[:, 0], self.edge_indices[:, 1]
        h[left, right] = self.h_raw
        h[right, left] = self.h_raw
        return h.to(self.complex_dtype)

    def rates(self) -> tuple[torch.Tensor, torch.Tensor]:
        return F.softplus(self.ih_rate_raw) + 1e-7, F.softplus(self.ho_rate_raw) + 1e-7

    def embed_input(self, rho: torch.Tensor) -> torch.Tensor:
        if rho.shape[-2:] != (self.input_dim, self.input_dim):
            raise ValueError("input density matrix has the wrong dimension")
        return F.pad(rho.to(device=self.h_raw.device, dtype=self.complex_dtype), (0, self.total_dim - self.input_dim, 0, self.total_dim - self.input_dim))

    def forward(self, rho: torch.Tensor) -> dict[str, torch.Tensor]:
        embedded = self.embed_input(rho)
        coherent = qsw.evolve_density_chebyshev(
            embedded,
            self.hamiltonian(),
            self.coherent_time,
            max_order=self.chebyshev_order,
            tol=self.chebyshev_tol,
        )
        ih_rates, ho_rates = self.rates()
        hidden = _exact_layer_dissipation(
            coherent, ih_rates, 0, self.hidden_start, self.input_hidden_time
        )
        output = _exact_layer_dissipation(
            hidden, ho_rates, self.hidden_start, self.real_index, self.hidden_output_time
        )
        return {
            "rho_in": embedded,
            "rho_coherent": coherent,
            "rho_hidden": hidden,
            "rho_out": output,
            "input_hidden_rates": ih_rates,
            "hidden_output_rates": ho_rates,
            **output_statistics(output, self.real_index, self.fake_index),
        }


def _full_single_qubit_gate(gate: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    identity = torch.eye(2, dtype=gate.dtype, device=gate.device)
    result = None
    for index in range(n_qubits):
        factor = gate if index == qubit else identity
        result = factor if result is None else torch.kron(result.contiguous(), factor.contiguous())
    return result


def _batched_full_single_qubit_gate(
    gate: torch.Tensor, qubit: int, n_qubits: int
) -> torch.Tensor:
    batch = gate.shape[0]
    identity = torch.eye(2, dtype=gate.dtype, device=gate.device).expand(batch, -1, -1)
    result = None
    for index in range(n_qubits):
        factor = gate if index == qubit else identity
        if result is None:
            result = factor
        else:
            left_dim = result.shape[-1]
            result = torch.einsum("bij,bkl->bikjl", result, factor).reshape(
                batch, left_dim * 2, left_dim * 2
            )
    return result


class AncillaVQCDiscriminator(nn.Module):
    """Parameter-matched unitary baseline with one readout ancilla."""

    def __init__(
        self,
        system_qubits: int = 2,
        n_layers: int = 8,
        real_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.system_qubits = int(system_qubits)
        self.n_qubits = self.system_qubits + 1
        self.input_dim = 1 << self.system_qubits
        self.total_dim = 1 << self.n_qubits
        self.readout_split = self.total_dim // 2
        self.n_layers = int(n_layers)
        self.ry_angles = nn.Parameter(0.05 * torch.randn(n_layers, self.n_qubits, dtype=real_dtype))
        self.rz_angles = nn.Parameter(0.05 * torch.randn(n_layers, self.n_qubits, dtype=real_dtype))

    @property
    def complex_dtype(self):
        return torch.complex128 if self.ry_angles.dtype == torch.float64 else torch.complex64

    def unitary(self) -> torch.Tensor:
        unitary = torch.eye(self.total_dim, dtype=self.complex_dtype, device=self.ry_angles.device)
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                ry = _ry(self.ry_angles[layer, qubit], self.complex_dtype)
                rz = _rz(self.rz_angles[layer, qubit], self.complex_dtype)
                unitary = _full_single_qubit_gate(rz, qubit, self.n_qubits) @ _full_single_qubit_gate(ry, qubit, self.n_qubits) @ unitary
            for control in range(self.n_qubits):
                target = (control + 1) % self.n_qubits
                permutation = _cnot_permutation(self.n_qubits, control, target, unitary.device)
                permutation_matrix = torch.eye(self.total_dim, dtype=self.complex_dtype, device=unitary.device)[:, permutation]
                unitary = permutation_matrix @ unitary
        return unitary

    def embed_input(self, rho: torch.Tensor) -> torch.Tensor:
        if rho.shape[-2:] != (self.input_dim, self.input_dim):
            raise ValueError("input density matrix has the wrong dimension")
        output_shape = (*rho.shape[:-2], self.total_dim, self.total_dim)
        embedded = torch.zeros(output_shape, dtype=self.complex_dtype, device=self.ry_angles.device)
        embedded[..., : self.input_dim, : self.input_dim] = rho.to(self.complex_dtype)
        return embedded

    def forward(self, rho: torch.Tensor) -> dict[str, torch.Tensor]:
        embedded = self.embed_input(rho)
        unitary = self.unitary()
        output = unitary @ embedded @ unitary.mH
        return {
            "rho_in": embedded,
            "rho_coherent": output,
            "rho_out": output,
            **partition_output_statistics(output, self.readout_split),
        }


class ConditionalLayeredQSNNDiscriminator(LayeredQSNNDiscriminator):
    """4-4-2 QSNN whose Hamiltonian and jump rates depend on a scalar condition."""

    def __init__(self, *args, coherent_backend: str = "exact", **kwargs):
        super().__init__(*args, **kwargs)
        if coherent_backend != "exact":
            raise ValueError("conditional QSNN currently supports coherent_backend='exact'")
        self.coherent_backend = coherent_backend
        self.h_condition = nn.Parameter(torch.zeros_like(self.h_raw))
        self.ih_rate_condition = nn.Parameter(torch.zeros_like(self.ih_rate_raw))
        self.ho_rate_condition = nn.Parameter(torch.zeros_like(self.ho_rate_raw))

    def _conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(
            conditions,
            device=self.h_raw.device,
            dtype=self.h_raw.dtype,
        ).reshape(-1)
        if not bool(((values >= 0) & (values <= 1)).all()):
            raise ValueError("conditions must lie in [0, 1]")
        return 2.0 * values - 1.0

    def conditional_hamiltonian(self, normalized: torch.Tensor) -> torch.Tensor:
        weights = self.h_raw.unsqueeze(0) + normalized[:, None] * self.h_condition.unsqueeze(0)
        batch = normalized.shape[0]
        h = torch.zeros(
            batch,
            self.total_dim,
            self.total_dim,
            dtype=self.h_raw.dtype,
            device=self.h_raw.device,
        )
        left, right = self.edge_indices[:, 0], self.edge_indices[:, 1]
        h[:, left, right] = weights
        h[:, right, left] = weights
        return h.to(self.complex_dtype)

    def conditional_rates(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ih = F.softplus(
            self.ih_rate_raw.unsqueeze(0)
            + normalized[:, None, None] * self.ih_rate_condition.unsqueeze(0)
        ) + 1e-7
        ho = F.softplus(
            self.ho_rate_raw.unsqueeze(0)
            + normalized[:, None, None] * self.ho_rate_condition.unsqueeze(0)
        ) + 1e-7
        return ih, ho

    def forward(self, rho: torch.Tensor, conditions: torch.Tensor) -> dict[str, torch.Tensor]:
        squeeze = rho.dim() == 2
        rho_batch = rho.unsqueeze(0) if squeeze else rho
        normalized = self._conditions(conditions)
        if normalized.shape[0] == 1 and rho_batch.shape[0] > 1:
            normalized = normalized.expand(rho_batch.shape[0])
        if normalized.shape[0] != rho_batch.shape[0]:
            raise ValueError("condition batch size must match density-matrix batch size")
        embedded = self.embed_input(rho_batch)
        hamiltonians = self.conditional_hamiltonian(normalized)
        # The small conditional benchmark uses exact unitary evolution. This
        # retains gradients through the condition-dependent H and avoids
        # detached spectral bounds in the Chebyshev implementation.
        unitaries = torch.matrix_exp((-1j) * hamiltonians * self.coherent_time)
        coherent = unitaries @ embedded @ unitaries.mH
        ih_rates, ho_rates = self.conditional_rates(normalized)
        hidden = _exact_layer_dissipation(
            coherent, ih_rates, 0, self.hidden_start, self.input_hidden_time
        )
        output = _exact_layer_dissipation(
            hidden, ho_rates, self.hidden_start, self.real_index, self.hidden_output_time
        )
        result = {
            "rho_in": embedded,
            "rho_coherent": coherent,
            "rho_hidden": hidden,
            "rho_out": output,
            "input_hidden_rates": ih_rates,
            "hidden_output_rates": ho_rates,
            **output_statistics(output, self.real_index, self.fake_index),
        }
        return {key: value[0] for key, value in result.items()} if squeeze else result


class ConditionalAncillaVQCDiscriminator(AncillaVQCDiscriminator):
    """Ancilla-VQC baseline with the same affine condition re-uploading."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ry_condition = nn.Parameter(torch.zeros_like(self.ry_angles))
        self.rz_condition = nn.Parameter(torch.zeros_like(self.rz_angles))

    def _conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(
            conditions,
            device=self.ry_angles.device,
            dtype=self.ry_angles.dtype,
        ).reshape(-1)
        if not bool(((values >= 0) & (values <= 1)).all()):
            raise ValueError("conditions must lie in [0, 1]")
        return 2.0 * values - 1.0

    def conditional_unitary(self, normalized: torch.Tensor) -> torch.Tensor:
        batch = normalized.shape[0]
        unitary = torch.eye(
            self.total_dim, dtype=self.complex_dtype, device=self.ry_angles.device
        ).expand(batch, -1, -1).clone()
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                ry_angle = self.ry_angles[layer, qubit] + normalized * self.ry_condition[layer, qubit]
                rz_angle = self.rz_angles[layer, qubit] + normalized * self.rz_condition[layer, qubit]
                ry = _batched_full_single_qubit_gate(
                    _ry(ry_angle, self.complex_dtype), qubit, self.n_qubits
                )
                rz = _batched_full_single_qubit_gate(
                    _rz(rz_angle, self.complex_dtype), qubit, self.n_qubits
                )
                unitary = rz @ ry @ unitary
            for control in range(self.n_qubits):
                target = (control + 1) % self.n_qubits
                permutation = _cnot_permutation(
                    self.n_qubits, control, target, unitary.device
                )
                unitary = unitary[:, permutation, :]
        return unitary

    def forward(self, rho: torch.Tensor, conditions: torch.Tensor) -> dict[str, torch.Tensor]:
        squeeze = rho.dim() == 2
        rho_batch = rho.unsqueeze(0) if squeeze else rho
        normalized = self._conditions(conditions)
        if normalized.shape[0] == 1 and rho_batch.shape[0] > 1:
            normalized = normalized.expand(rho_batch.shape[0])
        if normalized.shape[0] != rho_batch.shape[0]:
            raise ValueError("condition batch size must match density-matrix batch size")
        embedded = self.embed_input(rho_batch)
        unitary = self.conditional_unitary(normalized)
        output = unitary @ embedded @ unitary.mH
        result = {
            "rho_in": embedded,
            "rho_coherent": output,
            "rho_out": output,
            **partition_output_statistics(output, self.readout_split),
        }
        return {key: value[0] for key, value in result.items()} if squeeze else result
