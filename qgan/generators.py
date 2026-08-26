"""Small differentiable state-vector generators for QSNN-QGAN experiments."""

import math

import torch
import torch.nn as nn


def _ry(angles: torch.Tensor, complex_dtype: torch.dtype) -> torch.Tensor:
    half = 0.5 * angles
    cosine = torch.cos(half)
    sine = torch.sin(half)
    row0 = torch.stack([cosine, -sine], dim=-1)
    row1 = torch.stack([sine, cosine], dim=-1)
    return torch.stack([row0, row1], dim=-2).to(complex_dtype)


def _rz(angles: torch.Tensor, complex_dtype: torch.dtype) -> torch.Tensor:
    half = 0.5 * angles.to(complex_dtype)
    zeros = torch.zeros_like(half)
    phase0 = torch.exp(-1j * half)
    phase1 = torch.exp(1j * half)
    row0 = torch.stack([phase0, zeros], dim=-1)
    row1 = torch.stack([zeros, phase1], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _apply_single_qubit_gate(
    state: torch.Tensor,
    gate: torch.Tensor,
    qubit: int,
    n_qubits: int,
) -> torch.Tensor:
    batch = state.shape[0]
    left = 1 << qubit
    right = 1 << (n_qubits - qubit - 1)
    pairs = state.reshape(batch, left, 2, right).permute(0, 1, 3, 2)
    if gate.dim() == 2:
        gate = gate.unsqueeze(0).expand(batch, -1, -1)
    updated = torch.einsum("bij,blrj->blri", gate, pairs)
    return updated.permute(0, 1, 3, 2).reshape(batch, -1)


def _cnot_permutation(n_qubits: int, control: int, target: int, device) -> torch.Tensor:
    indices = torch.arange(1 << n_qubits, device=device)
    control_mask = 1 << (n_qubits - control - 1)
    target_mask = 1 << (n_qubits - target - 1)
    return torch.where((indices & control_mask) != 0, indices ^ target_mask, indices)


class PQCGenerator(nn.Module):
    """RY/RZ plus ring-CNOT pure-state generator implemented directly in PyTorch."""

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        noise_dim: int | None = None,
        conditional: bool = False,
        noise_reuploading: bool = False,
        alternating_entanglement: bool = False,
        canonicalize_output: bool = False,
        real_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if n_qubits <= 0 or n_layers <= 0:
            raise ValueError("n_qubits and n_layers must be positive")
        self.n_qubits = int(n_qubits)
        self.state_dim = 1 << self.n_qubits
        self.n_layers = int(n_layers)
        self.noise_dim = int(noise_dim or n_qubits)
        self.conditional = bool(conditional)
        self.noise_reuploading = bool(noise_reuploading)
        self.alternating_entanglement = bool(alternating_entanglement)
        self.canonicalize_output = bool(canonicalize_output)

        self.noise_projection = nn.Linear(
            self.noise_dim,
            self.n_qubits,
            bias=True,
            dtype=real_dtype,
        )
        self.ry_angles = nn.Parameter(
            0.05 * torch.randn(self.n_layers, self.n_qubits, dtype=real_dtype)
        )
        self.rz_angles = nn.Parameter(
            0.05 * torch.randn(self.n_layers, self.n_qubits, dtype=real_dtype)
        )
        self.layer_noise_projections = (
            nn.ModuleList(
                [
                    nn.Linear(
                        self.noise_dim,
                        2 * self.n_qubits,
                        bias=True,
                        dtype=real_dtype,
                    )
                    for _ in range(self.n_layers)
                ]
            )
            if self.noise_reuploading
            else None
        )
        self.label_angles = (
            nn.Embedding(2, self.n_qubits, dtype=real_dtype) if self.conditional else None
        )
        if self.label_angles is not None:
            nn.init.zeros_(self.label_angles.weight)

    @property
    def complex_dtype(self):
        return torch.complex128 if self.ry_angles.dtype == torch.float64 else torch.complex64

    def _initial_state(self, batch: int) -> torch.Tensor:
        state = torch.zeros(
            batch,
            self.state_dim,
            device=self.ry_angles.device,
            dtype=self.complex_dtype,
        )
        state[:, 0] = 1.0
        return state

    def _ring_entangle(self, state: torch.Tensor, *, reverse: bool = False) -> torch.Tensor:
        for control in range(self.n_qubits):
            target = (control - 1) % self.n_qubits if reverse else (control + 1) % self.n_qubits
            permutation = _cnot_permutation(
                self.n_qubits,
                control,
                target,
                state.device,
            )
            state = state[:, permutation]
        return state

    def statevector(self, noise: torch.Tensor, labels: torch.Tensor | None = None):
        """Generate normalized state vectors without forming density matrices."""
        if noise.dim() == 1:
            noise = noise.unsqueeze(0)
        if noise.shape[-1] != self.noise_dim:
            raise ValueError(
                f"noise must end in dimension {self.noise_dim}, got {tuple(noise.shape)}"
            )
        noise = noise.to(device=self.ry_angles.device, dtype=self.ry_angles.dtype)
        batch = noise.shape[0]
        state = self._initial_state(batch)

        encoded_angles = math.pi * torch.tanh(self.noise_projection(noise))
        if self.conditional:
            if labels is None:
                raise ValueError("conditional generator requires labels")
            labels = labels.reshape(-1).to(device=state.device, dtype=torch.long)
            if labels.shape[0] != batch:
                raise ValueError("labels batch size must match noise batch size")
            encoded_angles = encoded_angles + self.label_angles(labels)

        for qubit in range(self.n_qubits):
            state = _apply_single_qubit_gate(
                state,
                _ry(encoded_angles[:, qubit], self.complex_dtype),
                qubit,
                self.n_qubits,
            )

        for layer in range(self.n_layers):
            if self.layer_noise_projections is not None:
                reuploaded = math.pi * torch.tanh(self.layer_noise_projections[layer](noise))
                noise_ry, noise_rz = reuploaded.chunk(2, dim=-1)
            else:
                noise_ry = noise_rz = None
            for qubit in range(self.n_qubits):
                ry_angle = self.ry_angles[layer, qubit]
                rz_angle = self.rz_angles[layer, qubit]
                if noise_ry is not None and noise_rz is not None:
                    ry_angle = ry_angle + noise_ry[:, qubit]
                    rz_angle = rz_angle + noise_rz[:, qubit]
                state = _apply_single_qubit_gate(
                    state,
                    _ry(ry_angle, self.complex_dtype),
                    qubit,
                    self.n_qubits,
                )
                state = _apply_single_qubit_gate(
                    state,
                    _rz(rz_angle, self.complex_dtype),
                    qubit,
                    self.n_qubits,
                )
            state = self._ring_entangle(
                state,
                reverse=self.alternating_entanglement and layer % 2 == 1,
            )

        state = state / torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(1e-12)
        if self.canonicalize_output:
            # The image Decoder and real-data encoder only expose computational-
            # basis probabilities.  Canonicalizing to non-negative real
            # amplitudes prevents the discriminator from exploiting arbitrary
            # generator phases that carry no image information.
            state = state.abs().to(self.complex_dtype)
            state = state / torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(1e-12)
        return state

    def forward(self, noise: torch.Tensor, labels: torch.Tensor | None = None):
        state = self.statevector(noise, labels=labels)
        psi = state.unsqueeze(-1)
        rho = psi @ psi.mH
        return rho


class PurifiedPQCGenerator(nn.Module):
    """Generate a fixed mixed state through a trainable purification.

    The first ``system_qubits`` are retained and the remaining qubits are
    traced out.  Starting from a pure state on system plus environment is a
    convenient differentiable parameterization of a physical mixed state; two
    environment qubits are sufficient for an arbitrary two-qubit density
    matrix.
    """

    def __init__(
        self,
        system_qubits: int = 2,
        environment_qubits: int = 2,
        n_layers: int = 6,
        alternating_entanglement: bool = True,
        real_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if system_qubits <= 0 or environment_qubits <= 0 or n_layers <= 0:
            raise ValueError("qubit counts and n_layers must be positive")
        self.system_qubits = int(system_qubits)
        self.environment_qubits = int(environment_qubits)
        self.n_qubits = self.system_qubits + self.environment_qubits
        self.system_dim = 1 << self.system_qubits
        self.environment_dim = 1 << self.environment_qubits
        self.state_dim = 1 << self.n_qubits
        self.n_layers = int(n_layers)
        self.alternating_entanglement = bool(alternating_entanglement)
        self.ry_angles = nn.Parameter(
            0.05 * torch.randn(self.n_layers, self.n_qubits, dtype=real_dtype)
        )
        self.rz_angles = nn.Parameter(
            0.05 * torch.randn(self.n_layers, self.n_qubits, dtype=real_dtype)
        )

    @property
    def complex_dtype(self):
        return torch.complex128 if self.ry_angles.dtype == torch.float64 else torch.complex64

    def statevector(self) -> torch.Tensor:
        state = torch.zeros(
            1,
            self.state_dim,
            device=self.ry_angles.device,
            dtype=self.complex_dtype,
        )
        state[:, 0] = 1.0
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                state = _apply_single_qubit_gate(
                    state,
                    _ry(self.ry_angles[layer, qubit], self.complex_dtype),
                    qubit,
                    self.n_qubits,
                )
                state = _apply_single_qubit_gate(
                    state,
                    _rz(self.rz_angles[layer, qubit], self.complex_dtype),
                    qubit,
                    self.n_qubits,
                )
            reverse = self.alternating_entanglement and layer % 2 == 1
            for control in range(self.n_qubits):
                target = (
                    (control - 1) % self.n_qubits
                    if reverse
                    else (control + 1) % self.n_qubits
                )
                permutation = _cnot_permutation(
                    self.n_qubits,
                    control,
                    target,
                    state.device,
                )
                state = state[:, permutation]
        state = state.squeeze(0)
        return state / torch.linalg.vector_norm(state).clamp_min(1e-12)

    def forward(self) -> torch.Tensor:
        purification = self.statevector().reshape(self.system_dim, self.environment_dim)
        return purification @ purification.mH


class ConditionalPurifiedPQCGenerator(nn.Module):
    """Conditional mixed-state generator with layer-wise condition re-uploading."""

    def __init__(
        self,
        system_qubits: int = 2,
        environment_qubits: int = 2,
        n_layers: int = 6,
        condition_dim: int = 1,
        condition_feature_map: str = "linear",
        alternating_entanglement: bool = True,
        real_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if system_qubits <= 0 or environment_qubits <= 0 or n_layers <= 0 or condition_dim <= 0:
            raise ValueError("qubit counts, n_layers, and condition_dim must be positive")
        self.system_qubits = int(system_qubits)
        self.environment_qubits = int(environment_qubits)
        self.n_qubits = self.system_qubits + self.environment_qubits
        self.system_dim = 1 << self.system_qubits
        self.environment_dim = 1 << self.environment_qubits
        self.state_dim = 1 << self.n_qubits
        self.n_layers = int(n_layers)
        self.condition_dim = int(condition_dim)
        if condition_feature_map not in {"linear", "quadratic", "equivariant"}:
            raise ValueError(
                "condition_feature_map must be 'linear', 'quadratic', or 'equivariant'"
            )
        if self.condition_dim == 1 and condition_feature_map != "linear":
            raise ValueError("scalar conditions only support the linear feature map")
        self.condition_feature_map = condition_feature_map
        self.condition_feature_dim = (
            self.condition_dim
            if condition_feature_map == "linear"
            else (
                self.condition_dim + self.condition_dim * (self.condition_dim + 1) // 2
                if condition_feature_map == "quadratic"
                else 1
            )
        )
        self.alternating_entanglement = bool(alternating_entanglement)
        shape = (self.n_layers, self.n_qubits)
        condition_shape = (
            shape if self.condition_dim == 1 else (*shape, self.condition_feature_dim)
        )
        self.ry_angles = nn.Parameter(0.05 * torch.randn(*shape, dtype=real_dtype))
        self.rz_angles = nn.Parameter(0.05 * torch.randn(*shape, dtype=real_dtype))
        self.ry_condition = nn.Parameter(0.05 * torch.randn(*condition_shape, dtype=real_dtype))
        self.rz_condition = nn.Parameter(0.05 * torch.randn(*condition_shape, dtype=real_dtype))

    @property
    def complex_dtype(self):
        return torch.complex128 if self.ry_angles.dtype == torch.float64 else torch.complex64

    def _conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(
            conditions,
            device=self.ry_angles.device,
            dtype=self.ry_angles.dtype,
        )
        if self.condition_dim == 1:
            values = values.reshape(-1, 1)
            if not bool(((values >= 0) & (values <= 1)).all()):
                raise ValueError("scalar conditions must lie in [0, 1]")
            return 2.0 * values - 1.0
        if values.dim() == 1:
            values = values.unsqueeze(0)
        if values.dim() != 2 or values.shape[-1] != self.condition_dim:
            raise ValueError(
                f"conditions must have shape (batch, {self.condition_dim})"
            )
        if not bool(((values[:, 0] >= 0) & (values[:, 0] <= 1)).all()):
            raise ValueError("the first condition component must lie in [0, 1]")
        if not bool(((values[:, 1:] >= -1) & (values[:, 1:] <= 1)).all()):
            raise ValueError("remaining condition components must lie in [-1, 1]")
        normalized = values.clone()
        normalized[:, 0] = 2.0 * normalized[:, 0] - 1.0
        return normalized

    def _condition_features(self, normalized: torch.Tensor) -> torch.Tensor:
        if self.condition_feature_map == "linear":
            return normalized
        if self.condition_feature_map == "equivariant":
            return normalized[:, :1]
        products = [
            normalized[:, left] * normalized[:, right]
            for left in range(self.condition_dim)
            for right in range(left, self.condition_dim)
        ]
        return torch.cat([normalized, torch.stack(products, dim=-1)], dim=-1)

    def _conditional_angles(
        self, normalized: torch.Tensor, base: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        if self.condition_dim == 1:
            return base.unsqueeze(0) + normalized[:, :, None] * weights.unsqueeze(0)
        features = self._condition_features(normalized)
        return base.unsqueeze(0) + torch.einsum("bc,lqc->blq", features, weights)

    def statevector(self, conditions: torch.Tensor) -> torch.Tensor:
        normalized = self._conditions(conditions)
        if self.condition_feature_map == "equivariant":
            unique_p, inverse = torch.unique(normalized[:, 0], sorted=True, return_inverse=True)
            circuit_conditions = torch.zeros(
                unique_p.shape[0],
                self.condition_dim,
                device=normalized.device,
                dtype=normalized.dtype,
            )
            circuit_conditions[:, 0] = unique_p
        else:
            circuit_conditions = normalized
            inverse = None
        batch = circuit_conditions.shape[0]
        ry_angles = self._conditional_angles(
            circuit_conditions, self.ry_angles, self.ry_condition
        )
        rz_angles = self._conditional_angles(
            circuit_conditions, self.rz_angles, self.rz_condition
        )
        state = torch.zeros(
            batch,
            self.state_dim,
            device=self.ry_angles.device,
            dtype=self.complex_dtype,
        )
        state[:, 0] = 1.0
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                ry_angle = ry_angles[:, layer, qubit]
                rz_angle = rz_angles[:, layer, qubit]
                state = _apply_single_qubit_gate(
                    state,
                    _ry(ry_angle, self.complex_dtype),
                    qubit,
                    self.n_qubits,
                )
                state = _apply_single_qubit_gate(
                    state,
                    _rz(rz_angle, self.complex_dtype),
                    qubit,
                    self.n_qubits,
                )
            reverse = self.alternating_entanglement and layer % 2 == 1
            for control in range(self.n_qubits):
                target = (
                    (control - 1) % self.n_qubits
                    if reverse
                    else (control + 1) % self.n_qubits
                )
                permutation = _cnot_permutation(
                    self.n_qubits,
                    control,
                    target,
                    state.device,
                )
                state = state[:, permutation]
        state = state / torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(1e-12)
        if self.condition_feature_map == "equivariant":
            # The task supplies a known relative local rotation.  Building that
            # symmetry into the common generator prevents rotation-pool
            # memorization while leaving the p-dependent mixed state trainable.
            from .rotations import quaternion_to_su2

            state = state[inverse]
            rotation = quaternion_to_su2(normalized[:, 1:]).to(
                device=state.device, dtype=self.complex_dtype
            )
            state = _apply_single_qubit_gate(
                state,
                rotation,
                self.system_qubits - 1,
                self.n_qubits,
            )
        return state

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        state = self.statevector(conditions)
        purification = state.reshape(-1, self.system_dim, self.environment_dim)
        return purification @ purification.mH
