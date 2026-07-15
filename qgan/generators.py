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

    def forward(self, noise: torch.Tensor, labels: torch.Tensor | None = None):
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
        psi = state.unsqueeze(-1)
        rho = psi @ psi.mH
        return rho
