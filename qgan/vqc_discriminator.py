"""Unitary discriminator baseline with the same density-matrix contract as QSNN."""

import torch
import torch.nn as nn

import qsw
from .objectives import partition_output_statistics


class VQCDiscriminator(nn.Module):
    """
    Trainable unitary discriminator used as the standard-QGAN baseline.

    A trainable unitary acts on the original input Hilbert space.  The most
    significant qubit is then measured: its zero and one subspaces are the
    complete ``real`` and ``fake`` outcomes.  Unlike two rank-one output nodes,
    this measurement has no structural leakage.  The Trace-Z objective remains
    identical to the QSNN discriminator objective.
    """

    SUPPORTED_BACKENDS = {"chebyshev", "suzuki", "exact"}

    def __init__(
        self,
        input_dim: int,
        evolution_time: float = 1.0,
        backend: str = "chebyshev",
        chebyshev_order: int = 128,
        chebyshev_tol: float = 1e-10,
        suzuki_steps: int = 12,
        suzuki_order: int = 2,
        init_h: float = 0.02,
        real_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if input_dim < 2 or input_dim % 2 != 0:
            raise ValueError("input_dim must be even for a binary readout-qubit measurement")
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}")
        if chebyshev_order <= 0 or suzuki_steps <= 0:
            raise ValueError("evolution orders and step counts must be positive")

        self.input_dim = int(input_dim)
        self.total_dim = self.input_dim
        self.readout_split = self.input_dim // 2
        self.real_index = None
        self.fake_index = None
        self.evolution_time = float(evolution_time)
        self.backend = backend
        self.chebyshev_order = int(chebyshev_order)
        self.chebyshev_tol = float(chebyshev_tol)
        self.suzuki_steps = int(suzuki_steps)
        self.suzuki_order = int(suzuki_order)

        self.H_raw = nn.Parameter(
            init_h * torch.randn(self.total_dim, self.total_dim, dtype=real_dtype)
        )

    @property
    def complex_dtype(self):
        return torch.complex128 if self.H_raw.dtype == torch.float64 else torch.complex64

    def hamiltonian(self) -> torch.Tensor:
        return (0.5 * (self.H_raw + self.H_raw.T)).to(self.complex_dtype)

    def embed_input(self, rho_in: torch.Tensor) -> torch.Tensor:
        if rho_in.shape[-2:] != (self.input_dim, self.input_dim):
            raise ValueError(
                f"rho_in must end in ({self.input_dim}, {self.input_dim}), "
                f"got {tuple(rho_in.shape)}"
            )
        return rho_in.to(device=self.H_raw.device, dtype=self.complex_dtype)

    def _evolve(self, rho: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        if self.evolution_time == 0:
            return rho
        if self.backend == "chebyshev":
            return qsw.evolve_density_chebyshev(
                rho,
                H,
                self.evolution_time,
                max_order=self.chebyshev_order,
                tol=self.chebyshev_tol,
            )
        if self.backend == "suzuki":
            return qsw.evolve_density_suzuki(
                rho,
                H,
                self.evolution_time,
                steps=self.suzuki_steps,
                order=self.suzuki_order,
            )
        return qsw.evolve_unitary(rho, H, self.evolution_time)

    def forward(self, rho_in: torch.Tensor):
        rho0 = self.embed_input(rho_in)
        H = self.hamiltonian()
        rho_out = self._evolve(rho0, H)
        return {
            "rho_in": rho0,
            "rho_coherent": rho_out,
            "rho_out": rho_out,
            **partition_output_statistics(rho_out, self.readout_split),
        }

    def forward_state(self, state_in: torch.Tensor):
        """Apply the unitary discriminator directly to pure state vectors."""
        squeeze_back = state_in.dim() == 1 or (
            state_in.dim() == 2 and state_in.shape == (self.input_dim, 1)
        )
        if state_in.dim() == 1:
            state = state_in.unsqueeze(0)
        elif state_in.dim() == 2 and state_in.shape == (self.input_dim, 1):
            state = state_in[:, 0].unsqueeze(0)
        elif state_in.dim() == 2 and state_in.shape[-1] == self.input_dim:
            state = state_in
        elif state_in.dim() == 3 and state_in.shape[-2:] == (self.input_dim, 1):
            state = state_in[..., 0]
        else:
            raise ValueError(
                f"state_in must end in dimension {self.input_dim}; got {tuple(state_in.shape)}"
            )
        state = state.to(device=self.H_raw.device, dtype=self.complex_dtype)
        original_state = state
        H = self.hamiltonian()
        columns = state.unsqueeze(-1)
        if self.evolution_time == 0:
            evolved = columns
        elif self.backend == "chebyshev":
            evolved = qsw.evolve_state_chebyshev(
                columns,
                H,
                self.evolution_time,
                max_order=self.chebyshev_order,
                tol=self.chebyshev_tol,
            )
        elif self.backend == "suzuki":
            evolved = qsw.evolve_state_suzuki(
                columns,
                H,
                self.evolution_time,
                steps=self.suzuki_steps,
                order=self.suzuki_order,
            )
        else:
            evolved = qsw.evolve_state_exact(columns, H, self.evolution_time)
        state_out = evolved.squeeze(-1)
        probabilities = state_out.abs().square()
        p_real = probabilities[..., : self.readout_split].sum(dim=-1)
        p_fake = probabilities[..., self.readout_split :].sum(dim=-1)
        output_mass = p_real + p_fake
        normalized = torch.stack([p_real, p_fake], dim=-1)
        normalized = normalized / output_mass.unsqueeze(-1).clamp_min(1e-12)
        result = {
            "state_in": original_state,
            "state_coherent": state_out,
            "state_out": state_out,
            "p_real": p_real,
            "p_fake": p_fake,
            "output_mass": output_mass,
            "leakage": 1.0 - output_mass,
            "z_expectation": p_real - p_fake,
            "normalized_probs": normalized,
            "state_trace": output_mass,
        }
        if squeeze_back:
            result = {key: value[0] for key, value in result.items()}
        return result
