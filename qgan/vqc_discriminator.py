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
