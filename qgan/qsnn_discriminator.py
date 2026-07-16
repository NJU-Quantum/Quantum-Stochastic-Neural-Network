"""Thin density-matrix discriminator wrapper around the existing QSNN backend."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import qsw
from .objectives import output_statistics


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus initialization value must be positive")
    return math.log(math.expm1(value))


class QSNNDiscriminator(nn.Module):
    """
    QSNN real/fake discriminator accepting density matrices directly.

    Jump parameters are represented as a positive total physical rate for each
    input node plus a softmax real/fake branching probability.  This separates
    output mass from classification direction.  ``target_output_mass`` chooses
    the initial total rate analytically, preventing an inconclusive initial
    discriminator.  ``gamma_semantics`` only controls the backward-compatible
    ``init_gamma`` fallback when no target mass is supplied.
    """

    SUPPORTED_BACKENDS = {
        "cheby_suzuki",
        "suzuki_global",
        "exact_split",
        "exact_rk4",
    }
    SUPPORTED_ABLATIONS = {"full", "h_only", "l_only"}

    def __init__(
        self,
        input_dim: int,
        coherent_time: float = 1.0,
        dissipative_time: float = 1.0,
        backend: str = "cheby_suzuki",
        stage2_steps: int = 12,
        chebyshev_order: int = 128,
        chebyshev_tol: float = 1e-10,
        suzuki_steps: int = 12,
        suzuki_order: int = 2,
        init_h: float = 0.02,
        init_gamma: float = 0.1,
        target_output_mass: float | None = 0.8,
        min_positive: float = 1e-6,
        gamma_semantics: str = "amplitude",
        ablation: str = "full",
        real_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}")
        if gamma_semantics not in {"amplitude", "rate"}:
            raise ValueError("gamma_semantics must be 'amplitude' or 'rate'")
        if ablation not in self.SUPPORTED_ABLATIONS:
            raise ValueError(f"Unsupported ablation: {ablation}")
        if stage2_steps <= 0 or suzuki_steps <= 0:
            raise ValueError("evolution step counts must be positive")
        if target_output_mass is not None and not 0.0 < target_output_mass < 1.0:
            raise ValueError("target_output_mass must be strictly between 0 and 1")
        if target_output_mass is not None and dissipative_time <= 0:
            raise ValueError("positive dissipative_time is required with target_output_mass")

        self.input_dim = int(input_dim)
        self.total_dim = self.input_dim + 2
        self.real_index = self.input_dim
        self.fake_index = self.input_dim + 1
        self.coherent_time = float(coherent_time)
        self.dissipative_time = float(dissipative_time)
        self.backend = backend
        self.stage2_steps = int(stage2_steps)
        self.chebyshev_order = int(chebyshev_order)
        self.chebyshev_tol = float(chebyshev_tol)
        self.suzuki_steps = int(suzuki_steps)
        self.suzuki_order = int(suzuki_order)
        self.min_positive = float(min_positive)
        self.gamma_semantics = gamma_semantics
        self.target_output_mass = target_output_mass
        self.ablation = ablation

        self.H_raw = nn.Parameter(init_h * torch.randn(input_dim, input_dim, dtype=real_dtype))
        if target_output_mass is not None:
            initial_total_rate = -math.log1p(-target_output_mass) / self.dissipative_time
        elif gamma_semantics == "amplitude":
            initial_total_rate = 2.0 * init_gamma**2
        else:
            initial_total_rate = 2.0 * init_gamma
        self.total_rate_raw = nn.Parameter(
            torch.full(
                (input_dim,),
                _inverse_softplus(initial_total_rate),
                dtype=real_dtype,
            )
        )
        # Keep the historical name gamma_raw for compatibility with gradient
        # inspection code. It now stores real/fake branch logits.
        self.gamma_raw = nn.Parameter(0.01 * torch.randn(2, input_dim, dtype=real_dtype))

    @property
    def complex_dtype(self):
        return torch.complex128 if self.H_raw.dtype == torch.float64 else torch.complex64

    def hamiltonian(self) -> torch.Tensor:
        h = self.input_hamiltonian()
        return F.pad(h, (0, 2, 0, 2))

    def input_hamiltonian(self) -> torch.Tensor:
        h = 0.5 * (self.H_raw + self.H_raw.T)
        if self.ablation == "l_only":
            h = torch.zeros_like(h)
        return h.to(self.complex_dtype)

    def total_rates(self) -> torch.Tensor:
        rates = F.softplus(self.total_rate_raw) + self.min_positive
        if self.ablation == "h_only":
            rates = torch.zeros_like(rates)
        return rates

    def branch_probabilities(self) -> torch.Tensor:
        return torch.softmax(self.gamma_raw, dim=0)

    def effective_rates(self) -> torch.Tensor:
        return self.branch_probabilities() * self.total_rates().unsqueeze(0)

    def jump_amplitudes(self) -> torch.Tensor:
        return torch.sqrt(self.effective_rates()).to(self.complex_dtype)

    def embed_input(self, rho_in: torch.Tensor) -> torch.Tensor:
        if rho_in.shape[-2:] != (self.input_dim, self.input_dim):
            raise ValueError(
                f"rho_in must end in ({self.input_dim}, {self.input_dim}), got {tuple(rho_in.shape)}"
            )
        return F.pad(rho_in.to(device=self.H_raw.device, dtype=self.complex_dtype), (0, 2, 0, 2))

    def _coherent_stage(self, rho: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        if self.coherent_time == 0 or self.ablation == "l_only":
            return rho
        if self.backend == "cheby_suzuki":
            return qsw.evolve_density_chebyshev(
                rho,
                H,
                self.coherent_time,
                max_order=self.chebyshev_order,
                tol=self.chebyshev_tol,
            )
        if self.backend == "suzuki_global":
            return qsw.evolve_density_suzuki(
                rho,
                H,
                self.coherent_time,
                steps=self.suzuki_steps,
                order=self.suzuki_order,
            )
        return qsw.evolve_unitary(rho, H, self.coherent_time)

    def _open_stage(
        self,
        rho: torch.Tensor,
        H: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        if self.dissipative_time == 0:
            return rho
        if self.backend == "cheby_suzuki":
            return qsw.evolve_qsnn2d_cheby_suzuki(
                rho,
                H,
                gamma,
                self.dissipative_time,
                self.input_dim,
                steps=self.stage2_steps,
                chebyshev_order=self.chebyshev_order,
                chebyshev_tol=self.chebyshev_tol,
            )
        if self.backend == "suzuki_global":
            return qsw.evolve_qsnn2d_suzuki_global(
                rho,
                H,
                gamma,
                self.dissipative_time,
                self.input_dim,
                steps=self.stage2_steps,
                coherent_steps=max(1, self.suzuki_steps // self.stage2_steps),
                coherent_order=self.suzuki_order,
            )
        if self.backend == "exact_split":
            return qsw.evolve_qsnn2d_stage2_split(
                rho,
                H,
                gamma,
                self.dissipative_time,
                self.input_dim,
                steps=self.stage2_steps,
            )
        return qsw.evolve_qsnn2d_stage2_structured(
            rho,
            H,
            gamma,
            self.dissipative_time,
            self.input_dim,
            steps=self.stage2_steps,
        )

    def forward(self, rho_in: torch.Tensor):
        squeeze_back = rho_in.dim() == 2
        rho0 = self.embed_input(rho_in)
        H = self.hamiltonian()
        gamma = self.jump_amplitudes()
        rho_coherent = self._coherent_stage(rho0, H)
        rho_out = self._open_stage(rho_coherent, H, gamma)
        stats = output_statistics(rho_out, self.real_index, self.fake_index)
        result = {
            "rho_in": rho0,
            "rho_coherent": rho_coherent,
            "rho_out": rho_out,
            "jump_amplitudes": gamma,
            "effective_rates": self.effective_rates(),
            "total_rates": self.total_rates(),
            "branch_probabilities": self.branch_probabilities(),
            **stats,
        }
        if squeeze_back:
            result = {key: value[0] if value.dim() > 0 and value.shape[0] == 1 else value for key, value in result.items()}
        return result

    def _evolve_state(
        self,
        state: torch.Tensor,
        H: torch.Tensor,
        duration: float,
        spectral_bounds=None,
    ) -> torch.Tensor:
        if duration == 0 or self.ablation == "l_only":
            return state
        columns = state.unsqueeze(-1)
        if self.backend == "cheby_suzuki":
            evolved = qsw.evolve_state_chebyshev(
                columns,
                H,
                duration,
                max_order=self.chebyshev_order,
                tol=self.chebyshev_tol,
                spectral_bounds=spectral_bounds,
            )
        elif self.backend == "suzuki_global":
            evolved = qsw.evolve_state_suzuki(
                columns,
                H,
                duration,
                steps=self.suzuki_steps,
                order=self.suzuki_order,
            )
        else:
            evolved = qsw.evolve_state_exact(columns, H, duration)
        return evolved.squeeze(-1)

    def forward_state(self, state_in: torch.Tensor):
        """Memory-efficient discriminator path for pure input states.

        The structured jumps never return population from the two output nodes
        to the input subspace. Consequently the input block remains a
        sub-normalized pure state, while two scalars exactly track accumulated
        real/fake population. This avoids the quadratic density matrix and is
        the intended path for 128D, 256D, and padded full-resolution runs.
        """
        if self.backend == "exact_rk4":
            raise ValueError("state-vector training does not support backend=exact_rk4")
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
        H = self.input_hamiltonian()
        spectral_bounds = None
        if self.backend == "cheby_suzuki" and self.ablation != "l_only":
            spectral_bounds = qsw.chebyshev_spectral_bounds(H)

        state = self._evolve_state(
            state,
            H,
            self.coherent_time,
            spectral_bounds=spectral_bounds,
        )
        coherent_state = state
        p_real = torch.zeros(state.shape[0], device=state.device, dtype=state.real.dtype)
        p_fake = torch.zeros_like(p_real)

        if self.dissipative_time != 0:
            dt = self.dissipative_time / self.stage2_steps
            rates_by_output = self.effective_rates()
            total_rates = rates_by_output.sum(dim=0)
            transferred_fraction = 1.0 - torch.exp(-total_rates * dt)
            branch_fractions = torch.where(
                total_rates.unsqueeze(0) > 1e-12,
                rates_by_output / total_rates.clamp_min(1e-12).unsqueeze(0),
                torch.zeros_like(rates_by_output),
            )
            survival_amplitude = torch.exp(-0.5 * total_rates * dt).to(state.dtype)
            for _ in range(self.stage2_steps):
                state = self._evolve_state(
                    state,
                    H,
                    0.5 * dt,
                    spectral_bounds=spectral_bounds,
                )
                population = state.abs().square()
                transferred = population * transferred_fraction.unsqueeze(0)
                p_real = p_real + (
                    transferred * branch_fractions[0].unsqueeze(0)
                ).sum(dim=-1)
                p_fake = p_fake + (
                    transferred * branch_fractions[1].unsqueeze(0)
                ).sum(dim=-1)
                state = state * survival_amplitude.unsqueeze(0)
                state = self._evolve_state(
                    state,
                    H,
                    0.5 * dt,
                    spectral_bounds=spectral_bounds,
                )

        output_mass = p_real + p_fake
        leakage = 1.0 - output_mass
        normalized = torch.stack([p_real, p_fake], dim=-1)
        normalized = normalized / output_mass.unsqueeze(-1).clamp_min(1e-12)
        result = {
            "state_in": original_state,
            "state_coherent": coherent_state,
            "state_out": state,
            "p_real": p_real,
            "p_fake": p_fake,
            "output_mass": output_mass,
            "leakage": leakage,
            "z_expectation": p_real - p_fake,
            "normalized_probs": normalized,
            "state_trace": state.abs().square().sum(dim=-1) + output_mass,
            "jump_amplitudes": self.jump_amplitudes(),
            "effective_rates": self.effective_rates(),
            "total_rates": self.total_rates(),
            "branch_probabilities": self.branch_probabilities(),
        }
        if squeeze_back:
            for key in (
                "state_in",
                "state_coherent",
                "state_out",
                "p_real",
                "p_fake",
                "output_mass",
                "leakage",
                "z_expectation",
                "normalized_probs",
                "state_trace",
            ):
                result[key] = result[key][0]
        return result
