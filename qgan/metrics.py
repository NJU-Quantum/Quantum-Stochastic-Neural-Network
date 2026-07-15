"""Quantum-state, classification, and physicality metrics for QGAN runs."""

import torch


def _as_batch(rho: torch.Tensor) -> tuple[torch.Tensor, bool]:
    return (rho.unsqueeze(0), True) if rho.dim() == 2 else (rho, False)


def _restore_batch(value: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return value[0] if squeezed else value


def density_fidelity(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Squared Uhlmann fidelity for density matrices, with batch support."""
    rho_b, squeezed = _as_batch(rho)
    sigma_b, _ = _as_batch(sigma)
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (rho_b + rho_b.mH))
    root_eigenvalues = eigenvalues.clamp_min(0).sqrt().to(eigenvectors.dtype)
    square_root = eigenvectors @ torch.diag_embed(root_eigenvalues) @ eigenvectors.mH
    middle = square_root @ sigma_b @ square_root
    middle = 0.5 * (middle + middle.mH)
    fidelity = torch.linalg.eigvalsh(middle).clamp_min(0).sqrt().sum(dim=-1).square()
    return _restore_batch(fidelity.real.clamp(0, 1), squeezed)


def trace_distance(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Half the trace norm of ``rho - sigma``."""
    difference = rho - sigma
    return 0.5 * torch.linalg.svdvals(difference).sum(dim=-1).real


def diagonal_probabilities(rho: torch.Tensor) -> torch.Tensor:
    probabilities = torch.diagonal(rho, dim1=-2, dim2=-1).real.clamp_min(0)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def hellinger_distance(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Hellinger distance between computational-basis distributions."""
    p = diagonal_probabilities(rho)
    q = diagonal_probabilities(sigma)
    coefficient = torch.sqrt(p * q).sum(dim=-1).clamp(0, 1)
    return torch.sqrt((1.0 - coefficient).clamp_min(0))


def total_variation_distance(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    p = diagonal_probabilities(rho)
    q = diagonal_probabilities(sigma)
    return 0.5 * (p - q).abs().sum(dim=-1)


def purity(rho: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(rho @ rho, dim1=-2, dim2=-1).sum(dim=-1).real


def physicality_diagnostics(rho: torch.Tensor, include_min_eigenvalue: bool = False):
    trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)
    diagnostics = {
        "trace_drift_max": (trace.real - 1.0).abs().max(),
        "trace_imag_max": trace.imag.abs().max(),
        "hermiticity_drift_max": (rho - rho.mH).abs().max(),
    }
    if include_min_eigenvalue:
        hermitian = 0.5 * (rho + rho.mH)
        diagnostics["min_eigenvalue"] = torch.linalg.eigvalsh(hermitian).real.min()
    return diagnostics


def trainable_parameter_count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def empirical_pure_state_metrics(
    real_states: torch.Tensor,
    fake_states: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Metrics between empirical mixtures without forming ``N x N`` matrices.

    Each row is interpreted as a normalized pure state with uniform empirical
    weight. All expensive decompositions are bounded by the combined batch
    size, so this remains practical for the 1024-dimensional padded MNIST run.
    """
    if real_states.dim() != 2 or fake_states.dim() != 2:
        raise ValueError("real_states and fake_states must have shape (B,N)")
    if real_states.shape[-1] != fake_states.shape[-1]:
        raise ValueError("real and fake state dimensions must match")
    if real_states.shape[0] == 0 or fake_states.shape[0] == 0:
        raise ValueError("state batches must be non-empty")

    complex_dtype = (
        torch.complex128
        if real_states.dtype == torch.complex128 or fake_states.dtype == torch.complex128
        else torch.complex64
    )
    real = real_states.to(complex_dtype)
    fake = fake_states.to(complex_dtype)
    real_factor = real.T / real.shape[0] ** 0.5
    fake_factor = fake.T / fake.shape[0] ** 0.5

    cross_gram = real_factor.mH @ fake_factor
    fidelity = torch.linalg.svdvals(cross_gram).sum().square().real.clamp(0, 1)

    factors = torch.cat([real_factor, fake_factor], dim=-1)
    _q, r = torch.linalg.qr(factors, mode="reduced")
    signs = torch.cat(
        [
            torch.ones(real.shape[0], device=r.device, dtype=r.real.dtype),
            -torch.ones(fake.shape[0], device=r.device, dtype=r.real.dtype),
        ]
    ).to(r.dtype)
    small_difference = (r * signs.unsqueeze(0)) @ r.mH
    small_difference = 0.5 * (small_difference + small_difference.mH)
    trace_distance_value = 0.5 * torch.linalg.eigvalsh(small_difference).abs().sum().real

    real_probabilities = real.abs().square().mean(dim=0)
    fake_probabilities = fake.abs().square().mean(dim=0)
    coefficient = torch.sqrt(real_probabilities * fake_probabilities).sum().clamp(0, 1)
    hellinger = torch.sqrt((1.0 - coefficient).clamp_min(0))
    total_variation = 0.5 * (real_probabilities - fake_probabilities).abs().sum()
    real_gram = real_factor.mH @ real_factor
    fake_gram = fake_factor.mH @ fake_factor
    return {
        "fidelity_mean_states": fidelity,
        "trace_distance_mean_states": trace_distance_value,
        "hellinger_mean_states": hellinger,
        "total_variation_mean_states": total_variation,
        "purity_real_mean_state": real_gram.abs().square().sum().real,
        "purity_fake_mean_state": fake_gram.abs().square().sum().real,
    }
