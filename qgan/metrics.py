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
