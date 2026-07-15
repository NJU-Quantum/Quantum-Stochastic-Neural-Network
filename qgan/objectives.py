"""Trace-Z and direct-success objectives for QSNN-QGAN."""

import torch


def output_statistics(rho_out: torch.Tensor, real_index: int, fake_index: int):
    """Read raw output populations, leakage, and the Trace-Z expectation."""
    diagonal = torch.diagonal(rho_out, dim1=-2, dim2=-1).real
    p_real = diagonal[..., real_index]
    p_fake = diagonal[..., fake_index]
    output_mass = p_real + p_fake
    leakage = 1.0 - output_mass
    z_expectation = p_real - p_fake
    normalized = torch.stack([p_real, p_fake], dim=-1)
    normalized = normalized / output_mass.unsqueeze(-1).clamp_min(1e-12)
    return {
        "p_real": p_real,
        "p_fake": p_fake,
        "output_mass": output_mass,
        "leakage": leakage,
        "z_expectation": z_expectation,
        "normalized_probs": normalized,
    }


def partition_output_statistics(rho_out: torch.Tensor, split_index: int):
    """Read a complete two-outcome measurement split at ``split_index``."""
    diagonal = torch.diagonal(rho_out, dim1=-2, dim2=-1).real
    if split_index <= 0 or split_index >= diagonal.shape[-1]:
        raise ValueError("split_index must divide the density-matrix basis into two non-empty parts")
    p_real = diagonal[..., :split_index].sum(dim=-1)
    p_fake = diagonal[..., split_index:].sum(dim=-1)
    output_mass = p_real + p_fake
    leakage = 1.0 - output_mass
    z_expectation = p_real - p_fake
    normalized = torch.stack([p_real, p_fake], dim=-1)
    normalized = normalized / output_mass.unsqueeze(-1).clamp_min(1e-12)
    return {
        "p_real": p_real,
        "p_fake": p_fake,
        "output_mass": output_mass,
        "leakage": leakage,
        "z_expectation": z_expectation,
        "normalized_probs": normalized,
    }


def trace_z_value_from_outputs(real_output: dict, fake_output: dict):
    """Trace-Z value from discriminator output dictionaries."""
    return 0.5 + 0.25 * (
        real_output["z_expectation"].mean() - fake_output["z_expectation"].mean()
    )


def direct_success_value_from_outputs(real_output: dict, fake_output: dict):
    """Direct discriminator success from output dictionaries."""
    return 0.5 * (real_output["p_real"].mean() + fake_output["p_fake"].mean())


def trace_z_value(real_rho_out: torch.Tensor, fake_rho_out: torch.Tensor, real_index: int, fake_index: int):
    """Dallaire-Demers/Killoran Trace-Z adversarial value."""
    real_stats = output_statistics(real_rho_out, real_index, fake_index)
    fake_stats = output_statistics(fake_rho_out, real_index, fake_index)
    return trace_z_value_from_outputs(real_stats, fake_stats)


def direct_success_value(
    real_rho_out: torch.Tensor,
    fake_rho_out: torch.Tensor,
    real_index: int,
    fake_index: int,
):
    """Direct discriminator success probability, valid even with output leakage."""
    real_stats = output_statistics(real_rho_out, real_index, fake_index)
    fake_stats = output_statistics(fake_rho_out, real_index, fake_index)
    return direct_success_value_from_outputs(real_stats, fake_stats)


def discriminator_loss(
    real_rho_out: torch.Tensor,
    fake_rho_out: torch.Tensor,
    real_index: int,
    fake_index: int,
    mode: str = "trace_z",
):
    """Loss minimized by the discriminator."""
    if mode == "trace_z":
        value = trace_z_value(real_rho_out, fake_rho_out, real_index, fake_index)
    elif mode == "direct_success":
        value = direct_success_value(real_rho_out, fake_rho_out, real_index, fake_index)
    else:
        raise ValueError(f"Unsupported discriminator objective: {mode}")
    return -value


def generator_loss(fake_rho_out: torch.Tensor, real_index: int, fake_index: int):
    """Non-saturating Trace-Z generator loss from the implementation spec."""
    stats = output_statistics(fake_rho_out, real_index, fake_index)
    return -stats["z_expectation"].mean()
