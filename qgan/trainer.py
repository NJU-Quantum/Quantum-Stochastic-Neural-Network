"""Alternating optimization utilities for QSNN-QGAN."""

from contextlib import contextmanager

import torch

from .objectives import (
    direct_success_value_from_outputs,
    trace_z_value_from_outputs,
)


@contextmanager
def _temporarily_frozen(module):
    previous = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), previous):
            parameter.requires_grad_(requires_grad)


def gradient_norm(parameters) -> torch.Tensor:
    norms = [parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms))


class QGANTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        objective_mode: str = "trace_z",
        grad_clip: float | None = None,
        leakage_penalty: float = 0.0,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.objective_mode = objective_mode
        self.grad_clip = grad_clip
        self.leakage_penalty = float(leakage_penalty)
        if self.leakage_penalty < 0:
            raise ValueError("leakage_penalty must be non-negative")

    def _discriminator_loss(self, real_output, fake_output):
        if self.objective_mode == "trace_z":
            value = trace_z_value_from_outputs(real_output, fake_output)
        elif self.objective_mode == "direct_success":
            value = direct_success_value_from_outputs(real_output, fake_output)
        else:
            raise ValueError(f"Unsupported discriminator objective: {self.objective_mode}")
        leakage = 0.5 * (
            real_output["leakage"].mean() + fake_output["leakage"].mean()
        )
        return -value + self.leakage_penalty * leakage, value, leakage

    def discriminator_step(self, real_rho, noise, labels=None):
        self.optimizer_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_rho = self.generator(noise, labels=labels)
        real_out = self.discriminator(real_rho)
        fake_out = self.discriminator(fake_rho)
        loss_d, value, leakage = self._discriminator_loss(real_out, fake_out)
        loss_d.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
        grad_d = gradient_norm(self.discriminator.parameters())
        self.optimizer_d.step()
        return {
            "loss_d": loss_d.detach(),
            "value_d": value.detach(),
            "leakage_d": leakage.detach(),
            "grad_norm_d": grad_d.detach(),
            "real_output": real_out,
            "fake_output": fake_out,
        }

    def generator_step(self, noise, labels=None):
        self.optimizer_g.zero_grad(set_to_none=True)
        with _temporarily_frozen(self.discriminator):
            fake_rho = self.generator(noise, labels=labels)
            fake_out = self.discriminator(fake_rho)
            loss_g = -fake_out["z_expectation"].mean()
            loss_g.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip)
        grad_g = gradient_norm(self.generator.parameters())
        self.optimizer_g.step()
        return {
            "loss_g": loss_g.detach(),
            "grad_norm_g": grad_g.detach(),
            "fake_output": fake_out,
        }

    @torch.no_grad()
    def evaluate(self, real_rho, noise, labels=None):
        fake_rho = self.generator(noise, labels=labels)
        real_out = self.discriminator(real_rho)
        fake_out = self.discriminator(fake_rho)
        trace_value = trace_z_value_from_outputs(real_out, fake_out)
        direct_value = direct_success_value_from_outputs(real_out, fake_out)
        return {
            "V_trace": trace_value,
            "V_direct_success": direct_value,
            "real_output": real_out,
            "fake_output": fake_out,
        }
