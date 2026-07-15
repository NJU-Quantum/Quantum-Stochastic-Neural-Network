"""Probability-bottleneck convolutional autoencoder for MNIST QGAN runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class ProbabilityAutoencoder(nn.Module):
    """Encode 28x28 images as measurable probability vectors.

    The non-negative, L1-normalized bottleneck deliberately matches the
    diagonal probabilities produced by the quantum generator.  The decoder
    therefore consumes only measurable values rather than state-vector
    amplitudes.
    """

    image_size = 28

    def __init__(
        self,
        latent_dim: int = 64,
        base_channels: int = 16,
        latent_activation: str = "softplus_l1",
    ):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        self.latent_dim = int(latent_dim)
        self.base_channels = int(base_channels)
        if latent_activation not in {"softmax", "softplus_l1"}:
            raise ValueError("latent_activation must be softmax or softplus_l1")
        self.latent_activation = latent_activation

        self.encoder_features = nn.Sequential(
            nn.Conv2d(1, self.base_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                self.base_channels,
                2 * self.base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
        )
        hidden_dim = 2 * self.base_channels * 7 * 7
        self.to_logits = nn.Linear(hidden_dim, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                2 * self.base_channels,
                self.base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.ConvTranspose2d(
                self.base_channels,
                1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    @staticmethod
    def _as_images(images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 2 and images.shape[-1] == 28 * 28:
            return images.reshape(-1, 1, 28, 28)
        if images.dim() == 3 and images.shape[-2:] == (28, 28):
            return images.unsqueeze(1)
        if images.dim() == 4 and images.shape[1:] == (1, 28, 28):
            return images
        raise ValueError(
            "images must have shape (B,784), (B,28,28), or (B,1,28,28); "
            f"got {tuple(images.shape)}"
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        images = self._as_images(images)
        features = self.encoder_features(images).flatten(1)
        logits = self.to_logits(features)
        if self.latent_activation == "softmax":
            return torch.softmax(logits, dim=-1)
        positive = torch.nn.functional.softplus(logits) + 1e-8
        return positive / positive.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def decode(self, probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.dim() == 1:
            probabilities = probabilities.unsqueeze(0)
        if probabilities.dim() != 2 or probabilities.shape[-1] != self.latent_dim:
            raise ValueError(
                f"probabilities must have shape (B,{self.latent_dim}); "
                f"got {tuple(probabilities.shape)}"
            )
        # Scaling by latent_dim keeps a near-uniform probability vector at an
        # order-one magnitude while preserving its measurable information.
        hidden = torch.nn.functional.silu(self.from_latent(probabilities * self.latent_dim))
        hidden = hidden.reshape(-1, 2 * self.base_channels, 7, 7)
        return self.decoder(hidden)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = self.encode(images)
        return self.decode(probabilities), probabilities

    def freeze(self) -> "ProbabilityAutoencoder":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def model_config(self) -> dict[str, int]:
        return {
            "latent_dim": self.latent_dim,
            "base_channels": self.base_channels,
            "latent_activation": self.latent_activation,
        }


def save_autoencoder_artifact(
    path: str | Path,
    model: ProbabilityAutoencoder,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a self-describing frozen Autoencoder artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "probability_autoencoder",
            "model_config": model.model_config(),
            "model": model.state_dict(),
            "metadata": metadata or {},
        },
        destination,
    )
    return destination


def load_autoencoder_artifact(
    path: str | Path,
    *,
    map_location=None,
    freeze: bool = True,
) -> tuple[ProbabilityAutoencoder, dict[str, Any]]:
    """Load a :class:`ProbabilityAutoencoder` and its saved metadata."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported Autoencoder artifact: {payload.get('format_version')}")
    if payload.get("model_type") != "probability_autoencoder":
        raise ValueError(f"Unexpected model type: {payload.get('model_type')}")
    model = ProbabilityAutoencoder(**payload["model_config"])
    model.load_state_dict(payload["model"])
    if freeze:
        model.freeze()
    return model, payload
