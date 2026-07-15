"""Classical-image to quantum-state encoding utilities for QSNN-QGAN."""

import torch
import torch.nn.functional as F


def area_downsample(images: torch.Tensor, image_size: int) -> torch.Tensor:
    """Area-downsample ``(B,H,W)`` or ``(B,C,H,W)`` images."""
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if images.dim() == 3:
        resized = F.interpolate(
            images.unsqueeze(1),
            size=(image_size, image_size),
            mode="area",
        )
        return resized[:, 0]
    if images.dim() == 4:
        return F.interpolate(images, size=(image_size, image_size), mode="area")
    raise ValueError(f"images must have shape (B,H,W) or (B,C,H,W), got {tuple(images.shape)}")


def probability_amplitude_encode(pixels: torch.Tensor, eps: float = 1e-8):
    """
    Encode non-negative flattened pixels using probability amplitudes.

    The last dimension is treated as the pixel dimension. A one-dimensional
    input is handled as a single sample. Returns ``(probabilities, psi, rho)``.
    """
    if pixels.dim() == 0:
        raise ValueError("pixels must contain at least one feature dimension")
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    if torch.any(pixels < 0):
        raise ValueError("probability amplitude encoding requires non-negative pixels")

    values = pixels
    denominator = values.sum(dim=-1, keepdim=True) + eps * values.shape[-1]
    if eps == 0 and torch.any(denominator <= 0):
        raise ValueError("zero-sum pixels require eps > 0")

    probabilities = (values + eps) / denominator.clamp_min(torch.finfo(values.dtype).tiny)
    complex_dtype = torch.complex128 if values.dtype == torch.float64 else torch.complex64
    psi = torch.sqrt(probabilities).to(complex_dtype).unsqueeze(-1)
    rho = psi @ psi.mH
    return probabilities, psi, rho


def probabilities_from_density(
    rho: torch.Tensor,
    feature_dim: int | None = None,
    normalize: bool = False,
) -> torch.Tensor:
    """Read computational-basis probabilities from a density matrix."""
    probabilities = torch.diagonal(rho, dim1=-2, dim2=-1).real
    if feature_dim is not None:
        probabilities = probabilities[..., :feature_dim]
    if normalize:
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probabilities


def embed_binary_label_density(rho: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return ``|label><label| tensor rho`` for binary integer labels."""
    squeeze_back = rho.dim() == 2
    rho_b = rho.unsqueeze(0) if squeeze_back else rho
    labels_b = labels.reshape(-1).to(device=rho_b.device, dtype=torch.long)
    if labels_b.shape[0] != rho_b.shape[0]:
        raise ValueError("labels batch size must match rho batch size")
    if torch.any((labels_b < 0) | (labels_b > 1)):
        raise ValueError("binary labels must be 0 or 1")

    label_vectors = F.one_hot(labels_b, num_classes=2).to(rho_b.dtype)
    label_projectors = label_vectors.unsqueeze(-1) @ label_vectors.unsqueeze(-2)
    embedded = torch.einsum("bij,bkl->bikjl", label_projectors, rho_b)
    n = rho_b.shape[-1]
    embedded = embedded.reshape(rho_b.shape[0], 2 * n, 2 * n)
    return embedded[0] if squeeze_back else embedded


def pad_density_dimension(rho: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Embed a density matrix into the leading block of a larger Hilbert space."""
    current_dim = rho.shape[-1]
    if rho.shape[-2] != current_dim:
        raise ValueError("rho must be square")
    if target_dim < current_dim:
        raise ValueError(f"target_dim {target_dim} is smaller than current dimension {current_dim}")
    padding = target_dim - current_dim
    return F.pad(rho, (0, padding, 0, padding))


def padding_mass(rho: torch.Tensor, valid_dim: int) -> torch.Tensor:
    """Probability mass outside the leading ``valid_dim`` computational states."""
    probabilities = probabilities_from_density(rho)
    if valid_dim < 0 or valid_dim > probabilities.shape[-1]:
        raise ValueError("valid_dim must be within the density-matrix dimension")
    return probabilities[..., valid_dim:].sum(dim=-1)
