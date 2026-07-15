"""MNIST loading and deterministic subset preparation for QSNN-QGAN."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST

from .encoding import area_downsample


@dataclass(frozen=True)
class MNISTSubsetMetadata:
    train: bool
    digits: tuple[int, ...]
    image_size: int
    seed: int
    samples_per_class: int | None
    source_shape: tuple[int, int] = (28, 28)


def _select_digit_indices(
    targets: torch.Tensor,
    digits: tuple[int, ...],
    samples_per_class: int | None,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for digit in digits:
        indices = torch.nonzero(targets == digit, as_tuple=False).reshape(-1)
        permutation = torch.randperm(indices.numel(), generator=generator)
        indices = indices[permutation]
        if samples_per_class is not None:
            indices = indices[:samples_per_class]
        selected.append(indices)
    if not selected:
        raise ValueError("digits must contain at least one class")
    combined = torch.cat(selected)
    return combined[torch.randperm(combined.numel(), generator=generator)]


def load_mnist_tensor_dataset(
    root: str | Path,
    train: bool,
    digits=(0,),
    image_size: int = 8,
    samples_per_class: int | None = None,
    seed: int = 0,
    download: bool = False,
):
    """Load a deterministic MNIST subset as flattened non-negative tensors."""
    digits = tuple(int(digit) for digit in digits)
    if any(digit < 0 or digit > 9 for digit in digits):
        raise ValueError("MNIST digits must be between 0 and 9")
    dataset = MNIST(root=str(root), train=train, download=download)
    indices = _select_digit_indices(
        dataset.targets,
        digits,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    images = dataset.data[indices].to(torch.float32) / 255.0
    labels = dataset.targets[indices].to(torch.long)
    if image_size != 28:
        images = area_downsample(images, image_size)
    flattened = images.reshape(images.shape[0], -1).contiguous()
    metadata = MNISTSubsetMetadata(
        train=train,
        digits=digits,
        image_size=image_size,
        seed=seed,
        samples_per_class=samples_per_class,
    )
    return TensorDataset(flattened, labels), metadata
