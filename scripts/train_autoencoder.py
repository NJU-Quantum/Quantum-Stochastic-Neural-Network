"""Train the frozen probability-bottleneck Autoencoder used by QGAN runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

_MPL_CONFIG = Path(__file__).resolve().parents[1] / "tmp" / "matplotlib"
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qgan.autoencoder import ProbabilityAutoencoder, save_autoencoder_artifact
from qgan.checkpoint import runtime_metadata
from qgan.data import load_mnist_tensor_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda, but CUDA is unavailable")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    return config


def apply_overrides(config: dict, args) -> dict:
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.samples_per_class is not None:
        config["data"]["samples_per_class"] = args.samples_per_class
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = str(args.output_dir)
    if args.download:
        config["data"]["download"] = True
    return config


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    bce_total = mse_total = 0.0
    count = 0
    latent_sum = torch.zeros(model.latent_dim, device=device)
    latent_square_sum = torch.zeros(model.latent_dim, device=device)
    for pixels, _labels in loader:
        pixels = pixels.to(device, non_blocking=True)
        images = pixels.reshape(-1, 1, 28, 28)
        reconstruction, latent = model(images)
        batch = images.shape[0]
        bce_total += float(F.binary_cross_entropy(reconstruction, images, reduction="sum"))
        mse_total += float(F.mse_loss(reconstruction, images, reduction="sum"))
        latent_sum += latent.sum(dim=0)
        latent_square_sum += latent.square().sum(dim=0)
        count += batch
    if count == 0:
        raise RuntimeError("validation loader produced no samples")
    mean_latent = latent_sum / count
    latent_variance = (latent_square_sum / count - mean_latent.square()).clamp_min(0)
    latent_std = latent_variance.sqrt()
    entropy = -(mean_latent * mean_latent.clamp_min(1e-12).log()).sum()
    pixels_per_image = 28 * 28
    return {
        "val_bce_per_pixel": bce_total / (count * pixels_per_image),
        "val_mse_per_pixel": mse_total / (count * pixels_per_image),
        "latent_effective_dimensions": float(torch.exp(entropy)),
        "latent_max_mean_probability": float(mean_latent.max()),
        "latent_mean_component_std": float(latent_std.mean()),
        "latent_active_variance_dimensions": int((latent_std > 1e-3).sum()),
    }


@torch.no_grad()
def save_reconstruction_grid(model, dataset, output_dir: Path, device, count: int = 8) -> None:
    model.eval()
    count = min(int(count), len(dataset))
    if count <= 0:
        raise ValueError("reconstruction grid requires at least one sample")
    pixels = dataset.tensors[0][:count].to(device)
    images = pixels.reshape(-1, 1, 28, 28)
    reconstruction, latent = model(images)
    originals = images[:, 0].cpu()
    reconstructions = reconstruction[:, 0].cpu()
    figure, axes = plt.subplots(2, count, figsize=(1.7 * count, 3.6), constrained_layout=True)
    for column in range(count):
        axes[0, column].imshow(originals[column], cmap="gray", vmin=0, vmax=1)
        axes[1, column].imshow(reconstructions[column], cmap="gray", vmin=0, vmax=1)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_title("Original", fontsize=9)
    axes[1, 0].set_title("AE reconstruction", fontsize=9)
    figure.savefig(output_dir / "reconstruction_grid.png", dpi=180)
    plt.close(figure)
    torch.save(
        {
            "originals": originals,
            "reconstructions": reconstructions,
            "latent_probabilities": latent.cpu(),
        },
        output_dir / "reconstruction_samples.pt",
    )


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    seed = int(config["experiment"].get("seed", 0))
    seed_everything(seed)
    device = resolve_device(config["runtime"].get("device", "auto"))
    configured_output = Path(config["experiment"].get("output_dir", "outputs/autoencoder"))
    output_dir = configured_output if configured_output.is_absolute() else ROOT / configured_output
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    data_config = config["data"]
    dataset, metadata = load_mnist_tensor_dataset(
        root=ROOT / data_config.get("root", "datasets"),
        train=True,
        digits=data_config.get("digits", [0]),
        image_size=28,
        samples_per_class=data_config.get("samples_per_class"),
        seed=seed,
        download=bool(data_config.get("download", False)),
    )
    validation_fraction = float(data_config.get("validation_fraction", 0.1))
    if not 0 < validation_fraction < 1:
        raise ValueError("data.validation_fraction must be between 0 and 1")
    count = len(dataset)
    validation_count = max(1, int(round(count * validation_fraction)))
    if validation_count >= count:
        raise ValueError("Autoencoder training requires at least one train and validation sample")
    permutation = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    validation_indices = permutation[:validation_count]
    train_indices = permutation[validation_count:]
    pixels, labels = dataset.tensors
    train_dataset = TensorDataset(pixels[train_indices], labels[train_indices])
    validation_dataset = TensorDataset(pixels[validation_indices], labels[validation_indices])
    batch_size = int(config["training"].get("batch_size", 64))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = ProbabilityAutoencoder(
        latent_dim=int(config["model"].get("latent_dim", 64)),
        base_channels=int(config["model"].get("base_channels", 16)),
        latent_activation=config["model"].get("latent_activation", "softplus_l1"),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"].get("lr", 1e-3)))
    epochs = int(config["training"].get("epochs", 50))
    history = []
    best_validation = math.inf

    split_metadata = {
        **vars(metadata),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "validation_fraction": validation_fraction,
        "split_seed": seed,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "runtime": runtime_metadata(),
                "device": str(device),
                "data": split_metadata,
                "model_parameters": sum(p.numel() for p in model.parameters()),
            },
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        train_bce = train_mse = 0.0
        train_count = 0
        for batch_pixels, _labels in train_loader:
            batch_pixels = batch_pixels.to(device, non_blocking=True)
            images = batch_pixels.reshape(-1, 1, 28, 28)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, latent = model(images)
            reconstruction_loss = F.binary_cross_entropy(reconstruction, images)
            mean_latent = latent.mean(dim=0)
            balance_loss = (
                mean_latent
                * (mean_latent.clamp_min(1e-12) * model.latent_dim).log()
            ).sum()
            target_std = float(config["training"].get("latent_target_std", 0.005))
            component_std = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-8)
            variance_loss = F.relu(target_std - component_std).mean()
            loss = (
                reconstruction_loss
                + float(config["training"].get("latent_balance_weight", 0.05))
                * balance_loss
                + float(config["training"].get("latent_variance_weight", 10.0))
                * variance_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 5.0)))
            optimizer.step()
            batch = images.shape[0]
            train_bce += float(F.binary_cross_entropy(reconstruction.detach(), images, reduction="sum"))
            train_mse += float(F.mse_loss(reconstruction.detach(), images, reduction="sum"))
            train_count += batch

        validation = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_bce_per_pixel": train_bce / (train_count * 28 * 28),
            "train_mse_per_pixel": train_mse / (train_count * 28 * 28),
            **validation,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        append_csv(output_dir / "metrics.csv", row)
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        print(json.dumps(row, ensure_ascii=False))

        artifact_metadata = {
            "epoch": epoch,
            "metrics": row,
            "config": config,
            "data": split_metadata,
            "runtime": runtime_metadata(),
        }
        save_autoencoder_artifact(
            output_dir / "checkpoint_latest.pt",
            model,
            metadata=artifact_metadata,
        )
        if validation["val_bce_per_pixel"] < best_validation:
            best_validation = validation["val_bce_per_pixel"]
            save_autoencoder_artifact(
                output_dir / "checkpoint_best.pt",
                model,
                metadata=artifact_metadata,
            )

    best_payload = torch.load(output_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"])
    save_reconstruction_grid(model, validation_dataset, output_dir, device)


if __name__ == "__main__":
    main()
