"""Train QSNN-QGAN or its unitary baseline from a reproducible YAML config."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from qgan.autoencoder import load_autoencoder_artifact
from qgan.checkpoint import load_checkpoint, runtime_metadata, save_checkpoint
from qgan.data import load_mnist_tensor_dataset
from qgan.encoding import probability_amplitude_encode
from qgan.generators import PQCGenerator
from qgan.metrics import (
    density_fidelity,
    hellinger_distance,
    physicality_diagnostics,
    purity,
    trace_distance,
    total_variation_distance,
    trainable_parameter_count,
)
from qgan.qsnn_discriminator import QSNNDiscriminator
from qgan.trainer import QGANTrainer
from qgan.vqc_discriminator import VQCDiscriminator


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--n-d-steps", type=int)
    parser.add_argument("--n-g-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--autoencoder-checkpoint", type=Path)
    parser.add_argument("--discriminator", choices=("qsnn", "vqc"))
    parser.add_argument("--backend")
    parser.add_argument("--ablation", choices=("full", "h_only", "l_only"))
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


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
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.samples_per_class is not None:
        config["data"]["samples_per_class"] = args.samples_per_class
    if args.max_steps_per_epoch is not None:
        config["training"]["max_steps_per_epoch"] = args.max_steps_per_epoch
    if args.n_d_steps is not None:
        config["training"]["n_d_steps"] = args.n_d_steps
    if args.n_g_steps is not None:
        config["training"]["n_g_steps"] = args.n_g_steps
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = str(args.output_dir)
    if args.autoencoder_checkpoint is not None:
        config["data"]["autoencoder_checkpoint"] = str(args.autoencoder_checkpoint)
    if args.discriminator is not None:
        config["model"]["discriminator"] = args.discriminator
    if args.backend is not None:
        config["model"]["backend"] = args.backend
    if args.ablation is not None:
        config["model"]["ablation"] = args.ablation
    if args.download:
        config["data"]["download"] = True
    return config


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


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_dataset(config: dict, seed: int):
    data_config = config["data"]
    if data_config["dataset"] == "synthetic":
        count = int(data_config.get("num_samples", 16))
        input_dim = int(config["model"]["input_dim"])
        generator = torch.Generator().manual_seed(seed)
        pixels = torch.rand(count, input_dim, generator=generator)
        labels = torch.zeros(count, dtype=torch.long)
        return TensorDataset(pixels, labels), {"dataset": "synthetic", "num_samples": count}
    if data_config["dataset"] == "mnist":
        dataset, metadata = load_mnist_tensor_dataset(
            root=ROOT / data_config.get("root", "datasets"),
            train=True,
            digits=data_config.get("digits", [0]),
            image_size=int(data_config["image_size"]),
            samples_per_class=data_config.get("samples_per_class"),
            seed=seed,
            download=bool(data_config.get("download", False)),
        )
        return dataset, vars(metadata)
    raise ValueError(f"Unsupported dataset: {data_config['dataset']}")


@torch.no_grad()
def prepare_autoencoder_representation(dataset, config: dict, device: torch.device):
    """Replace 784-pixel samples with frozen 64D probability latents."""
    data_config = config["data"]
    representation = data_config.get("representation", "direct_pixels")
    if representation == "direct_pixels":
        return dataset, None, None
    if representation != "autoencoder":
        raise ValueError(f"Unsupported data representation: {representation}")
    if len(dataset.tensors) != 2:
        raise ValueError("Autoencoder preprocessing expects a (pixels, labels) TensorDataset")
    checkpoint_value = data_config.get("autoencoder_checkpoint")
    if not checkpoint_value:
        raise ValueError("data.autoencoder_checkpoint is required for Autoencoder representation")
    checkpoint_path = resolve_project_path(checkpoint_value).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {checkpoint_path}")
    autoencoder, payload = load_autoencoder_artifact(checkpoint_path, map_location=device)
    autoencoder = autoencoder.to(device).freeze()
    expected_dim = int(config["model"]["input_dim"])
    if autoencoder.latent_dim != expected_dim:
        raise ValueError(
            f"Autoencoder latent_dim {autoencoder.latent_dim} != model.input_dim {expected_dim}"
        )
    pixels, labels = dataset.tensors
    if pixels.shape[-1] != 28 * 28:
        raise ValueError(
            "Autoencoder representation requires original 28x28 inputs; "
            f"got {pixels.shape[-1]} features"
        )
    latent_batches = []
    encoding_batch_size = int(data_config.get("autoencoder_batch_size", 256))
    for start in range(0, len(dataset), encoding_batch_size):
        batch = pixels[start : start + encoding_batch_size].to(device)
        latent_batches.append(autoencoder.encode(batch).cpu())
    latents = torch.cat(latent_batches, dim=0).contiguous()
    prepared = TensorDataset(latents, labels, pixels)
    reference = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "model_config": payload["model_config"],
        "metadata": payload.get("metadata", {}),
    }
    return prepared, autoencoder, reference


def build_models(config: dict, device: torch.device):
    model = config["model"]
    dtype_name = config["runtime"].get("complex_dtype", "complex64")
    if dtype_name not in {"complex64", "complex128"}:
        raise ValueError("runtime.complex_dtype must be complex64 or complex128")
    real_dtype = torch.float64 if dtype_name == "complex128" else torch.float32
    input_dim = int(model["input_dim"])
    n_qubits = int(round(math.log2(input_dim)))
    if 1 << n_qubits != input_dim:
        raise ValueError("PQC generator currently requires model.input_dim to be a power of two")
    generator = PQCGenerator(
        n_qubits=n_qubits,
        n_layers=int(model.get("generator_layers", 2)),
        noise_dim=int(model.get("noise_dim", n_qubits)),
        noise_reuploading=bool(model.get("noise_reuploading", False)),
        alternating_entanglement=bool(model.get("alternating_entanglement", False)),
        canonicalize_output=bool(model.get("canonicalize_output", False)),
        real_dtype=real_dtype,
    ).to(device)

    discriminator_name = model["discriminator"]
    if discriminator_name == "qsnn":
        discriminator = QSNNDiscriminator(
            input_dim=input_dim,
            coherent_time=float(model.get("coherent_time", model.get("evolution_time", 1.0))),
            dissipative_time=float(model.get("dissipative_time", model.get("evolution_time", 1.0))),
            backend=model.get("backend", "cheby_suzuki"),
            stage2_steps=int(model.get("stage2_steps", 12)),
            chebyshev_order=int(model.get("chebyshev_order", 128)),
            chebyshev_tol=float(model.get("chebyshev_tol", 1e-10)),
            suzuki_steps=int(model.get("suzuki_steps", 12)),
            suzuki_order=int(model.get("suzuki_order", 2)),
            init_h=float(model.get("init_h", 0.02)),
            init_gamma=float(model.get("init_gamma", 0.1)),
            target_output_mass=model.get("target_output_mass", 0.8),
            gamma_semantics=model.get("gamma_semantics", "amplitude"),
            ablation=model.get("ablation", "full"),
            real_dtype=real_dtype,
        ).to(device)
    elif discriminator_name == "vqc":
        discriminator = VQCDiscriminator(
            input_dim=input_dim,
            evolution_time=float(model.get("evolution_time", 1.0)),
            backend=model.get("backend", "chebyshev"),
            chebyshev_order=int(model.get("chebyshev_order", 128)),
            chebyshev_tol=float(model.get("chebyshev_tol", 1e-10)),
            suzuki_steps=int(model.get("suzuki_steps", 12)),
            suzuki_order=int(model.get("suzuki_order", 2)),
            init_h=float(model.get("init_h", 0.02)),
            real_dtype=real_dtype,
        ).to(device)
    else:
        raise ValueError(f"Unsupported discriminator: {discriminator_name}")
    return generator, discriminator


def scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


@torch.no_grad()
def evaluation_metrics(
    trainer,
    real_rho,
    noise,
    *,
    real_probabilities=None,
    original_pixels=None,
    autoencoder=None,
) -> dict[str, float]:
    evaluation = trainer.evaluate(real_rho, noise)
    real_output = evaluation["real_output"]
    fake_output = evaluation["fake_output"]
    fake_rho = fake_output["rho_in"][..., : trainer.discriminator.input_dim, : trainer.discriminator.input_dim]
    # Spectral metrics are diagnostics rather than training operations.  CPU
    # complex128 is substantially more robust for the repeated/near-zero
    # eigenvalues of low-rank batch-mean density matrices than CUDA complex64.
    mean_real = real_rho.mean(dim=0).to(device="cpu", dtype=torch.complex128)
    mean_fake = fake_rho.mean(dim=0).to(device="cpu", dtype=torch.complex128)
    diagnostic_rho = fake_output["rho_out"].to(device="cpu", dtype=torch.complex128)
    physicality = physicality_diagnostics(diagnostic_rho, include_min_eigenvalue=True)
    real_correct = (real_output["p_real"] >= real_output["p_fake"]).float().mean()
    fake_correct = (fake_output["p_fake"] > fake_output["p_real"]).float().mean()
    metrics = {
        "V_trace": scalar(evaluation["V_trace"]),
        "V_direct_success": scalar(evaluation["V_direct_success"]),
        "accuracy": scalar(0.5 * (real_correct + fake_correct)),
        "z_real": scalar(real_output["z_expectation"].mean()),
        "z_fake": scalar(fake_output["z_expectation"].mean()),
        "output_mass_real": scalar(real_output["output_mass"].mean()),
        "output_mass_fake": scalar(fake_output["output_mass"].mean()),
        "leakage_real": scalar(real_output["leakage"].mean()),
        "leakage_fake": scalar(fake_output["leakage"].mean()),
        "fidelity_mean_states": scalar(density_fidelity(mean_real, mean_fake)),
        "trace_distance_mean_states": scalar(trace_distance(mean_real, mean_fake)),
        "hellinger_mean_states": scalar(hellinger_distance(mean_real, mean_fake)),
        "total_variation_mean_states": scalar(total_variation_distance(mean_real, mean_fake)),
        "purity_real_mean_state": scalar(purity(mean_real)),
        "purity_fake_mean_state": scalar(purity(mean_fake)),
        **{name: scalar(value) for name, value in physicality.items()},
    }
    if autoencoder is not None:
        if real_probabilities is None or original_pixels is None:
            raise ValueError("Autoencoder image metrics require real probabilities and original pixels")
        real_probabilities = real_probabilities.to(device=fake_rho.device, dtype=torch.float32)
        fake_probabilities = torch.diagonal(fake_rho, dim1=-2, dim2=-1).real.to(torch.float32)
        fake_probabilities = fake_probabilities / fake_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        originals = original_pixels.to(device=fake_rho.device, dtype=torch.float32).reshape(
            -1, 1, 28, 28
        )
        reconstructed_real = autoencoder.decode(real_probabilities)
        generated_images = autoencoder.decode(fake_probabilities)
        mean_fake_latent = fake_probabilities.mean(dim=0)
        latent_entropy = -(
            mean_fake_latent * mean_fake_latent.clamp_min(1e-12).log()
        ).sum()
        metrics.update(
            {
                "autoencoder_reconstruction_mse": scalar(
                    F.mse_loss(reconstructed_real, originals)
                ),
                "generated_mean_image_mse": scalar(
                    F.mse_loss(generated_images.mean(dim=0), originals.mean(dim=0))
                ),
                "generated_image_pixel_variance": scalar(
                    generated_images.var(dim=0, unbiased=False).mean()
                ),
                "real_image_pixel_variance": scalar(
                    originals.var(dim=0, unbiased=False).mean()
                ),
                "generated_latent_effective_dimensions": scalar(torch.exp(latent_entropy)),
            }
        )
    return metrics


@torch.no_grad()
def save_decoded_samples(
    output_dir: Path,
    generator,
    autoencoder,
    device: torch.device,
    *,
    seed: int,
    sample_count: int = 16,
    stem: str = "generated_samples_784",
) -> None:
    """Persist measurable latent probabilities and their 28x28 reconstructions."""
    generator.eval()
    autoencoder.eval()
    noise_generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(sample_count, generator.noise_dim, generator=noise_generator).to(device)
    generated_rho = generator(noise)
    probabilities = torch.diagonal(generated_rho, dim1=-2, dim2=-1).real.to(torch.float32)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    images = autoencoder.decode(probabilities).cpu()
    torch.save(
        {
            "latent_probabilities": probabilities.cpu(),
            "decoded_images": images,
            "seed": seed,
        },
        output_dir / f"{stem}.pt",
    )
    side = int(math.ceil(math.sqrt(sample_count)))
    figure, axes = plt.subplots(side, side, figsize=(1.8 * side, 1.8 * side), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < sample_count:
            axis.imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
    figure.suptitle("Generated 28x28 images decoded from measurable 64D probabilities")
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}.png", dpi=180)
    plt.close(figure)


@torch.no_grad()
def save_decoder_baselines(
    output_dir: Path,
    autoencoder,
    dataset,
    device: torch.device,
    *,
    seed: int,
    count: int = 4,
) -> None:
    """Show how much digit structure the frozen Decoder supplies by itself."""
    real_latents, _labels, original_pixels = dataset.tensors
    count = min(count, len(dataset))
    real_latents = real_latents[:count].to(device)
    originals = original_pixels[:count].reshape(-1, 1, 28, 28).to(device)
    reconstructions = autoencoder.decode(real_latents)
    random_generator = torch.Generator().manual_seed(seed)
    random_probabilities = torch.rand(
        count,
        autoencoder.latent_dim,
        generator=random_generator,
    ).to(device)
    random_probabilities = random_probabilities / random_probabilities.sum(
        dim=-1, keepdim=True
    )
    random_decoded = autoencoder.decode(random_probabilities)
    uniform_probabilities = torch.full_like(real_latents, 1.0 / autoencoder.latent_dim)
    uniform_decoded = autoencoder.decode(uniform_probabilities)
    rows = [originals, reconstructions, random_decoded, uniform_decoded]
    labels = ["Original", "AE reconstruction", "Random simplex", "Uniform latent"]
    figure, axes = plt.subplots(4, count, figsize=(1.8 * count, 7.2), squeeze=False)
    for row, (images, label) in enumerate(zip(rows, labels)):
        for column in range(count):
            axes[row, column].imshow(images[column, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[row, column].axis("off")
        axes[row, 0].set_title(label, fontsize=9)
    figure.tight_layout()
    figure.savefig(output_dir / "decoder_baselines.png", dpi=180)
    plt.close(figure)
    torch.save(
        {
            "originals": originals.cpu(),
            "real_latents": real_latents.cpu(),
            "reconstructions": reconstructions.cpu(),
            "random_probabilities": random_probabilities.cpu(),
            "random_decoded": random_decoded.cpu(),
            "uniform_probabilities": uniform_probabilities.cpu(),
            "uniform_decoded": uniform_decoded.cpu(),
        },
        output_dir / "decoder_baselines.pt",
    )


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    seed = int(config["experiment"].get("seed", 0))
    seed_everything(seed)
    device = resolve_device(config["runtime"].get("device", "auto"))
    output_dir = resolve_project_path(
        config["experiment"].get(
            "output_dir", f"outputs/qgan/{config['experiment']['name']}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    dataset, data_metadata = make_dataset(config, seed)
    dataset, autoencoder, autoencoder_reference = prepare_autoencoder_representation(
        dataset,
        config,
        device,
    )
    if autoencoder_reference is not None:
        data_metadata = {
            **data_metadata,
            "representation": "autoencoder",
            "original_dimension": 28 * 28,
            "latent_dimension": autoencoder.latent_dim,
            "autoencoder_sha256": autoencoder_reference["sha256"],
        }
    input_dim = int(config["model"]["input_dim"])
    if dataset.tensors[0].shape[-1] != input_dim:
        raise ValueError(
            f"encoded data dimension {dataset.tensors[0].shape[-1]} != model.input_dim {input_dim}"
        )
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=loader_generator,
        pin_memory=device.type == "cuda",
    )
    generator, discriminator = build_models(config, device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(config["training"]["lr_g"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=float(config["training"]["lr_d"]))
    trainer = QGANTrainer(
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        objective_mode=config["model"].get("loss_mode", "trace_z"),
        grad_clip=config["training"].get("grad_clip"),
        leakage_penalty=float(config["training"].get("leakage_penalty", 0.0)),
    )

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        restored = load_checkpoint(
            args.resume,
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            map_location=device,
        )
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["step"])
        loader_rng_state = restored.get("extra", {}).get("loader_rng_state")
        if loader_rng_state is not None:
            loader_generator.set_state(loader_rng_state.cpu())
        restored_autoencoder = restored.get("extra", {}).get("autoencoder_reference")
        if autoencoder_reference is not None and restored_autoencoder is not None:
            if restored_autoencoder.get("sha256") != autoencoder_reference["sha256"]:
                raise ValueError("Resume checkpoint was trained with a different Autoencoder artifact")

    if autoencoder is not None and args.resume is None:
        save_decoded_samples(
            output_dir,
            generator,
            autoencoder,
            device,
            seed=seed + 20260714,
            stem="generated_samples_initial_784",
        )
        save_decoder_baselines(
            output_dir,
            autoencoder,
            dataset,
            device,
            seed=seed + 20260715,
        )

    run_metadata = {
        "runtime": runtime_metadata(),
        "data": data_metadata,
        "device": str(device),
        "generator_parameters": trainable_parameter_count(generator),
        "discriminator_parameters": trainable_parameter_count(discriminator),
        "autoencoder": autoencoder_reference,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, ensure_ascii=False, default=str)

    metrics_json = output_dir / "metrics.json"
    if args.resume is not None and metrics_json.exists():
        with metrics_json.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
    else:
        history = []
    epochs = int(config["training"]["epochs"])
    max_steps = config["training"].get("max_steps_per_epoch")
    n_d_steps = int(config["training"].get("n_d_steps", 1))
    n_g_steps = int(config["training"].get("n_g_steps", 1))
    if n_d_steps <= 0 or n_g_steps <= 0:
        raise ValueError("training.n_d_steps and training.n_g_steps must be positive")
    noise_dim = generator.noise_dim
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        loss_d_total = loss_g_total = grad_d_total = grad_g_total = 0.0
        batches = 0
        last_real_rho = None
        last_real_probabilities = None
        last_original_pixels = None
        for batch_data in loader:
            if max_steps is not None and batches >= int(max_steps):
                break
            features = batch_data[0].to(device, non_blocking=True)
            original_pixels = (
                batch_data[2].to(device, non_blocking=True) if len(batch_data) >= 3 else None
            )
            real_probabilities, _, real_rho = probability_amplitude_encode(features)
            for _ in range(n_d_steps):
                d_result = trainer.discriminator_step(
                    real_rho,
                    torch.randn(real_rho.shape[0], noise_dim, device=device),
                )
                loss_d_total += scalar(d_result["loss_d"])
                grad_d_total += scalar(d_result["grad_norm_d"])
            for _ in range(n_g_steps):
                g_result = trainer.generator_step(
                    torch.randn(real_rho.shape[0], noise_dim, device=device)
                )
                loss_g_total += scalar(g_result["loss_g"])
                grad_g_total += scalar(g_result["grad_norm_g"])
            batches += 1
            global_step += 1
            last_real_rho = real_rho
            last_real_probabilities = real_probabilities
            last_original_pixels = original_pixels

        if batches == 0 or last_real_rho is None:
            raise RuntimeError("training produced no batches")
        evaluation = evaluation_metrics(
            trainer,
            last_real_rho,
            torch.randn(last_real_rho.shape[0], noise_dim, device=device),
            real_probabilities=last_real_probabilities,
            original_pixels=last_original_pixels,
            autoencoder=autoencoder,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "loss_D": loss_d_total / (batches * n_d_steps),
            "loss_G": loss_g_total / (batches * n_g_steps),
            "grad_norm_D": grad_d_total / (batches * n_d_steps),
            "grad_norm_G": grad_g_total / (batches * n_g_steps),
            "discriminator_updates": batches * n_d_steps,
            "generator_updates": batches * n_g_steps,
            **evaluation,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
            ),
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"NaN/Inf metric detected: {row}")
        history.append(row)
        append_csv(output_dir / "metrics.csv", row)
        with metrics_json.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        print(json.dumps(row, ensure_ascii=False))

        checkpoint_interval = int(config["training"].get("checkpoint_every", 1))
        if (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == epochs:
            save_checkpoint(
                output_dir / "checkpoint_latest.pt",
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                epoch=epoch,
                step=global_step,
                config=config,
                preprocessing_version=(
                    "autoencoder_probability_64_v1"
                    if autoencoder is not None
                    else "probability_amplitude_v1"
                ),
                extra={
                    "last_metrics": row,
                    "data_metadata": data_metadata,
                    "loader_rng_state": loader_generator.get_state(),
                    "autoencoder_reference": autoencoder_reference,
                },
            )
            if autoencoder is not None:
                save_decoded_samples(
                    output_dir,
                    generator,
                    autoencoder,
                    device,
                    seed=seed + 20260714,
                )


if __name__ == "__main__":
    main()
