"""Summarize a directory of QGAN runs into tables, curves, and sample grids."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

_MPL_CONFIG = Path(__file__).resolve().parents[1] / "tmp" / "matplotlib"
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qgan.autoencoder import load_autoencoder_artifact
from qgan.generators import PQCGenerator


DISPLAY_NAMES = {
    "qsnn_full": "QSNN full",
    "vqc": "VQC",
    "qsnn_l_only": "QSNN L-only",
    "qsnn_h_only": "QSNN H-only",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [{key: float(value) for key, value in row.items()} for row in rows]


def discover_runs(root: Path):
    runs = {}
    for name in DISPLAY_NAMES:
        metrics_path = root / name / "metrics.csv"
        if metrics_path.exists():
            rows = read_metrics(metrics_path)
            if rows:
                runs[name] = rows
    return runs


def save_final_summary(root: Path, runs) -> None:
    selected = [
        "epoch",
        "loss_D",
        "loss_G",
        "V_trace",
        "V_direct_success",
        "accuracy",
        "output_mass_real",
        "output_mass_fake",
        "leakage_real",
        "leakage_fake",
        "fidelity_mean_states",
        "trace_distance_mean_states",
        "total_variation_mean_states",
        "grad_norm_D",
        "grad_norm_G",
        "epoch_seconds",
        "peak_cuda_memory_mb",
    ]
    summary = []
    for name, rows in runs.items():
        final = rows[-1]
        summary.append({"model": name, **{key: final[key] for key in selected}})

    with (root / "final_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        if summary:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    with (root / "final_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        "# QGAN comparison",
        "",
        "| Model | Epoch | V trace | Direct success | Leakage real | Leakage fake | Fidelity | TV distance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {name} | {epoch:.0f} | {V_trace:.6f} | {V_direct_success:.6f} | "
            "{leakage_real:.6f} | {leakage_fake:.6f} | {fidelity_mean_states:.6f} | "
            "{total_variation_mean_states:.6f} |".format(
                name=DISPLAY_NAMES[row["model"]],
                **row,
            )
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(root: Path, runs) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    panels = [
        (axes[0, 0], (("V_trace", "V trace"), ("V_direct_success", "Direct success")), "Adversarial values"),
        (axes[0, 1], (("loss_D", "D loss"), ("loss_G", "G loss")), "Losses"),
        (axes[1, 0], (("leakage_real", "Real leakage"), ("leakage_fake", "Fake leakage")), "Output leakage"),
        (
            axes[1, 1],
            (("fidelity_mean_states", "Fidelity"), ("total_variation_mean_states", "TV distance")),
            "Generated-state quality",
        ),
    ]
    for axis, metrics, title in panels:
        for name, rows in runs.items():
            epochs = [row["epoch"] + 1 for row in rows]
            for key, metric_label in metrics:
                axis.plot(
                    epochs,
                    [row[key] for row in rows],
                    label=f"{DISPLAY_NAMES[name]} — {metric_label}",
                    linewidth=1.4,
                )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.savefig(root / "training_curves.png", dpi=180)
    plt.close(figure)


def load_generator(run_dir: Path):
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = config["model"]
    input_dim = int(model["input_dim"])
    n_qubits = int(round(math.log2(input_dim)))
    generator = PQCGenerator(
        n_qubits=n_qubits,
        n_layers=int(model.get("generator_layers", 2)),
        noise_dim=int(model.get("noise_dim", n_qubits)),
        noise_reuploading=bool(model.get("noise_reuploading", False)),
        alternating_entanglement=bool(model.get("alternating_entanglement", False)),
        canonicalize_output=bool(model.get("canonicalize_output", False)),
    )
    checkpoint = torch.load(
        run_dir / "checkpoint_latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    autoencoder = None
    if config.get("data", {}).get("representation") == "autoencoder":
        checkpoint_path = Path(config["data"]["autoencoder_checkpoint"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        autoencoder, _payload = load_autoencoder_artifact(checkpoint_path, map_location="cpu")
        autoencoder.eval()
    return generator, autoencoder


@torch.no_grad()
def plot_samples(root: Path, runs) -> None:
    available = [name for name in DISPLAY_NAMES if name in runs]
    if not available:
        return
    sample_count = 4
    figure, axes = plt.subplots(
        len(available),
        sample_count,
        figsize=(2.2 * sample_count, 2.2 * len(available)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, name in enumerate(available):
        generator, autoencoder = load_generator(root / name)
        noise_rng = torch.Generator().manual_seed(20260713)
        noise = torch.randn(sample_count, generator.noise_dim, generator=noise_rng)
        rho = generator(noise)
        probabilities = torch.diagonal(rho, dim1=-2, dim2=-1).real
        if autoencoder is not None:
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            images = autoencoder.decode(probabilities)[:, 0]
        else:
            side = int(round(math.sqrt(generator.state_dim)))
            images = probabilities.reshape(sample_count, side, side)
        for column in range(sample_count):
            image = images[column]
            display = image if autoencoder is not None else image / image.max().clamp_min(1e-12)
            axes[row, column].imshow(display, cmap="gray", vmin=0, vmax=1)
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].set_title(DISPLAY_NAMES[name], fontsize=9)
    figure.savefig(root / "generated_samples.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(root)
    if not runs:
        raise RuntimeError(f"No completed metrics found under {root}")
    save_final_summary(root, runs)
    plot_curves(root, runs)
    plot_samples(root, runs)
    print(f"Summarized {len(runs)} runs in {root}")


if __name__ == "__main__":
    main()
