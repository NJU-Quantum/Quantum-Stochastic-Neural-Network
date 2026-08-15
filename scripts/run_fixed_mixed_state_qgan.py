"""Compare QSNN and ancilla-VQC discriminators on a fixed Werner target."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qgan.generators import PurifiedPQCGenerator
from qgan.metrics import density_fidelity, physicality_diagnostics, purity, trace_distance, trainable_parameter_count
from qgan.mixed_state_discriminators import AncillaVQCDiscriminator, LayeredQSNNDiscriminator
from qgan.mixed_states import bell_population, negativity, pauli_correlations, werner_state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "fixed_werner_p06.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar(value):
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def gradient_norm(parameters):
    values = [p.grad.detach().norm() for p in parameters if p.grad is not None]
    return scalar(torch.linalg.vector_norm(torch.stack(values))) if values else 0.0


def state_metrics(rho, target):
    correlations = pauli_correlations(rho)
    target_correlations = pauli_correlations(target)
    distance = trace_distance(target, rho)
    diagnostics = physicality_diagnostics(rho, include_min_eigenvalue=True)
    return {
        "fidelity": scalar(density_fidelity(target, rho)),
        "trace_distance": scalar(distance),
        "purity": scalar(purity(rho)),
        "purity_error": abs(scalar(purity(rho) - purity(target))),
        "bell_population": scalar(bell_population(rho)),
        "bell_population_error": abs(scalar(bell_population(rho) - bell_population(target))),
        "negativity": scalar(negativity(rho)),
        "negativity_error": abs(scalar(negativity(rho) - negativity(target))),
        "pauli_correlation_mae": sum(
            abs(scalar(correlations[key] - target_correlations[key])) for key in ("xx", "yy", "zz")
        ) / 3.0,
        "helstrom_success": 0.5 + 0.5 * scalar(distance),
        "trace_drift": scalar(diagnostics["trace_drift_max"]),
        "hermiticity_drift": scalar(diagnostics["hermiticity_drift_max"]),
        "min_eigenvalue": scalar(diagnostics["min_eigenvalue"]),
    }


def build_generator(config, dtype, device):
    return PurifiedPQCGenerator(
        system_qubits=int(config["system_qubits"]),
        environment_qubits=int(config["environment_qubits"]),
        n_layers=int(config["layers"]),
        alternating_entanglement=bool(config.get("alternating_entanglement", True)),
        real_dtype=dtype,
    ).to(device)


def build_discriminator(name, config, dtype, device):
    if name == "qsnn":
        return LayeredQSNNDiscriminator(
            input_dim=4,
            hidden_dim=int(config["hidden_dim"]),
            coherent_time=float(config["coherent_time"]),
            input_hidden_time=float(config["input_hidden_time"]),
            hidden_output_time=float(config["hidden_output_time"]),
            chebyshev_order=int(config["chebyshev_order"]),
            chebyshev_tol=float(config["chebyshev_tol"]),
            init_h=float(config["init_h"]),
            target_layer_mass=float(config["target_layer_mass"]),
            real_dtype=dtype,
        ).to(device)
    return AncillaVQCDiscriminator(
        system_qubits=2,
        n_layers=int(config["layers"]),
        real_dtype=dtype,
    ).to(device)


def discriminator_value(real_output, fake_output):
    # Treat any QSNN population not yet at an output node as a random guess.
    return 0.5 + 0.25 * (real_output["z_expectation"] - fake_output["z_expectation"])


def capacity_check(initial_state, generator_config, target, training, dtype, device):
    generator = build_generator(generator_config, dtype, device)
    generator.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(generator.parameters(), lr=float(training["capacity_lr"]))
    for _ in range(int(training["capacity_steps"])):
        optimizer.zero_grad(set_to_none=True)
        rho = generator()
        loss = (rho - target).abs().square().sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return state_metrics(generator(), target)


def train_one(name, seed, initial_state, config, target, dtype, device):
    generator = build_generator(config["generator"], dtype, device)
    generator.load_state_dict(copy.deepcopy(initial_state))
    discriminator = build_discriminator(name, config["discriminators"][name], dtype, device)
    training = config["training"]
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(training["lr_g"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=float(training["lr_d"]))
    grad_clip = float(training["grad_clip"])
    records = []
    started = time.perf_counter()
    epochs = int(training["epochs"])
    record_every = int(training["record_every"])

    for epoch in range(epochs + 1):
        if epoch > 0:
            for _ in range(int(training["discriminator_steps"])):
                optimizer_d.zero_grad(set_to_none=True)
                fake = generator().detach()
                real_output = discriminator(target)
                fake_output = discriminator(fake)
                loss_d = -discriminator_value(real_output, fake_output)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                grad_d = gradient_norm(discriminator.parameters())
                optimizer_d.step()
            for _ in range(int(training["generator_steps"])):
                optimizer_g.zero_grad(set_to_none=True)
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                fake = generator()
                fake_output = discriminator(fake)
                loss_g = -fake_output["z_expectation"]
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
                grad_g = gradient_norm(generator.parameters())
                optimizer_g.step()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
        else:
            loss_d = torch.tensor(float("nan"), device=device)
            loss_g = torch.tensor(float("nan"), device=device)
            grad_d = grad_g = 0.0

        if epoch % record_every == 0 or epoch == epochs:
            with torch.no_grad():
                fake = generator()
                real_output = discriminator(target)
                fake_output = discriminator(fake)
                metrics = state_metrics(fake, target)
                value = scalar(discriminator_value(real_output, fake_output))
                records.append(
                    {
                        "seed": seed,
                        "model": name,
                        "epoch": epoch,
                        "loss_d": scalar(loss_d),
                        "loss_g": scalar(loss_g),
                        "grad_norm_d": grad_d,
                        "grad_norm_g": grad_g,
                        "discriminator_success": value,
                        "helstrom_gap": metrics["helstrom_success"] - value,
                        "output_mass_real": scalar(real_output["output_mass"]),
                        "output_mass_fake": scalar(fake_output["output_mass"]),
                        **metrics,
                    }
                )

    elapsed = time.perf_counter() - started
    final = dict(records[-1])
    final_window = records[-min(10, len(records)) :]
    fidelity_threshold = next((row["epoch"] for row in records if row["fidelity"] >= 0.99), None)
    distance_threshold = next((row["epoch"] for row in records if row["trace_distance"] <= 0.05), None)
    final.update(
        {
            "seconds": elapsed,
            "generator_parameters": trainable_parameter_count(generator),
            "discriminator_parameters": trainable_parameter_count(discriminator),
            "best_fidelity": max(row["fidelity"] for row in records),
            "min_trace_distance": min(row["trace_distance"] for row in records),
            "final_window_fidelity": sum(row["fidelity"] for row in final_window) / len(final_window),
            "final_window_trace_distance": sum(row["trace_distance"] for row in final_window) / len(final_window),
            "epoch_fidelity_099": fidelity_threshold if fidelity_threshold is not None else -1,
            "epoch_trace_distance_005": distance_threshold if distance_threshold is not None else -1,
        }
    )
    return records, final, generator.state_dict(), discriminator.state_dict()


def mean_std(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(tensor.mean()), "std": float(tensor.std(unbiased=True)) if len(values) > 1 else 0.0}


def summarize(final_rows):
    fields = [
        "fidelity",
        "trace_distance",
        "purity_error",
        "bell_population_error",
        "negativity_error",
        "pauli_correlation_mae",
        "best_fidelity",
        "min_trace_distance",
        "final_window_fidelity",
        "final_window_trace_distance",
        "seconds",
    ]
    result = {
        name: {field: mean_std([row[field] for row in final_rows if row["model"] == name]) for field in fields}
        for name in ("qsnn", "vqc")
    }
    for name in ("qsnn", "vqc"):
        selected = [row for row in final_rows if row["model"] == name]
        result[name]["success_rate_fidelity_099"] = sum(row["epoch_fidelity_099"] >= 0 for row in selected) / len(selected)
        result[name]["success_rate_trace_distance_005"] = sum(row["epoch_trace_distance_005"] >= 0 for row in selected) / len(selected)
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path, records):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plots = [
        ("fidelity", "Fidelity", True),
        ("trace_distance", "Trace distance", False),
        ("negativity_error", "Negativity error", False),
        ("grad_norm_g", "Generator gradient norm", False),
    ]
    for axis, (field, title, maximize) in zip(axes.flat, plots):
        for name, color in (("qsnn", "tab:blue"), ("vqc", "tab:orange")):
            model_rows = [row for row in records if row["model"] == name]
            epochs = sorted({row["epoch"] for row in model_rows})
            means, lowers, uppers = [], [], []
            for epoch in epochs:
                values = torch.tensor([row[field] for row in model_rows if row["epoch"] == epoch], dtype=torch.float64)
                means.append(float(values.mean()))
                std = float(values.std(unbiased=True)) if len(values) > 1 else 0.0
                lowers.append(means[-1] - std)
                uppers.append(means[-1] + std)
            axis.plot(epochs, means, label=name.upper(), color=color)
            axis.fill_between(epochs, lowers, uppers, alpha=0.18, color=color)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        if field == "grad_norm_g":
            axis.set_yscale("log")
        if maximize:
            axis.set_ylim(0, 1.01)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.seeds is not None:
        config["experiment"]["seeds"] = args.seeds
    if args.device is not None:
        config["runtime"]["device"] = args.device
    output_dir = args.output_dir or ROOT / config["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = config["runtime"].get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    dtype = torch.float64 if config["runtime"].get("dtype") == "float64" else torch.float32
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    target = werner_state(float(config["target"]["p"]), dtype=complex_dtype, device=device)

    all_records, final_rows, capacity_rows = [], [], []
    for seed in config["experiment"]["seeds"]:
        seed_everything(int(seed))
        initial_generator = build_generator(config["generator"], dtype, device)
        initial_state = copy.deepcopy(initial_generator.state_dict())
        capacity = capacity_check(initial_state, config["generator"], target, config["training"], dtype, device)
        capacity_rows.append({"seed": seed, **capacity})
        for name in ("qsnn", "vqc"):
            seed_everything(int(seed) + (1000 if name == "qsnn" else 2000))
            records, final, generator_state, discriminator_state = train_one(
                name, int(seed), initial_state, config, target, dtype, device
            )
            all_records.extend(records)
            final_rows.append(final)
            torch.save(
                {"generator": generator_state, "discriminator": discriminator_state, "config": config},
                output_dir / f"{name}_seed{seed}.pt",
            )
            print(f"seed={seed} model={name} F={final['fidelity']:.6f} Dtr={final['trace_distance']:.6f}")

    summary = {
        "target": {"p": float(config["target"]["p"]), **state_metrics(target, target)},
        "capacity_check": {field: mean_std([row[field] for row in capacity_rows]) for field in ("fidelity", "trace_distance")},
        "models": summarize(final_rows),
        "config": config,
        "runtime": {"device": str(device), "torch": torch.__version__, "cuda": torch.cuda.is_available()},
    }
    write_csv(output_dir / "training_metrics.csv", all_records)
    write_csv(output_dir / "final_metrics.csv", final_rows)
    write_csv(output_dir / "capacity_check.csv", capacity_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    plot_curves(output_dir / "training_curves.png", all_records)
    print(json.dumps(summary["models"], indent=2))


if __name__ == "__main__":
    main()
