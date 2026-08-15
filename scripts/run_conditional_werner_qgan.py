"""Conditional Werner-family QGAN comparison with held-out interpolation tests."""

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

from qgan.generators import ConditionalPurifiedPQCGenerator
from qgan.metrics import density_fidelity, physicality_diagnostics, purity, trace_distance, trainable_parameter_count
from qgan.mixed_state_discriminators import ConditionalAncillaVQCDiscriminator, ConditionalLayeredQSNNDiscriminator
from qgan.mixed_states import bell_population, negativity, pauli_correlations, werner_state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "conditional_werner_family.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--capacity-steps", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar(value):
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def mean_std(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(tensor.mean()), "std": float(tensor.std(unbiased=True)) if len(values) > 1 else 0.0}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gradient_norm(parameters):
    values = [parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None]
    return scalar(torch.linalg.vector_norm(torch.stack(values))) if values else 0.0


def build_generator(config, dtype, device):
    return ConditionalPurifiedPQCGenerator(
        system_qubits=int(config["system_qubits"]),
        environment_qubits=int(config["environment_qubits"]),
        n_layers=int(config["layers"]),
        alternating_entanglement=bool(config.get("alternating_entanglement", True)),
        real_dtype=dtype,
    ).to(device)


def build_discriminator(name, config, dtype, device):
    if name == "qsnn":
        return ConditionalLayeredQSNNDiscriminator(
            input_dim=4,
            hidden_dim=int(config["hidden_dim"]),
            coherent_time=float(config["coherent_time"]),
            input_hidden_time=float(config["input_hidden_time"]),
            hidden_output_time=float(config["hidden_output_time"]),
            chebyshev_order=int(config["chebyshev_order"]),
            chebyshev_tol=float(config["chebyshev_tol"]),
            init_h=float(config["init_h"]),
            target_layer_mass=float(config["target_layer_mass"]),
            coherent_backend=config.get("coherent_backend", "exact"),
            real_dtype=dtype,
        ).to(device)
    return ConditionalAncillaVQCDiscriminator(
        system_qubits=2,
        n_layers=int(config["layers"]),
        real_dtype=dtype,
    ).to(device)


def condition_tensor(values, dtype, device):
    return torch.tensor(values, dtype=dtype, device=device)


def target_states(conditions, complex_dtype):
    return werner_state(conditions, dtype=complex_dtype, device=conditions.device)


def discriminator_value(real_output, fake_output):
    return 0.5 + 0.25 * (
        real_output["z_expectation"].mean() - fake_output["z_expectation"].mean()
    )


@torch.no_grad()
def evaluate_grid(generator, conditions, complex_dtype, negativity_epsilon=1e-3):
    target = target_states(conditions, complex_dtype)
    generated = generator(conditions)
    fidelities = density_fidelity(target, generated)
    distances = trace_distance(target, generated)
    generated_purity = purity(generated)
    target_purity = purity(target)
    generated_bell = bell_population(generated)
    target_bell = bell_population(target)
    generated_negativity = negativity(generated)
    target_negativity = negativity(target)
    generated_correlations = pauli_correlations(generated)
    target_correlations = pauli_correlations(target)
    pauli_mae_per_state = sum(
        (generated_correlations[key] - target_correlations[key]).abs() for key in ("xx", "yy", "zz")
    ) / 3.0
    adjacent = trace_distance(generated[1:], generated[:-1])
    deltas = conditions[1:] - conditions[:-1]
    endpoint_distance = trace_distance(generated[0], generated[-1])
    positive = torch.nonzero(generated_negativity > negativity_epsilon).flatten()
    # A model that never produces entanglement is assigned a conservative
    # boundary just beyond the sampled interval instead of a NaN, keeping
    # reports serializable while visibly penalizing conditional collapse.
    estimated_boundary = scalar(conditions[positive[0]]) if positive.numel() else 1.01
    diagnostics = physicality_diagnostics(generated, include_min_eigenvalue=True)
    summary = {
        "mean_fidelity": scalar(fidelities.mean()),
        "min_fidelity": scalar(fidelities.min()),
        "mean_trace_distance": scalar(distances.mean()),
        "max_trace_distance": scalar(distances.max()),
        "purity_mae": scalar((generated_purity - target_purity).abs().mean()),
        "bell_population_mae": scalar((generated_bell - target_bell).abs().mean()),
        "negativity_mae": scalar((generated_negativity - target_negativity).abs().mean()),
        "pauli_correlation_mae": scalar(pauli_mae_per_state.mean()),
        "condition_sensitivity": scalar((adjacent / deltas).mean()),
        "endpoint_trace_distance": scalar(endpoint_distance),
        "estimated_entanglement_boundary": estimated_boundary,
        "entanglement_boundary_error": abs(estimated_boundary - 1.0 / 3.0),
        "trace_drift": scalar(diagnostics["trace_drift_max"]),
        "hermiticity_drift": scalar(diagnostics["hermiticity_drift_max"]),
        "min_eigenvalue": scalar(diagnostics["min_eigenvalue"]),
    }
    rows = []
    for index, p in enumerate(conditions):
        rows.append(
            {
                "p": scalar(p),
                "fidelity": scalar(fidelities[index]),
                "trace_distance": scalar(distances[index]),
                "target_purity": scalar(target_purity[index]),
                "generated_purity": scalar(generated_purity[index]),
                "target_bell_population": scalar(target_bell[index]),
                "generated_bell_population": scalar(generated_bell[index]),
                "target_negativity": scalar(target_negativity[index]),
                "generated_negativity": scalar(generated_negativity[index]),
                "pauli_correlation_mae": scalar(pauli_mae_per_state[index]),
            }
        )
    return summary, rows


def update_ema(ema_state, generator, decay):
    with torch.no_grad():
        for name, value in generator.state_dict().items():
            ema_state[name].mul_(decay).add_(value, alpha=1.0 - decay)


def capacity_check(initial_state, config, train_conditions, interp_conditions, dense_conditions, complex_dtype, dtype, device):
    generator = build_generator(config["generator"], dtype, device)
    generator.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.Adam(generator.parameters(), lr=float(config["training"]["capacity_lr"]))
    target = target_states(train_conditions, complex_dtype)
    for _ in range(int(config["training"]["capacity_steps"])):
        optimizer.zero_grad(set_to_none=True)
        generated = generator(train_conditions)
        loss = (generated - target).abs().square().mean()
        loss.backward()
        optimizer.step()
    train_summary, _ = evaluate_grid(generator, train_conditions, complex_dtype)
    interp_summary, _ = evaluate_grid(generator, interp_conditions, complex_dtype)
    dense_summary, _ = evaluate_grid(generator, dense_conditions, complex_dtype)
    return {
        "train_mean_fidelity": train_summary["mean_fidelity"],
        "interp_mean_fidelity": interp_summary["mean_fidelity"],
        "dense_min_fidelity": dense_summary["min_fidelity"],
        "dense_max_trace_distance": dense_summary["max_trace_distance"],
    }


def train_one(name, seed, initial_state, config, grids, complex_dtype, dtype, device):
    generator = build_generator(config["generator"], dtype, device)
    generator.load_state_dict(copy.deepcopy(initial_state))
    discriminator = build_discriminator(name, config["discriminators"][name], dtype, device)
    training = config["training"]
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(training["lr_g"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=float(training["lr_d"]))
    decay_start = int(training["lr_decay_start"])
    decay_factor = float(training["lr_decay_factor"])
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optimizer_g,
        lambda epoch: 1.0 if epoch < decay_start else max(decay_factor, 1.0 - (1.0 - decay_factor) * (epoch - decay_start) / max(1, int(training["epochs"]) - decay_start)),
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, scheduler_g.lr_lambdas[0])
    train_conditions = grids["train"]
    train_target = target_states(train_conditions, complex_dtype)
    ema_state = copy.deepcopy(generator.state_dict())
    best_state = copy.deepcopy(generator.state_dict())
    best_validation = -float("inf")
    records = []
    started = time.perf_counter()

    for epoch in range(int(training["epochs"]) + 1):
        if epoch > 0:
            for _ in range(int(training["discriminator_steps"])):
                optimizer_d.zero_grad(set_to_none=True)
                fake = generator(train_conditions).detach()
                real_output = discriminator(train_target, train_conditions)
                fake_output = discriminator(fake, train_conditions)
                loss_d = -discriminator_value(real_output, fake_output)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), float(training["grad_clip"]))
                grad_d = gradient_norm(discriminator.parameters())
                optimizer_d.step()
            for _ in range(int(training["generator_steps"])):
                optimizer_g.zero_grad(set_to_none=True)
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                fake = generator(train_conditions)
                fake_output = discriminator(fake, train_conditions)
                loss_g = -fake_output["z_expectation"].mean()
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), float(training["grad_clip"]))
                grad_g = gradient_norm(generator.parameters())
                optimizer_g.step()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
                update_ema(ema_state, generator, float(training["ema_decay"]))
            scheduler_g.step()
            scheduler_d.step()
        else:
            loss_d = loss_g = torch.tensor(float("nan"), device=device)
            grad_d = grad_g = 0.0

        if epoch % int(training["record_every"]) == 0 or epoch == int(training["epochs"]):
            train_summary, _ = evaluate_grid(generator, grids["train"], complex_dtype)
            validation_summary, _ = evaluate_grid(generator, grids["validation"], complex_dtype)
            interpolation_summary, _ = evaluate_grid(generator, grids["interpolation"], complex_dtype)
            if validation_summary["mean_fidelity"] > best_validation:
                best_validation = validation_summary["mean_fidelity"]
                best_state = copy.deepcopy(generator.state_dict())
            with torch.no_grad():
                real_output = discriminator(train_target, train_conditions)
                fake_output = discriminator(generator(train_conditions), train_conditions)
                output_mass = 0.5 * (real_output["output_mass"].mean() + fake_output["output_mass"].mean())
            records.append(
                {
                    "seed": seed,
                    "model": name,
                    "epoch": epoch,
                    "loss_d": scalar(loss_d),
                    "loss_g": scalar(loss_g),
                    "grad_norm_d": grad_d,
                    "grad_norm_g": grad_g,
                    "lr_g": optimizer_g.param_groups[0]["lr"],
                    "output_mass": scalar(output_mass),
                    "train_mean_fidelity": train_summary["mean_fidelity"],
                    "train_min_fidelity": train_summary["min_fidelity"],
                    "interp_mean_fidelity": interpolation_summary["mean_fidelity"],
                    "interp_min_fidelity": interpolation_summary["min_fidelity"],
                    "validation_mean_fidelity": validation_summary["mean_fidelity"],
                    "train_mean_trace_distance": train_summary["mean_trace_distance"],
                    "interp_mean_trace_distance": interpolation_summary["mean_trace_distance"],
                }
            )

    elapsed = time.perf_counter() - started
    checkpoints = {"final": copy.deepcopy(generator.state_dict()), "best": best_state, "ema": ema_state}
    variant_summaries, dense_rows = {}, []
    for variant, state in checkpoints.items():
        generator.load_state_dict(state)
        variant_summaries[variant] = {}
        for grid_name in ("train", "interpolation", "boundary", "dense"):
            summary, rows = evaluate_grid(generator, grids[grid_name], complex_dtype)
            variant_summaries[variant][grid_name] = summary
            if grid_name == "dense":
                dense_rows.extend({"seed": seed, "model": name, "variant": variant, **row} for row in rows)

    final = {
        "seed": seed,
        "model": name,
        "seconds": elapsed,
        "generator_parameters": trainable_parameter_count(generator),
        "discriminator_parameters": trainable_parameter_count(discriminator),
        "best_validation_epoch": max(records, key=lambda row: row["validation_mean_fidelity"])["epoch"],
    }
    for variant in ("final", "best", "ema"):
        for grid_name in ("train", "interpolation", "boundary", "dense"):
            for key, value in variant_summaries[variant][grid_name].items():
                final[f"{variant}_{grid_name}_{key}"] = value
    return records, final, dense_rows, checkpoints, discriminator.state_dict()


def aggregate(final_rows):
    metrics = [
        "best_train_mean_fidelity",
        "best_interpolation_mean_fidelity",
        "best_dense_min_fidelity",
        "best_dense_mean_trace_distance",
        "best_dense_max_trace_distance",
        "best_dense_purity_mae",
        "best_dense_bell_population_mae",
        "best_dense_negativity_mae",
        "best_dense_pauli_correlation_mae",
        "best_dense_condition_sensitivity",
        "best_dense_endpoint_trace_distance",
        "best_dense_entanglement_boundary_error",
        "final_interpolation_mean_fidelity",
        "ema_interpolation_mean_fidelity",
        "seconds",
    ]
    return {
        model: {
            metric: mean_std([row[metric] for row in final_rows if row["model"] == model and math.isfinite(row[metric])])
            for metric in metrics
        }
        for model in ("qsnn", "vqc")
    }


def plot_training(path, records):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fields = [
        ("train_mean_fidelity", "Training mean fidelity", False),
        ("interp_mean_fidelity", "Held-out interpolation fidelity", False),
        ("interp_mean_trace_distance", "Interpolation trace distance", False),
        ("grad_norm_g", "Generator gradient norm", True),
    ]
    for axis, (field, title, log_scale) in zip(axes.flat, fields):
        for model, color in (("qsnn", "tab:blue"), ("vqc", "tab:orange")):
            selected = [row for row in records if row["model"] == model]
            epochs = sorted({row["epoch"] for row in selected})
            means, lower, upper = [], [], []
            for epoch in epochs:
                values = torch.tensor([row[field] for row in selected if row["epoch"] == epoch], dtype=torch.float64)
                mean = float(values.mean())
                std = float(values.std(unbiased=True)) if len(values) > 1 else 0.0
                means.append(mean)
                lower.append(mean - std)
                upper.append(mean + std)
            axis.plot(epochs, means, label=model.upper(), color=color)
            axis.fill_between(epochs, lower, upper, color=color, alpha=0.18)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        if log_scale:
            axis.set_yscale("log")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_dense(path, dense_rows):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fields = [
        ("fidelity", "Fidelity"),
        ("trace_distance", "Trace distance"),
        ("generated_purity", "Purity"),
        ("generated_negativity", "Negativity"),
    ]
    for axis, (field, title) in zip(axes.flat, fields):
        for model, color in (("qsnn", "tab:blue"), ("vqc", "tab:orange")):
            selected = [row for row in dense_rows if row["model"] == model and row["variant"] == "best"]
            ps = sorted({row["p"] for row in selected})
            means = [sum(row[field] for row in selected if row["p"] == p) / len([row for row in selected if row["p"] == p]) for p in ps]
            axis.plot(ps, means, color=color, label=model.upper())
        if field == "generated_purity":
            axis.plot(ps, [(1 + 3 * p * p) / 4 for p in ps], "k--", label="Target")
        if field == "generated_negativity":
            axis.plot(ps, [max(0, (3 * p - 1) / 4) for p in ps], "k--", label="Target")
            axis.axvline(1 / 3, color="gray", linestyle=":")
        axis.set_title(title)
        axis.set_xlabel("p")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    axes[1, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.capacity_steps is not None:
        config["training"]["capacity_steps"] = args.capacity_steps
    if args.seeds is not None:
        config["experiment"]["seeds"] = args.seeds
    if args.device is not None:
        config["runtime"]["device"] = args.device
    output_dir = args.output_dir or ROOT / config["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = config["runtime"].get("device", "cpu")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    dtype = torch.float64 if config["runtime"].get("dtype") == "float64" else torch.float32
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    conditions = config["conditions"]
    grids = {
        "train": condition_tensor(conditions["train"], dtype, device),
        "validation": condition_tensor(conditions["validation"], dtype, device),
        "interpolation": condition_tensor(conditions["interpolation"], dtype, device),
        "boundary": condition_tensor(conditions["boundary"], dtype, device),
        "dense": torch.linspace(0, 1, int(conditions["dense_points"]), dtype=dtype, device=device),
    }

    all_records, final_rows, dense_rows, capacity_rows = [], [], [], []
    for seed in config["experiment"]["seeds"]:
        seed_everything(int(seed))
        initial_generator = build_generator(config["generator"], dtype, device)
        initial_state = copy.deepcopy(initial_generator.state_dict())
        capacity = capacity_check(initial_state, config, grids["train"], grids["interpolation"], grids["dense"], complex_dtype, dtype, device)
        capacity_rows.append({"seed": seed, **capacity})
        for name in ("qsnn", "vqc"):
            seed_everything(int(seed) + (1000 if name == "qsnn" else 2000))
            records, final, condition_rows, checkpoints, discriminator_state = train_one(
                name, int(seed), initial_state, config, grids, complex_dtype, dtype, device
            )
            all_records.extend(records)
            final_rows.append(final)
            dense_rows.extend(condition_rows)
            torch.save(
                {"generator_variants": checkpoints, "discriminator": discriminator_state, "config": config},
                output_dir / f"{name}_seed{seed}.pt",
            )
            print(
                f"seed={seed} model={name} best_interp_F={final['best_interpolation_mean_fidelity']:.6f} "
                f"dense_min_F={final['best_dense_min_fidelity']:.6f}"
            )

    summary = {
        "capacity_check": {key: mean_std([row[key] for row in capacity_rows]) for key in capacity_rows[0] if key != "seed"},
        "models": aggregate(final_rows),
        "config": config,
        "runtime": {"device": str(device), "torch": torch.__version__, "cuda": torch.cuda.is_available()},
    }
    write_csv(output_dir / "training_metrics.csv", all_records)
    write_csv(output_dir / "final_metrics.csv", final_rows)
    write_csv(output_dir / "dense_metrics.csv", dense_rows)
    write_csv(output_dir / "capacity_check.csv", capacity_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    plot_training(output_dir / "training_curves.png", all_records)
    plot_dense(output_dir / "dense_condition_curves.png", dense_rows)
    print(json.dumps(summary["models"], indent=2))


if __name__ == "__main__":
    main()
