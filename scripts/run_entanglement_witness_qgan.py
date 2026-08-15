"""Run the adversarial two-qubit entanglement-witness benchmark."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qgan.entanglement_witness import (  # noqa: E402
    SeparableMixtureGenerator,
    calibrated_witness,
    certified_separable_score_bound,
    effective_observable,
    observable_score,
    pauli_coefficients,
    werner_psi_plus_witness,
)
from qgan.metrics import density_fidelity, trace_distance, trainable_parameter_count  # noqa: E402
from qgan.mixed_state_discriminators import AncillaVQCDiscriminator, LayeredQSNNDiscriminator  # noqa: E402
from qgan.mixed_states import negativity, werner_state  # noqa: E402


def scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return float(value)


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def random_product_states(count: int, dtype: torch.dtype, device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    theta = torch.acos(1.0 - 2.0 * torch.rand(count, 2, generator=generator, dtype=dtype))
    phi = 2.0 * math.pi * torch.rand(count, 2, generator=generator, dtype=dtype)
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    first = torch.stack(
        [torch.cos(theta[:, 0] / 2), torch.exp(1j * phi[:, 0].to(complex_dtype)) * torch.sin(theta[:, 0] / 2)],
        dim=-1,
    ).to(complex_dtype)
    second = torch.stack(
        [torch.cos(theta[:, 1] / 2), torch.exp(1j * phi[:, 1].to(complex_dtype)) * torch.sin(theta[:, 1] / 2)],
        dim=-1,
    ).to(complex_dtype)
    states = torch.einsum("ki,kj->kij", first, second).reshape(count, 4).to(device)
    return states[:, :, None] @ states.conj()[:, None, :]


def build_generator(config, dtype, device):
    return SeparableMixtureGenerator(
        components=int(config["components"]), real_dtype=dtype
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
        system_qubits=2, n_layers=int(config["layers"]), real_dtype=dtype
    ).to(device)


def score(discriminator, rho):
    return discriminator(rho)["z_expectation"].real


def capacity_check(initial_state, target_p, config, complex_dtype, dtype, device):
    generator = build_generator(config["generator"], dtype, device)
    generator.load_state_dict(copy.deepcopy(initial_state))
    target = werner_state(target_p, dtype=complex_dtype, device=device)
    optimizer = torch.optim.Adam(
        generator.parameters(), lr=float(config["generator"]["capacity_lr"])
    )
    for _ in range(int(config["generator"]["capacity_steps"])):
        optimizer.zero_grad(set_to_none=True)
        generated = generator()
        loss = (generated - target).abs().square().mean()
        loss.backward()
        optimizer.step()
    generated = generator().detach()
    return {
        "target_p": float(target_p),
        "fidelity": scalar(density_fidelity(target, generated)),
        "trace_distance": scalar(trace_distance(target, generated)),
        "negativity": scalar(negativity(generated)),
    }


def record_state(discriminator, generator, target, validation_states):
    observable = effective_observable(discriminator)
    target_score = scalar(observable_score(observable, target))
    validation_scores = observable_score(observable, validation_states.to(observable.dtype))
    fake = generator().detach()
    fake_score = scalar(observable_score(observable, fake))
    target_output = discriminator(target)
    fake_output = discriminator(fake)
    return {
        "target_score": target_score,
        "fake_score": fake_score,
        "adversarial_gap": target_score - fake_score,
        "validation_sep_max": scalar(validation_scores.max()),
        "validation_gap": target_score - scalar(validation_scores.max()),
        "fake_fidelity": scalar(density_fidelity(target, fake)),
        "fake_trace_distance": scalar(trace_distance(target, fake)),
        "fake_negativity": scalar(negativity(fake)),
        "target_output_mass": scalar(target_output["output_mass"]),
        "fake_output_mass": scalar(fake_output["output_mass"]),
    }


def estimate_boundary(ps: np.ndarray, values: np.ndarray) -> float:
    for index in range(len(ps) - 1):
        left, right = values[index], values[index + 1]
        if left >= 0.0 and right < 0.0:
            weight = left / max(left - right, 1e-15)
            return float(ps[index] + weight * (ps[index + 1] - ps[index]))
    return float("nan")


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    labels = list(left)
    first = np.array([left[label] for label in labels], dtype=np.float64)
    second = np.array([right[label] for label in labels], dtype=np.float64)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denominator) if denominator > 0 else float("nan")


def evaluate_checkpoint(
    discriminator,
    generator,
    target,
    dense_ps,
    config,
    complex_dtype,
    variant,
    seed,
    model,
):
    observable = effective_observable(discriminator)
    certification = config["certification"]
    bound = certified_separable_score_bound(
        observable,
        tolerance=float(certification["tolerance"]),
        max_cells=int(certification["max_cells"]),
    )
    witness = calibrated_witness(observable, bound.upper)
    target_score = scalar(observable_score(observable, target))
    certified_margin = target_score - bound.upper
    witness_norm = scalar(torch.linalg.eigvalsh(witness).abs().max())
    generated = generator().detach()
    dense_states = werner_state(dense_ps, dtype=complex_dtype, device=dense_ps.device)
    dense_values = observable_score(witness, dense_states).detach().cpu().numpy()
    dense_p_values = dense_ps.detach().cpu().numpy()
    boundary = estimate_boundary(dense_p_values, dense_values)
    learned_coefficients = pauli_coefficients(witness)
    theory = werner_psi_plus_witness(dtype=complex_dtype, device=target.device)
    theory_coefficients = pauli_coefficients(theory)

    # Direct-score reconstruction error is a guard against accidentally using
    # a nonlinear discriminator readout in a witness experiment.
    probes = werner_state(
        torch.tensor([0.0, 0.2, 0.6, 1.0], dtype=dense_ps.dtype, device=dense_ps.device),
        dtype=complex_dtype,
        device=dense_ps.device,
    )
    direct = torch.stack([score(discriminator, probe) for probe in probes])
    reconstructed = observable_score(observable, probes)
    linearity_error = scalar((direct - reconstructed).abs().max())
    target_output = discriminator(target)
    fake_output = discriminator(generated)
    result = {
        "seed": seed,
        "model": model,
        "variant": variant,
        "target_score": target_score,
        "separable_lower": bound.lower,
        "separable_upper": bound.upper,
        "separable_bound_gap": bound.gap,
        "certification_converged": int(bound.converged),
        "certification_evaluations": bound.evaluations,
        "certified_margin": certified_margin,
        "certified_detected": int(
            certified_margin > float(certification["detection_tolerance"])
        ),
        "witness_expectation_target": -certified_margin,
        "normalized_margin": certified_margin / max(witness_norm, 1e-12),
        "estimated_boundary": boundary,
        "boundary_error": abs(boundary - 1.0 / 3.0) if math.isfinite(boundary) else float("nan"),
        "pauli_cosine": cosine_similarity(learned_coefficients, theory_coefficients),
        "fake_fidelity": scalar(density_fidelity(target, generated)),
        "fake_trace_distance": scalar(trace_distance(target, generated)),
        "fake_negativity": scalar(negativity(generated)),
        "target_output_mass": scalar(target_output["output_mass"]),
        "fake_output_mass": scalar(fake_output["output_mass"]),
        "linearity_error": linearity_error,
        "observable_norm": scalar(torch.linalg.eigvalsh(observable).abs().max()),
        "witness_norm": witness_norm,
    }
    curve_rows = [
        {
            "seed": seed,
            "model": model,
            "variant": variant,
            "p": float(p),
            "witness_expectation": float(value),
            "theory_expectation": float((1.0 - 3.0 * p) / 4.0),
            "normalized_witness_expectation": float(value / max(witness_norm, 1e-12)),
            "normalized_theory_expectation": float((1.0 - 3.0 * p) / 2.0),
        }
        for p, value in zip(dense_p_values, dense_values)
    ]
    coefficient_rows = [
        {
            "seed": seed,
            "model": model,
            "variant": variant,
            "pauli": label,
            "learned": value,
            "theory": theory_coefficients[label],
            "learned_normalized": value / max(witness_norm, 1e-12),
            "theory_normalized": theory_coefficients[label] / 0.5,
        }
        for label, value in learned_coefficients.items()
    ]
    return result, curve_rows, coefficient_rows, observable.detach().cpu(), witness.detach().cpu()


def train_one(name, seed, initial_state, config, dtype, device, validation_states):
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    generator = build_generator(config["generator"], dtype, device)
    generator.load_state_dict(copy.deepcopy(initial_state))
    torch.manual_seed(seed + 10_000)
    discriminator = build_discriminator(name, config["discriminators"][name], dtype, device)
    target = werner_state(float(config["target"]["werner_p"]), dtype=complex_dtype, device=device)
    training = config["training"]
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(training["lr_g"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=float(training["lr_d"]))
    decay_start = int(training["lr_decay_start"])
    epochs = int(training["epochs"])
    decay_factor = float(training["lr_decay_factor"])
    schedule = lambda epoch: 1.0 if epoch < decay_start else max(
        decay_factor,
        1.0 - (1.0 - decay_factor) * (epoch - decay_start) / max(1, epochs - decay_start),
    )
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, schedule)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, schedule)
    records = []
    best_validation = -math.inf
    best_generator = copy.deepcopy(generator.state_dict())
    best_discriminator = copy.deepcopy(discriminator.state_dict())
    best_epoch = 0
    started = time.perf_counter()

    for epoch in range(epochs + 1):
        if epoch > 0:
            for _ in range(int(training["discriminator_steps"])):
                optimizer_d.zero_grad(set_to_none=True)
                fake = generator().detach()
                loss_d = -(score(discriminator, target) - score(discriminator, fake))
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), float(training["grad_clip"]))
                optimizer_d.step()
            for _ in range(int(training["generator_steps"])):
                optimizer_g.zero_grad(set_to_none=True)
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                fake = generator()
                loss_g = -score(discriminator, fake)
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), float(training["grad_clip"]))
                optimizer_g.step()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
            scheduler_g.step()
            scheduler_d.step()
        else:
            loss_d = loss_g = torch.tensor(float("nan"), device=device)

        if epoch % int(training["record_every"]) == 0 or epoch == epochs:
            diagnostics = record_state(discriminator, generator, target, validation_states)
            row = {
                "seed": seed,
                "model": name,
                "epoch": epoch,
                "loss_d": scalar(loss_d),
                "loss_g": scalar(loss_g),
                "lr_d": optimizer_d.param_groups[0]["lr"],
                "lr_g": optimizer_g.param_groups[0]["lr"],
                **diagnostics,
            }
            records.append(row)
            if diagnostics["validation_gap"] > best_validation:
                best_validation = diagnostics["validation_gap"]
                best_generator = copy.deepcopy(generator.state_dict())
                best_discriminator = copy.deepcopy(discriminator.state_dict())
                best_epoch = epoch

    elapsed = time.perf_counter() - started
    checkpoints = {
        "best": (best_generator, best_discriminator),
        "final": (copy.deepcopy(generator.state_dict()), copy.deepcopy(discriminator.state_dict())),
    }
    metadata = {
        "seed": seed,
        "model": name,
        "best_epoch": best_epoch,
        "best_validation_gap": best_validation,
        "seconds": elapsed,
        "generator_parameters": trainable_parameter_count(generator),
        "discriminator_parameters": trainable_parameter_count(discriminator),
    }
    return records, checkpoints, metadata


def paired_statistics(rows, metric, higher_is_better=True):
    qsnn = {row["seed"]: row[metric] for row in rows if row["model"] == "qsnn" and math.isfinite(row[metric])}
    vqc = {row["seed"]: row[metric] for row in rows if row["model"] == "vqc" and math.isfinite(row[metric])}
    seeds = sorted(set(qsnn) & set(vqc))
    first = np.array([qsnn[seed] for seed in seeds], dtype=np.float64)
    second = np.array([vqc[seed] for seed in seeds], dtype=np.float64)
    differences = first - second
    if len(seeds) >= 2:
        t_p = float(stats.ttest_rel(first, second).pvalue)
        try:
            wilcoxon_p = float(stats.wilcoxon(first, second).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
    else:
        t_p = wilcoxon_p = float("nan")
    wins = int(np.sum(differences > 0)) if higher_is_better else int(np.sum(differences < 0))
    return {
        "paired_seeds": len(seeds),
        "mean_qsnn_minus_vqc": float(differences.mean()) if len(differences) else float("nan"),
        "qsnn_wins": wins,
        "paired_t_p": t_p,
        "wilcoxon_p": wilcoxon_p,
    }


def aggregate_results(rows):
    metrics = [
        "certified_margin",
        "normalized_margin",
        "boundary_error",
        "pauli_cosine",
        "fake_fidelity",
        "fake_trace_distance",
        "separable_bound_gap",
        "linearity_error",
        "seconds",
    ]
    result = {}
    for model in ("qsnn", "vqc"):
        selected = [row for row in rows if row["model"] == model]
        result[model] = {
            metric: mean_std([row[metric] for row in selected if math.isfinite(row[metric])])
            for metric in metrics
        }
        result[model]["certified_successes"] = int(sum(row["certified_detected"] for row in selected))
        result[model]["runs"] = len(selected)
    result["paired"] = {
        "certified_margin": paired_statistics(rows, "certified_margin", True),
        "normalized_margin": paired_statistics(rows, "normalized_margin", True),
        "boundary_error": paired_statistics(rows, "boundary_error", False),
        "pauli_cosine": paired_statistics(rows, "pauli_cosine", True),
    }
    return result


def plot_training(path: Path, records):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fields = [
        ("adversarial_gap", "Target - generator score"),
        ("validation_gap", "Target - validation separable max"),
        ("fake_trace_distance", "Target-generator trace distance"),
        ("target_output_mass", "Target output mass"),
    ]
    for axis, (field, title) in zip(axes.flat, fields):
        for model, color in (("qsnn", "tab:blue"), ("vqc", "tab:orange")):
            selected = [row for row in records if row["model"] == model]
            epochs = sorted({row["epoch"] for row in selected})
            means, stds = [], []
            for epoch in epochs:
                values = np.array([row[field] for row in selected if row["epoch"] == epoch])
                means.append(values.mean())
                stds.append(values.std(ddof=1) if len(values) > 1 else 0.0)
            means, stds = np.asarray(means), np.asarray(stds)
            axis.plot(epochs, means, color=color, label=model.upper())
            axis.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.18)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_witness_curves(path: Path, curve_rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    best_rows = [row for row in curve_rows if row["variant"] == "best"]
    ps = sorted({row["p"] for row in best_rows})
    panels = [
        ("witness_expectation", [(1.0 - 3.0 * p) / 4.0 for p in ps], "Raw certified witness"),
        ("normalized_witness_expectation", [(1.0 - 3.0 * p) / 2.0 for p in ps], r"Normalized by $\|W\|_\infty$"),
    ]
    for axis, (field, theory, title) in zip(axes, panels):
        axis.plot(ps, theory, "k--", label="Analytic witness")
        for model, color in (("qsnn", "tab:blue"), ("vqc", "tab:orange")):
            means, stds = [], []
            for p in ps:
                values = np.array([row[field] for row in best_rows if row["model"] == model and row["p"] == p])
                means.append(values.mean())
                stds.append(values.std(ddof=1) if len(values) > 1 else 0.0)
            means, stds = np.asarray(means), np.asarray(stds)
            axis.plot(ps, means, color=color, label=model.upper())
            axis.fill_between(ps, means - stds, means + stds, color=color, alpha=0.18)
        axis.axhline(0.0, color="gray", linewidth=1)
        axis.axvline(1.0 / 3.0, color="gray", linestyle=":", label=r"$p=1/3$")
        axis.set_title(title)
        axis.set_xlabel("Werner p")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"Witness expectation $Tr(W\rho_W(p))$")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pauli(path: Path, coefficient_rows):
    labels = ["II", "XX", "YY", "ZZ"]
    best_rows = [row for row in coefficient_rows if row["variant"] == "best"]
    x = np.arange(len(labels))
    width = 0.25
    fig, axis = plt.subplots(figsize=(8, 5.5))
    theory = [next(row["theory_normalized"] for row in best_rows if row["pauli"] == label) for label in labels]
    axis.bar(x - width, theory, width, color="black", alpha=0.65, label="Analytic")
    for offset, (model, color) in enumerate((("qsnn", "tab:blue"), ("vqc", "tab:orange"))):
        means = []
        stds = []
        for label in labels:
            values = np.array([row["learned_normalized"] for row in best_rows if row["model"] == model and row["pauli"] == label])
            means.append(values.mean())
            stds.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        axis.bar(x + offset * width, means, width, yerr=stds, color=color, alpha=0.8, label=model.upper(), capsize=3)
    axis.set_xticks(x, labels)
    axis.set_ylabel(r"Pauli coefficient after $\|W\|_\infty$ normalization")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/entanglement_witness_qgan.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device")
    parser.add_argument("--skip-capacity", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with (ROOT / args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        config["training"]["lr_decay_start"] = min(
            int(config["training"]["lr_decay_start"]), max(1, args.epochs // 2)
        )
    if args.seeds is not None:
        config["experiment"]["seeds"] = args.seeds
    if args.device:
        config["runtime"]["device"] = args.device
    output_dir = ROOT / config["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    dtype = torch.float64 if config["runtime"]["dtype"] == "float64" else torch.float32
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    device = torch.device(config["runtime"]["device"])
    seeds = list(config["experiment"]["seeds"])
    capacity_rows = []
    records, result_rows, curve_rows, coefficient_rows = [], [], [], []
    checkpoint_payload = {}
    dense_ps = torch.linspace(
        0.0, 1.0, int(config["target"]["dense_points"]), dtype=dtype, device=device
    )

    if args.postprocess_only:
        with (output_dir / "training_records.csv").open("r", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        payload = torch.load(output_dir / "checkpoints.pt", map_location=device, weights_only=False)
        for seed in seeds:
            for model in ("qsnn", "vqc"):
                entry = payload[f"{model}_seed{seed}"]
                metadata = entry["metadata"]
                for variant in ("best", "final"):
                    generator = build_generator(config["generator"], dtype, device)
                    discriminator = build_discriminator(model, config["discriminators"][model], dtype, device)
                    generator.load_state_dict(entry[variant]["generator"])
                    discriminator.load_state_dict(entry[variant]["discriminator"])
                    target = werner_state(
                        float(config["target"]["werner_p"]), dtype=complex_dtype, device=device
                    )
                    result, curves, coefficients, _observable, _witness = evaluate_checkpoint(
                        discriminator, generator, target, dense_ps, config, complex_dtype, variant, seed, model
                    )
                    result.update(metadata)
                    result_rows.append(result)
                    curve_rows.extend(curves)
                    coefficient_rows.extend(coefficients)
        best_results = [row for row in result_rows if row["variant"] == "best"]
        aggregate = aggregate_results(best_results)
        write_csv(output_dir / "witness_results.csv", result_rows)
        write_csv(output_dir / "witness_curves.csv", curve_rows)
        write_csv(output_dir / "pauli_coefficients.csv", coefficient_rows)
        with (output_dir / "aggregate.json").open("w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, ensure_ascii=False, indent=2)
        plot_witness_curves(output_dir / "witness_curves.png", curve_rows)
        plot_pauli(output_dir / "pauli_coefficients.png", coefficient_rows)
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))
        return

    for seed in seeds:
        torch.manual_seed(seed)
        initial_generator = build_generator(config["generator"], dtype, device)
        initial_state = copy.deepcopy(initial_generator.state_dict())
        if not args.skip_capacity:
            for target_p in (
                float(config["target"]["separable_control_p"]),
                float(config["target"]["boundary_p"]),
            ):
                capacity_rows.append(
                    {"seed": seed, **capacity_check(initial_state, target_p, config, complex_dtype, dtype, device)}
                )
        validation_states = random_product_states(
            int(config["training"]["validation_product_states"]), dtype, device, seed + 50_000
        )
        for model in ("qsnn", "vqc"):
            model_records, checkpoints, metadata = train_one(
                model, seed, initial_state, config, dtype, device, validation_states
            )
            records.extend(model_records)
            checkpoint_payload[f"{model}_seed{seed}"] = {"metadata": metadata}
            for variant, (generator_state, discriminator_state) in checkpoints.items():
                generator = build_generator(config["generator"], dtype, device)
                discriminator = build_discriminator(model, config["discriminators"][model], dtype, device)
                generator.load_state_dict(generator_state)
                discriminator.load_state_dict(discriminator_state)
                target = werner_state(
                    float(config["target"]["werner_p"]), dtype=complex_dtype, device=device
                )
                result, curves, coefficients, observable, witness = evaluate_checkpoint(
                    discriminator, generator, target, dense_ps, config, complex_dtype, variant, seed, model
                )
                result.update(metadata)
                result_rows.append(result)
                curve_rows.extend(curves)
                coefficient_rows.extend(coefficients)
                checkpoint_payload[f"{model}_seed{seed}"][variant] = {
                    "generator": generator_state,
                    "discriminator": discriminator_state,
                    "observable": observable,
                    "witness": witness,
                }
            print(
                f"seed={seed} model={model} best_epoch={metadata['best_epoch']} "
                f"seconds={metadata['seconds']:.1f}",
                flush=True,
            )

    best_results = [row for row in result_rows if row["variant"] == "best"]
    aggregate = aggregate_results(best_results)
    write_csv(output_dir / "capacity_checks.csv", capacity_rows)
    write_csv(output_dir / "training_records.csv", records)
    write_csv(output_dir / "witness_results.csv", result_rows)
    write_csv(output_dir / "witness_curves.csv", curve_rows)
    write_csv(output_dir / "pauli_coefficients.csv", coefficient_rows)
    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    torch.save(checkpoint_payload, output_dir / "checkpoints.pt")
    plot_training(output_dir / "training_curves.png", records)
    plot_witness_curves(output_dir / "witness_curves.png", curve_rows)
    plot_pauli(output_dir / "pauli_coefficients.png", coefficient_rows)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
