"""Follow-up entanglement-witness benchmark with tuning, EMA, and shot noise.

The protocol deliberately separates calibration, validation, and formal seeds so
that discriminator hyperparameters are frozen before the reported comparison.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qsnn_matplotlib"))

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

from qgan.entanglement_witness import effective_observable, observable_score, pauli_coefficients  # noqa: E402
from qgan.metrics import density_fidelity, trace_distance, trainable_parameter_count  # noqa: E402
from qgan.mixed_states import negativity, werner_state  # noqa: E402
from scripts.run_entanglement_witness_qgan import (  # noqa: E402
    build_discriminator,
    build_generator,
    evaluate_checkpoint,
    mean_std,
    random_product_states,
    scalar,
    score,
    write_csv,
)


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_json(path: Path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def runtime_config(config, model: str, candidate: dict, epochs: int, target_p: float) -> dict:
    discriminator = copy.deepcopy(config["base_discriminators"][model])
    if model == "qsnn":
        discriminator["target_layer_mass"] = float(candidate["target_layer_mass"])
    training = copy.deepcopy(config["training"])
    training.update(
        {
            "epochs": int(epochs),
            "lr_d": float(candidate["lr_d"]),
            "generator_steps": int(candidate["generator_steps"]),
            "lr_decay_start": int(float(training["lr_decay_fraction"]) * epochs),
        }
    )
    return {
        "generator": copy.deepcopy(config["generator"]),
        "discriminators": {model: discriminator},
        "training": training,
        "target": {"werner_p": float(target_p), "dense_points": int(config["target"]["dense_points"])},
        "certification": copy.deepcopy(config["certification"]),
    }


def update_ema(ema_state: dict[str, torch.Tensor], state: dict[str, torch.Tensor], decay: float):
    with torch.no_grad():
        for name, value in state.items():
            if value.is_floating_point() or value.is_complex():
                ema_state[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_state[name] = value.detach().clone()


def diagnostics(discriminator, generator, target, validation_states) -> dict:
    observable = effective_observable(discriminator)
    target_score = scalar(observable_score(observable, target))
    validation_scores = observable_score(observable, validation_states.to(observable.dtype))
    validation_max = scalar(validation_scores.max())
    gap = target_score - validation_max
    identity = torch.eye(observable.shape[-1], dtype=observable.dtype, device=observable.device)
    proxy_witness = validation_max * identity - observable
    proxy_norm = scalar(torch.linalg.eigvalsh(proxy_witness).abs().max())
    fake = generator().detach()
    fake_score = scalar(observable_score(observable, fake))
    target_output = discriminator(target)
    fake_output = discriminator(fake)
    return {
        "target_score": target_score,
        "fake_score": fake_score,
        "adversarial_gap": target_score - fake_score,
        "validation_sep_max": validation_max,
        "validation_gap": gap,
        "normalized_validation_gap": gap / max(proxy_norm, 1.0e-12),
        "validation_witness_norm": proxy_norm,
        "fake_fidelity": scalar(density_fidelity(target, fake)),
        "fake_trace_distance": scalar(trace_distance(target, fake)),
        "fake_negativity": scalar(negativity(fake)),
        "target_output_mass": scalar(target_output["output_mass"]),
        "fake_output_mass": scalar(fake_output["output_mass"]),
    }


def train_one(
    model: str,
    candidate: dict,
    seed: int,
    initial_generator_state: dict,
    config: dict,
    dtype: torch.dtype,
    device: torch.device,
    validation_states: torch.Tensor,
):
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    generator = build_generator(config["generator"], dtype, device)
    generator.load_state_dict(copy.deepcopy(initial_generator_state))
    torch.manual_seed(seed + 10_000)
    discriminator = build_discriminator(model, config["discriminators"][model], dtype, device)
    ema_discriminator = build_discriminator(model, config["discriminators"][model], dtype, device)
    ema_state = copy.deepcopy(discriminator.state_dict())
    ema_discriminator.load_state_dict(ema_state)
    target = werner_state(float(config["target"]["werner_p"]), dtype=complex_dtype, device=device)
    training = config["training"]
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(training["lr_g"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=float(training["lr_d"]))
    epochs = int(training["epochs"])
    decay_start = int(training["lr_decay_start"])
    decay_factor = float(training["lr_decay_factor"])

    def schedule(epoch):
        if epoch < decay_start:
            return 1.0
        return max(
            decay_factor,
            1.0 - (1.0 - decay_factor) * (epoch - decay_start) / max(1, epochs - decay_start),
        )

    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, schedule)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, schedule)
    best_scores = {"instant": -math.inf, "ema": -math.inf}
    best_epochs = {"instant": 0, "ema": 0}
    best_states = {}
    records = []
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
                update_ema(ema_state, discriminator.state_dict(), float(training["ema_decay"]))
            for _ in range(int(training["generator_steps"])):
                optimizer_g.zero_grad(set_to_none=True)
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                loss_g = -score(discriminator, generator())
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), float(training["grad_clip"]))
                optimizer_g.step()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
            scheduler_g.step()
            scheduler_d.step()
        else:
            loss_d = loss_g = torch.tensor(float("nan"), dtype=dtype, device=device)

        if epoch % int(training["record_every"]) == 0 or epoch == epochs:
            ema_discriminator.load_state_dict(ema_state)
            for source, active in (("instant", discriminator), ("ema", ema_discriminator)):
                values = diagnostics(active, generator, target, validation_states)
                records.append(
                    {
                        "seed": seed,
                        "model": model,
                        "candidate": candidate["name"],
                        "target_p": float(config["target"]["werner_p"]),
                        "source": source,
                        "epoch": epoch,
                        "loss_d": scalar(loss_d),
                        "loss_g": scalar(loss_g),
                        "lr_d": optimizer_d.param_groups[0]["lr"],
                        "lr_g": optimizer_g.param_groups[0]["lr"],
                        **values,
                    }
                )
                selection_score = values["normalized_validation_gap"]
                if selection_score > best_scores[source]:
                    best_scores[source] = selection_score
                    best_epochs[source] = epoch
                    best_states[source] = (
                        copy.deepcopy(generator.state_dict()),
                        copy.deepcopy(active.state_dict()),
                    )

    ema_discriminator.load_state_dict(ema_state)
    checkpoints = {
        "best_instant": best_states["instant"],
        "best_ema": best_states["ema"],
        "final_instant": (copy.deepcopy(generator.state_dict()), copy.deepcopy(discriminator.state_dict())),
        "final_ema": (copy.deepcopy(generator.state_dict()), copy.deepcopy(ema_discriminator.state_dict())),
    }
    metadata = {
        "seed": seed,
        "model": model,
        "candidate": candidate["name"],
        "target_p": float(config["target"]["werner_p"]),
        "best_epoch_instant": best_epochs["instant"],
        "best_epoch_ema": best_epochs["ema"],
        "best_normalized_validation_gap_instant": best_scores["instant"],
        "best_normalized_validation_gap_ema": best_scores["ema"],
        "seconds": time.perf_counter() - started,
        "generator_parameters": trainable_parameter_count(generator),
        "discriminator_parameters": trainable_parameter_count(discriminator),
    }
    return records, checkpoints, metadata


def evaluate_variants(
    model,
    candidate,
    seed,
    checkpoints,
    metadata,
    config,
    dtype,
    device,
    variants=None,
):
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    dense_ps = torch.linspace(0.0, 1.0, int(config["target"]["dense_points"]), dtype=dtype, device=device)
    target = werner_state(float(config["target"]["werner_p"]), dtype=complex_dtype, device=device)
    results, curves, coefficients, witnesses = [], [], [], {}
    variants = list(checkpoints) if variants is None else variants
    for variant in variants:
        generator_state, discriminator_state = checkpoints[variant]
        generator = build_generator(config["generator"], dtype, device)
        discriminator = build_discriminator(model, config["discriminators"][model], dtype, device)
        generator.load_state_dict(generator_state)
        discriminator.load_state_dict(discriminator_state)
        result, one_curve, one_coefficients, observable, witness = evaluate_checkpoint(
            discriminator,
            generator,
            target,
            dense_ps,
            config,
            complex_dtype,
            variant,
            seed,
            model,
        )
        result.update(metadata)
        results.append(result)
        curves.extend([{**row, "candidate": candidate["name"], "target_p": metadata["target_p"]} for row in one_curve])
        coefficients.extend(
            [{**row, "candidate": candidate["name"], "target_p": metadata["target_p"]} for row in one_coefficients]
        )
        witnesses[variant] = {"observable": observable, "witness": witness}
    return results, curves, coefficients, witnesses


def choose_candidates(rows: list[dict], top_k: int) -> dict[str, list[dict]]:
    selected = {}
    for model in ("qsnn", "vqc"):
        model_rows = [row for row in rows if row["model"] == model and row["variant"] in ("best_ema", "best_instant")]
        candidates = sorted({row["candidate"] for row in model_rows})
        ranked = []
        for candidate in candidates:
            source_scores = []
            for variant in ("best_ema", "best_instant"):
                values = [float(row["normalized_margin"]) for row in model_rows if row["candidate"] == candidate and row["variant"] == variant]
                source_scores.append((float(np.mean(values)), variant, float(np.std(values, ddof=1)) if len(values) > 1 else 0.0))
            mean, variant, std = max(source_scores, key=lambda item: item[0])
            ranked.append({"candidate": candidate, "checkpoint_source": variant, "mean_normalized_margin": mean, "std_normalized_margin": std})
        selected[model] = sorted(ranked, key=lambda item: item["mean_normalized_margin"], reverse=True)[:top_k]
    return selected


def candidate_by_name(config: dict, model: str, name: str) -> dict:
    return next(candidate for candidate in config["calibration_candidates"][model] if candidate["name"] == name)


def run_search_stage(config, stage: str, selections=None):
    output_dir = ROOT / config["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64 if config["runtime"]["dtype"] == "float64" else torch.float32
    device = torch.device(config["runtime"]["device"])
    seeds = [int(seed) for seed in config["seeds"][stage]]
    epochs = int(config["training"][f"{stage}_epochs"])
    target_p = float(config["target"]["calibration_p"])
    training_rows, result_rows, curve_rows, coefficient_rows = [], [], [], []

    for seed in seeds:
        torch.manual_seed(seed)
        initial_generator = build_generator(config["generator"], dtype, device)
        initial_state = copy.deepcopy(initial_generator.state_dict())
        validation_states = random_product_states(
            int(config["training"]["validation_product_states"]), dtype, device, seed + 50_000
        )
        for model in ("qsnn", "vqc"):
            if selections is None:
                candidates = config["calibration_candidates"][model]
            else:
                candidates = [candidate_by_name(config, model, item["candidate"]) for item in selections[model]]
            for candidate in candidates:
                run_config = runtime_config(config, model, candidate, epochs, target_p)
                records, checkpoints, metadata = train_one(
                    model, candidate, seed, initial_state, run_config, dtype, device, validation_states
                )
                results, curves, coefficients, _ = evaluate_variants(
                    model, candidate, seed, checkpoints, metadata, run_config, dtype, device
                )
                training_rows.extend(records)
                result_rows.extend(results)
                curve_rows.extend(curves)
                coefficient_rows.extend(coefficients)
                write_csv(output_dir / f"{stage}_training.csv", training_rows)
                write_csv(output_dir / f"{stage}_results.csv", result_rows)
                write_csv(output_dir / f"{stage}_curves.csv", curve_rows)
                write_csv(output_dir / f"{stage}_coefficients.csv", coefficient_rows)
                print(
                    f"stage={stage} seed={seed} model={model} candidate={candidate['name']} "
                    f"seconds={metadata['seconds']:.1f}",
                    flush=True,
                )
    return result_rows


def run_calibration(config):
    output_dir = ROOT / config["experiment"]["output_dir"]
    rows = run_search_stage(config, "calibration")
    selection = choose_candidates(rows, int(config["selection"]["calibration_top_k"]))
    save_json(output_dir / "calibration_selection.json", selection)
    plot_selection(output_dir / "calibration_selection.png", rows, "Calibration seeds")
    print(json.dumps(selection, indent=2), flush=True)


def run_validation(config):
    output_dir = ROOT / config["experiment"]["output_dir"]
    with (output_dir / "calibration_selection.json").open("r", encoding="utf-8") as handle:
        calibration_selection = json.load(handle)
    rows = run_search_stage(config, "validation", calibration_selection)
    frozen = choose_candidates(rows, 1)
    save_json(output_dir / "frozen_selection.json", frozen)
    plot_selection(output_dir / "validation_selection.png", rows, "Validation seeds")
    print(json.dumps(frozen, indent=2), flush=True)


def run_formal(config):
    output_dir = ROOT / config["experiment"]["output_dir"]
    with (output_dir / "frozen_selection.json").open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    dtype = torch.float64 if config["runtime"]["dtype"] == "float64" else torch.float32
    device = torch.device(config["runtime"]["device"])
    epochs = int(config["training"]["formal_epochs"])
    training_rows, result_rows, curve_rows, coefficient_rows = [], [], [], []
    witness_payload = {}

    for target_index, target_p in enumerate(config["target"]["formal_ps"]):
        for seed in [int(value) for value in config["seeds"]["formal"]]:
            torch.manual_seed(seed)
            initial_generator = build_generator(config["generator"], dtype, device)
            initial_state = copy.deepcopy(initial_generator.state_dict())
            validation_states = random_product_states(
                int(config["training"]["validation_product_states"]), dtype, device, seed + 50_000 + 1000 * target_index
            )
            for model in ("qsnn", "vqc"):
                choice = frozen[model][0]
                candidate = candidate_by_name(config, model, choice["candidate"])
                source = choice["checkpoint_source"]
                final_source = source.replace("best_", "final_")
                run_config = runtime_config(config, model, candidate, epochs, float(target_p))
                records, checkpoints, metadata = train_one(
                    model, candidate, seed, initial_state, run_config, dtype, device, validation_states
                )
                results, curves, coefficients, witnesses = evaluate_variants(
                    model,
                    candidate,
                    seed,
                    checkpoints,
                    metadata,
                    run_config,
                    dtype,
                    device,
                    variants=[source, final_source],
                )
                for result in results:
                    result["primary"] = int(result["variant"] == source)
                training_rows.extend(records)
                result_rows.extend(results)
                curve_rows.extend(curves)
                coefficient_rows.extend(coefficients)
                key = f"p{float(target_p):.6f}_{model}_seed{seed}"
                witness_payload[key] = {
                    "target_p": float(target_p),
                    "model": model,
                    "seed": seed,
                    "candidate": candidate["name"],
                    "variant": source,
                    "witness": witnesses[source]["witness"],
                }
                write_csv(output_dir / "formal_training.csv", training_rows)
                write_csv(output_dir / "formal_results.csv", result_rows)
                write_csv(output_dir / "formal_curves.csv", curve_rows)
                write_csv(output_dir / "formal_coefficients.csv", coefficient_rows)
                torch.save(witness_payload, output_dir / "formal_witnesses.pt")
                print(
                    f"stage=formal p={float(target_p):.3f} seed={seed} model={model} "
                    f"candidate={candidate['name']} source={source} seconds={metadata['seconds']:.1f}",
                    flush=True,
                )
    primary_rows = [row for row in result_rows if int(row["primary"]) == 1]
    aggregate = aggregate_formal(primary_rows)
    save_json(output_dir / "formal_aggregate.json", aggregate)
    plot_formal(output_dir / "formal_summary.png", primary_rows)
    plot_formal_training(output_dir / "formal_training.png", training_rows, frozen)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


def paired(rows, metric, higher=True):
    qsnn = {(float(row["target_p"]), int(row["seed"])): float(row[metric]) for row in rows if row["model"] == "qsnn"}
    vqc = {(float(row["target_p"]), int(row["seed"])): float(row[metric]) for row in rows if row["model"] == "vqc"}
    keys = sorted(set(qsnn) & set(vqc))
    left = np.asarray([qsnn[key] for key in keys])
    right = np.asarray([vqc[key] for key in keys])
    difference = left - right
    if len(keys) > 1:
        t_p = float(stats.ttest_rel(left, right).pvalue)
        try:
            w_p = float(stats.wilcoxon(left, right).pvalue)
        except ValueError:
            w_p = 1.0
    else:
        t_p = w_p = float("nan")
    return {
        "pairs": len(keys),
        "mean_qsnn_minus_vqc": float(difference.mean()),
        "qsnn_wins": int(np.sum(difference > 0 if higher else difference < 0)),
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
    }


def aggregate_formal(rows):
    metrics = ["certified_margin", "normalized_margin", "boundary_error", "pauli_cosine", "target_output_mass", "seconds"]
    aggregate = {}
    for target_p in sorted({float(row["target_p"]) for row in rows}):
        target_rows = [row for row in rows if float(row["target_p"]) == target_p]
        target_result = {}
        for model in ("qsnn", "vqc"):
            model_rows = [row for row in target_rows if row["model"] == model]
            target_result[model] = {metric: mean_std([float(row[metric]) for row in model_rows]) for metric in metrics}
            target_result[model]["certified_successes"] = sum(int(row["certified_detected"]) for row in model_rows)
            target_result[model]["runs"] = len(model_rows)
        target_result["paired"] = {
            metric: paired(target_rows, metric, higher=metric not in ("boundary_error", "seconds"))
            for metric in ("certified_margin", "normalized_margin", "boundary_error", "pauli_cosine")
        }
        aggregate[str(target_p)] = target_result
    return aggregate


def pauli_expectations(rho: torch.Tensor) -> dict[str, float]:
    dtype = rho.dtype
    device = rho.device
    matrices = {
        "I": torch.eye(2, dtype=dtype, device=device),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device),
        "Z": torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device),
    }
    return {
        first + second: scalar(torch.trace(rho @ torch.kron(matrices[first], matrices[second])).real)
        for first in "IXYZ"
        for second in "IXYZ"
    }


def run_shots(config):
    output_dir = ROOT / config["experiment"]["output_dir"]
    payload = torch.load(output_dir / "formal_witnesses.pt", map_location="cpu", weights_only=False)
    repetitions = int(config["shots"]["repetitions"])
    z_value = float(config["shots"]["confidence_z"])
    rows = []
    for key, entry in payload.items():
        witness = entry["witness"].to(torch.complex128)
        coefficients = pauli_coefficients(witness)
        rho = werner_state(float(entry["target_p"]), dtype=torch.complex128, device="cpu")
        expectations = pauli_expectations(rho)
        true_value = scalar(observable_score(witness, rho))
        for shots in config["shots"]["per_pauli_setting"]:
            rng = np.random.default_rng(
                900_000 + int(round(1000 * float(entry["target_p"]))) + 10_000 * int(entry["seed"])
                + (0 if entry["model"] == "qsnn" else 1_000) + int(shots)
            )
            estimates = np.full(repetitions, coefficients["II"], dtype=np.float64)
            variance = 0.0
            for label, coefficient in coefficients.items():
                if label == "II" or abs(coefficient) < 1.0e-14:
                    continue
                mu = float(np.clip(expectations[label], -1.0, 1.0))
                positive = rng.binomial(int(shots), 0.5 * (1.0 + mu), size=repetitions)
                estimates += coefficient * (2.0 * positive / int(shots) - 1.0)
                variance += coefficient * coefficient * (1.0 - mu * mu) / int(shots)
            standard_error = math.sqrt(max(variance, 0.0))
            detected = estimates + z_value * standard_error < 0.0
            rows.append(
                {
                    "target_p": float(entry["target_p"]),
                    "model": entry["model"],
                    "seed": int(entry["seed"]),
                    "candidate": entry["candidate"],
                    "variant": entry["variant"],
                    "shots_per_pauli": int(shots),
                    "repetitions": repetitions,
                    "true_witness_expectation": true_value,
                    "theoretical_standard_error": standard_error,
                    "mean_estimate": float(estimates.mean()),
                    "empirical_standard_deviation": float(estimates.std(ddof=1)),
                    "certification_probability": float(detected.mean()),
                }
            )
    write_csv(output_dir / "shots_results.csv", rows)
    aggregate = aggregate_shots(rows)
    save_json(output_dir / "shots_aggregate.json", aggregate)
    plot_shots(output_dir / "shot_noise.png", rows)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


def aggregate_shots(rows):
    result = {}
    for target_p in sorted({float(row["target_p"]) for row in rows}):
        result[str(target_p)] = {}
        for model in ("qsnn", "vqc"):
            result[str(target_p)][model] = {}
            for shots in sorted({int(row["shots_per_pauli"]) for row in rows}):
                values = [float(row["certification_probability"]) for row in rows if float(row["target_p"]) == target_p and row["model"] == model and int(row["shots_per_pauli"]) == shots]
                result[str(target_p)][model][str(shots)] = mean_std(values)
    return result


def plot_selection(path: Path, rows: list[dict], title: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis, model in zip(axes, ("qsnn", "vqc")):
        subset = [row for row in rows if row["model"] == model and row["variant"] in ("best_ema", "best_instant")]
        names = sorted({row["candidate"] for row in subset})
        x = np.arange(len(names))
        for offset, (variant, label) in enumerate((("best_instant", "Instant"), ("best_ema", "EMA"))):
            means = [np.mean([float(row["normalized_margin"]) for row in subset if row["candidate"] == name and row["variant"] == variant]) for name in names]
            axis.bar(x + (offset - 0.5) * 0.36, means, 0.36, label=label)
        axis.set_xticks(x, names, rotation=35, ha="right")
        axis.set_ylabel("Certified normalized margin")
        axis.set_title(f"{title}: {model.upper()}")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_formal(path: Path, rows: list[dict]):
    target_ps = sorted({float(row["target_p"]) for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fields = [
        ("normalized_margin", "Certified normalized margin"),
        ("boundary_error", r"Boundary error $|\hat p-1/3|$"),
        ("pauli_cosine", "Pauli cosine to analytic witness"),
    ]
    x = np.arange(len(target_ps))
    for axis, (field, label) in zip(axes, fields):
        for offset, (model, color) in enumerate((("qsnn", "tab:blue"), ("vqc", "tab:orange"))):
            means, errors = [], []
            for target_p in target_ps:
                values = [float(row[field]) for row in rows if row["model"] == model and float(row["target_p"]) == target_p]
                means.append(np.mean(values))
                errors.append(np.std(values, ddof=1))
            axis.bar(x + (offset - 0.5) * 0.36, means, 0.36, yerr=errors, label=model.upper(), color=color, capsize=3)
        axis.set_xticks(x, [f"p={p:g}" for p in target_ps])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_formal_training(path: Path, rows: list[dict], frozen: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for axis, target_p in zip(axes, sorted({float(row["target_p"]) for row in rows})):
        for model, color in (("qsnn", "tab:blue"), ("vqc", "tab:orange")):
            source = frozen[model][0]["checkpoint_source"].replace("best_", "")
            subset = [row for row in rows if row["model"] == model and float(row["target_p"]) == target_p and row["source"] == source]
            epochs = sorted({int(row["epoch"]) for row in subset})
            means = [np.mean([float(row["normalized_validation_gap"]) for row in subset if int(row["epoch"]) == epoch]) for epoch in epochs]
            axis.plot(epochs, means, color=color, label=f"{model.upper()} ({source})")
        axis.set_title(f"Training target p={target_p:g}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Normalized validation gap")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_shots(path: Path, rows: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, target_p in zip(axes, sorted({float(row["target_p"]) for row in rows})):
        styles = {
            "qsnn": ("tab:blue", "o", "-"),
            "vqc": ("tab:orange", "s", "--"),
        }
        for model in ("qsnn", "vqc"):
            color, marker, linestyle = styles[model]
            shots = sorted({int(row["shots_per_pauli"]) for row in rows})
            means = [np.mean([float(row["certification_probability"]) for row in rows if row["model"] == model and float(row["target_p"]) == target_p and int(row["shots_per_pauli"]) == count]) for count in shots]
            axis.plot(shots, means, marker=marker, linestyle=linestyle, color=color, label=model.upper())
        axis.set_xscale("log")
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(f"Target p={target_p:g}")
        axis.set_xlabel("Shots per Pauli setting")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Probability that upper 95% CI is below zero")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/entanglement_witness_followup.yaml")
    parser.add_argument("--stage", choices=("calibration", "validation", "formal", "shots", "all"), default="all")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, help="Override the selected training stage's epoch count.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override the selected training stage's seeds.")
    return parser.parse_args()


def main():
    args = parse_args()
    with (ROOT / args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    if args.device:
        config["runtime"]["device"] = args.device
    if args.stage in ("calibration", "validation", "formal"):
        if args.epochs is not None:
            config["training"][f"{args.stage}_epochs"] = args.epochs
        if args.seeds is not None:
            config["seeds"][args.stage] = args.seeds
    output_dir = ROOT / config["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    stages = ("calibration", "validation", "formal", "shots") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "calibration":
            run_calibration(config)
        elif stage == "validation":
            run_validation(config)
        elif stage == "formal":
            run_formal(config)
        else:
            run_shots(config)


if __name__ == "__main__":
    main()
