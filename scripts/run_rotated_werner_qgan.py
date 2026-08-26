"""Random-local-rotation Werner-family QSNN-QGAN versus VQC-QGAN benchmark."""

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
from qgan.metrics import (
    density_fidelity,
    physicality_diagnostics,
    purity,
    trace_distance,
    trainable_parameter_count,
)
from qgan.mixed_state_discriminators import (
    ConditionalAncillaVQCDiscriminator,
    ConditionalLayeredQSNNDiscriminator,
)
from qgan.mixed_states import negativity, werner_state
from qgan.rotations import (
    condition_grid,
    local_bloch_vectors,
    pauli_tensor,
    random_quaternions,
    rotate_second_qubit,
    rotated_werner_state,
    stress_quaternions,
)


STAGE_ORDER = ("capacity", "smoke", "calibration", "validation", "formal")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "random_local_rotated_werner.yaml"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stage", choices=(*STAGE_ORDER, "report", "all"), default="all")
    parser.add_argument("--epochs", type=int, help="Override GAN epochs for the requested run")
    parser.add_argument("--capacity-steps", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, help="Override seeds for a single stage")
    parser.add_argument("--device")
    parser.add_argument("--allow-capacity-failure", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar(value):
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def mean_std(values):
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return {
        "mean": scalar(tensor.mean()),
        "std": scalar(tensor.std(unbiased=True)) if tensor.numel() > 1 else 0.0,
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
    return rows


def gradient_norm(parameters):
    gradients = [p.grad.detach().norm() for p in parameters if p.grad is not None]
    return scalar(torch.linalg.vector_norm(torch.stack(gradients))) if gradients else 0.0


def build_generator(config, dtype, device):
    return ConditionalPurifiedPQCGenerator(
        system_qubits=int(config["system_qubits"]),
        environment_qubits=int(config["environment_qubits"]),
        n_layers=int(config["layers"]),
        condition_dim=int(config.get("condition_dim", 5)),
        condition_feature_map=config.get("condition_feature_map", "linear"),
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
            condition_dim=5,
            real_dtype=dtype,
        ).to(device)
    if name == "vqc":
        return ConditionalAncillaVQCDiscriminator(
            system_qubits=2,
            n_layers=int(config["layers"]),
            condition_dim=5,
            real_dtype=dtype,
        ).to(device)
    raise ValueError(f"unknown discriminator: {name}")


def target_states(conditions, complex_dtype):
    return rotated_werner_state(
        conditions[:, 0], conditions[:, 1:], dtype=complex_dtype, device=conditions.device
    )


def discriminator_value(real_output, fake_output):
    return 0.5 + 0.25 * (
        real_output["z_expectation"].mean() - fake_output["z_expectation"].mean()
    )


def build_grids(config, dtype, device):
    conditions = config["conditions"]
    rotations = conditions["rotations"]
    train_q = random_quaternions(
        int(rotations["train_count"]), int(rotations["train_seed"]), dtype=dtype, device=device
    )
    validation_q = random_quaternions(
        int(rotations["validation_count"]),
        int(rotations["validation_seed"]),
        dtype=dtype,
        device=device,
    )
    test_q = random_quaternions(
        int(rotations["test_count"]), int(rotations["test_seed"]), dtype=dtype, device=device
    )
    stress_q = stress_quaternions(dtype=dtype, device=device)
    train_p = torch.tensor(conditions["train_p"], dtype=dtype, device=device)
    interp_p = torch.tensor(conditions["interpolation_p"], dtype=dtype, device=device)
    separable_p = torch.tensor(conditions["separable_p"], dtype=dtype, device=device)
    return {
        "train": condition_grid(train_p, train_q),
        "validation": condition_grid(train_p, validation_q),
        "rotation_test": condition_grid(train_p, test_q),
        "joint_ood": condition_grid(interp_p, test_q),
        "stress": condition_grid(interp_p, stress_q),
        "separable": condition_grid(separable_p, test_q),
        "smoke": condition_grid(train_p[:2], train_q[:2]),
        "pools": {
            "train_q": train_q,
            "validation_q": validation_q,
            "test_q": test_q,
            "stress_q": stress_q,
        },
    }


@torch.no_grad()
def quick_metrics(generator, conditions, complex_dtype):
    target = target_states(conditions, complex_dtype)
    generated = generator(conditions)
    fidelity = density_fidelity(target, generated)
    distance = trace_distance(target, generated)
    return {
        "mean_fidelity": scalar(fidelity.mean()),
        "min_fidelity": scalar(fidelity.min()),
        "mean_trace_distance": scalar(distance.mean()),
        "max_trace_distance": scalar(distance.max()),
    }


@torch.no_grad()
def evaluate_states(generated, conditions, complex_dtype, reference_generated=None):
    target = target_states(conditions, complex_dtype)
    fidelity = density_fidelity(target, generated)
    distance = trace_distance(target, generated)
    target_purity = purity(target)
    generated_purity = purity(generated)
    target_negativity = negativity(target)
    generated_negativity = negativity(generated)
    tensor_error = (pauli_tensor(target) - pauli_tensor(generated)).abs().mean(dim=(-2, -1))
    bloch_error = (local_bloch_vectors(target) - local_bloch_vectors(generated)).abs().mean(dim=-1)

    unique_p, inverse = torch.unique(conditions[:, 0], sorted=True, return_inverse=True)
    identity_q = torch.zeros(unique_p.shape[0], 4, dtype=conditions.dtype, device=conditions.device)
    identity_q[:, 0] = 1.0
    if reference_generated is None:
        reference_generated = werner_state(unique_p, dtype=complex_dtype, device=conditions.device)
    generated_reference_rows = reference_generated[inverse]
    covariant_prediction = rotate_second_qubit(generated_reference_rows, conditions[:, 1:])
    covariance_error = trace_distance(generated, covariant_prediction)
    target_reference_rows = werner_state(conditions[:, 0], dtype=complex_dtype, device=conditions.device)
    generated_orbit_distance = trace_distance(generated, generated_reference_rows)
    target_orbit_distance = trace_distance(target, target_reference_rows)
    diversity_error = (generated_orbit_distance - target_orbit_distance).abs()
    diagnostics = physicality_diagnostics(generated, include_min_eigenvalue=True)

    summary = {
        "mean_fidelity": scalar(fidelity.mean()),
        "min_fidelity": scalar(fidelity.min()),
        "mean_trace_distance": scalar(distance.mean()),
        "max_trace_distance": scalar(distance.max()),
        "purity_mae": scalar((generated_purity - target_purity).abs().mean()),
        "negativity_mae": scalar((generated_negativity - target_negativity).abs().mean()),
        "pauli_tensor_mae": scalar(tensor_error.mean()),
        "local_bloch_mae": scalar(bloch_error.mean()),
        "covariance_trace_error": scalar(covariance_error.mean()),
        "rotation_diversity_mae": scalar(diversity_error.mean()),
        "trace_drift": scalar(diagnostics["trace_drift_max"]),
        "hermiticity_drift": scalar(diagnostics["hermiticity_drift_max"]),
        "min_eigenvalue": scalar(diagnostics["min_eigenvalue"]),
    }
    rows = []
    for index, condition in enumerate(conditions):
        rows.append(
            {
                "p": scalar(condition[0]),
                "q0": scalar(condition[1]),
                "q1": scalar(condition[2]),
                "q2": scalar(condition[3]),
                "q3": scalar(condition[4]),
                "fidelity": scalar(fidelity[index]),
                "trace_distance": scalar(distance[index]),
                "target_purity": scalar(target_purity[index]),
                "generated_purity": scalar(generated_purity[index]),
                "target_negativity": scalar(target_negativity[index]),
                "generated_negativity": scalar(generated_negativity[index]),
                "pauli_tensor_mae": scalar(tensor_error[index]),
                "local_bloch_mae": scalar(bloch_error[index]),
                "covariance_trace_error": scalar(covariance_error[index]),
                "rotation_diversity_error": scalar(diversity_error[index]),
            }
        )
    return summary, rows


@torch.no_grad()
def evaluate_generator(generator, conditions, complex_dtype):
    unique_p = torch.unique(conditions[:, 0], sorted=True)
    references = torch.zeros(unique_p.shape[0], 5, dtype=conditions.dtype, device=conditions.device)
    references[:, 0] = unique_p
    references[:, 1] = 1.0
    reference_generated = generator(references)
    return evaluate_states(generator(conditions), conditions, complex_dtype, reference_generated)


def update_ema(ema_state, generator, decay):
    with torch.no_grad():
        for name, value in generator.state_dict().items():
            ema_state[name].mul_(decay).add_(value, alpha=1.0 - decay)


def make_schedule(size, epochs, batch_size, seed):
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 9187)
    return [torch.randperm(size, generator=generator)[: min(batch_size, size)] for _ in range(epochs)]


def shuffled_targets(conditions, seed, complex_dtype):
    p_values, counts = torch.unique(conditions[:, 0], sorted=True, return_counts=True)
    if not bool((counts == counts[0]).all()):
        raise ValueError("shuffled-label control requires a Cartesian condition grid")
    rotation_count = int(counts[0])
    target_conditions = conditions.clone().reshape(len(p_values), rotation_count, 5)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 44001)
    for p_index in range(len(p_values)):
        permutation = torch.randperm(rotation_count, generator=generator).to(conditions.device)
        target_conditions[p_index, :, 1:] = target_conditions[p_index, permutation, 1:]
    return target_states(target_conditions.reshape(-1, 5), complex_dtype)


def trial_config(base_config, model, candidate):
    result = copy.deepcopy(base_config)
    for key in ("lr_d", "lr_g", "generator_steps", "discriminator_steps"):
        if key in candidate:
            result["training"][key] = candidate[key]
    if model == "qsnn" and "target_layer_mass" in candidate:
        result["discriminators"]["qsnn"]["target_layer_mass"] = candidate["target_layer_mass"]
    return result


def train_one(
    model,
    seed,
    candidate,
    initial_generator_state,
    config,
    grids,
    complex_dtype,
    dtype,
    device,
    epochs=None,
    smoke=False,
    shuffled=False,
):
    local = trial_config(config, model, candidate)
    training = local["training"]
    epochs = int(epochs if epochs is not None else training["epochs"])
    train_conditions = grids["smoke"] if smoke else grids["train"]
    correct_target = target_states(train_conditions, complex_dtype)
    real_target = shuffled_targets(train_conditions, seed, complex_dtype) if shuffled else correct_target
    schedule = make_schedule(
        len(train_conditions), epochs, int(training["batch_size"]), int(seed)
    )

    generator = build_generator(local["generator"], dtype, device)
    generator.load_state_dict(copy.deepcopy(initial_generator_state))
    seed_everything(int(seed) + (1000 if model == "qsnn" else 2000))
    discriminator = build_discriminator(model, local["discriminators"][model], dtype, device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(training["lr_g"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=float(training["lr_d"]))
    decay_start = int(training["lr_decay_start"])
    decay_factor = float(training["lr_decay_factor"])

    def lr_multiplier(epoch):
        if epoch < decay_start:
            return 1.0
        progress = (epoch - decay_start) / max(1, epochs - decay_start)
        return max(decay_factor, 1.0 - (1.0 - decay_factor) * progress)

    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_multiplier)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_multiplier)
    ema_state = copy.deepcopy(generator.state_dict())
    best_state = copy.deepcopy(generator.state_dict())
    best_source = "instant"
    best_epoch = 0
    best_validation = float("inf")
    records = []
    started = time.perf_counter()
    record_every = min(int(training["record_every"]), max(1, epochs))

    for epoch in range(epochs + 1):
        if epoch:
            indices = schedule[epoch - 1].to(device)
            batch_conditions = train_conditions[indices]
            batch_real = real_target[indices]
            for _ in range(int(training["discriminator_steps"])):
                optimizer_d.zero_grad(set_to_none=True)
                fake = generator(batch_conditions).detach()
                real_output = discriminator(batch_real, batch_conditions)
                fake_output = discriminator(fake, batch_conditions)
                loss_d = -discriminator_value(real_output, fake_output)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(), float(training["grad_clip"])
                )
                grad_d = gradient_norm(discriminator.parameters())
                optimizer_d.step()
            for _ in range(int(training["generator_steps"])):
                optimizer_g.zero_grad(set_to_none=True)
                discriminator.requires_grad_(False)
                fake_output = discriminator(generator(batch_conditions), batch_conditions)
                loss_g = -fake_output["z_expectation"].mean()
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), float(training["grad_clip"]))
                grad_g = gradient_norm(generator.parameters())
                optimizer_g.step()
                discriminator.requires_grad_(True)
                update_ema(ema_state, generator, float(training["ema_decay"]))
            scheduler_g.step()
            scheduler_d.step()
        else:
            loss_d = loss_g = torch.tensor(0.0, device=device)
            grad_d = grad_g = 0.0

        if epoch % record_every == 0 or epoch == epochs:
            current_state = copy.deepcopy(generator.state_dict())
            validation = quick_metrics(generator, grids["validation"], complex_dtype)
            joint = quick_metrics(generator, grids["joint_ood"], complex_dtype)
            candidates_at_epoch = [("instant", validation, current_state)]
            generator.load_state_dict(ema_state)
            ema_validation = quick_metrics(generator, grids["validation"], complex_dtype)
            candidates_at_epoch.append(("ema", ema_validation, copy.deepcopy(ema_state)))
            generator.load_state_dict(current_state)
            for source, metrics, state in candidates_at_epoch:
                if metrics["mean_trace_distance"] < best_validation:
                    best_validation = metrics["mean_trace_distance"]
                    best_state = copy.deepcopy(state)
                    best_source = source
                    best_epoch = epoch
            with torch.no_grad():
                output_mass = discriminator(
                    generator(train_conditions[: min(32, len(train_conditions))]),
                    train_conditions[: min(32, len(train_conditions))],
                )["output_mass"].mean()
            records.append(
                {
                    "seed": seed,
                    "model": model,
                    "candidate": candidate["name"],
                    "control": "shuffled" if shuffled else "none",
                    "epoch": epoch,
                    "loss_d": scalar(loss_d),
                    "loss_g": scalar(loss_g),
                    "grad_norm_d": grad_d,
                    "grad_norm_g": grad_g,
                    "output_mass": scalar(output_mass),
                    "lr_g": optimizer_g.param_groups[0]["lr"],
                    "validation_mean_fidelity": validation["mean_fidelity"],
                    "validation_mean_trace_distance": validation["mean_trace_distance"],
                    "ema_validation_mean_trace_distance": ema_validation["mean_trace_distance"],
                    "joint_mean_fidelity": joint["mean_fidelity"],
                    "joint_mean_trace_distance": joint["mean_trace_distance"],
                }
            )

    seconds = time.perf_counter() - started
    generator.load_state_dict(best_state)
    final = {
        "seed": seed,
        "model": model,
        "candidate": candidate["name"],
        "control": "shuffled" if shuffled else "none",
        "seconds": seconds,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_source": best_source,
        "generator_parameters": trainable_parameter_count(generator),
        "discriminator_parameters": trainable_parameter_count(discriminator),
    }
    condition_rows = []
    for grid_name in ("validation", "rotation_test", "joint_ood", "stress", "separable"):
        summary, rows = evaluate_generator(generator, grids[grid_name], complex_dtype)
        for key, value in summary.items():
            final[f"{grid_name}_{key}"] = value
        if grid_name in ("joint_ood", "stress"):
            condition_rows.extend(
                {
                    "seed": seed,
                    "model": model,
                    "candidate": candidate["name"],
                    "control": "shuffled" if shuffled else "none",
                    "grid": grid_name,
                    **row,
                }
                for row in rows
            )
    checkpoint = {
        "generator": best_state,
        "discriminator": discriminator.state_dict(),
        "candidate": candidate,
        "config": local,
        "best_epoch": best_epoch,
        "best_source": best_source,
    }
    return records, final, condition_rows, checkpoint


def capacity_one(seed, initial_state, config, grids, complex_dtype, dtype, device):
    generator = build_generator(config["generator"], dtype, device)
    generator.load_state_dict(copy.deepcopy(initial_state))
    training = config["training"]
    steps = int(training["capacity_steps"])
    conditions = grids["train"]
    target = target_states(conditions, complex_dtype)
    schedule = make_schedule(
        len(conditions), steps, int(training["capacity_batch_size"]), int(seed) + 20000
    )
    optimizer = torch.optim.Adam(generator.parameters(), lr=float(training["capacity_lr"]))
    curve = []
    started = time.perf_counter()
    for step in range(steps + 1):
        if step:
            indices = schedule[step - 1].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (generator(conditions[indices]) - target[indices]).abs().square().mean()
            loss.backward()
            optimizer.step()
        else:
            loss = (generator(conditions[:32]) - target[:32]).abs().square().mean()
        if step % 50 == 0 or step == steps:
            joint = quick_metrics(generator, grids["joint_ood"], complex_dtype)
            curve.append({"seed": seed, "step": step, "loss": scalar(loss), **joint})
    row = {"seed": seed, "seconds": time.perf_counter() - started}
    for grid_name in ("train", "validation", "joint_ood", "stress"):
        metrics = quick_metrics(generator, grids[grid_name], complex_dtype)
        row.update({f"{grid_name}_{key}": value for key, value in metrics.items()})
    return row, curve


def stage_seeds(config, stage, override):
    if override is not None:
        return override
    return config["stages"][f"{stage}_seeds"]


def run_capacity(config, grids, output_dir, complex_dtype, dtype, device, seeds):
    stage_dir = output_dir / "capacity"
    rows, curves = [], []
    for seed in seeds:
        seed_everything(int(seed))
        initial = build_generator(config["generator"], dtype, device).state_dict()
        row, curve = capacity_one(
            int(seed), initial, config, grids, complex_dtype, dtype, device
        )
        rows.append(row)
        curves.extend(curve)
        print(
            f"capacity seed={seed} joint_F={row['joint_ood_mean_fidelity']:.6f} "
            f"joint_D={row['joint_ood_mean_trace_distance']:.6f}"
        )
    thresholds = config["capacity_thresholds"]
    aggregate = {
        key: mean_std([row[key] for row in rows])
        for key in rows[0]
        if key != "seed"
    }
    passed = (
        aggregate["joint_ood_mean_fidelity"]["mean"] >= float(thresholds["joint_mean_fidelity"])
        and min(row["joint_ood_min_fidelity"] for row in rows) >= float(thresholds["joint_min_fidelity"])
        and aggregate["joint_ood_mean_trace_distance"]["mean"]
        <= float(thresholds["joint_mean_trace_distance"])
    )
    summary = {"passed": passed, "thresholds": thresholds, "metrics": aggregate}
    write_csv(stage_dir / "capacity_metrics.csv", rows)
    write_csv(stage_dir / "capacity_curve.csv", curves)
    write_json(stage_dir / "summary.json", summary)
    return summary


def candidates_for(config, model, names=None):
    candidates = config["calibration"][model]
    if names is None:
        return candidates
    return [candidate for candidate in candidates if candidate["name"] in names]


def run_trials(stage, config, grids, output_dir, complex_dtype, dtype, device, seeds, choices, epochs=None, smoke=False, shuffled=False):
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    all_records, finals, condition_rows = [], [], []
    for seed in seeds:
        seed_everything(int(seed))
        initial_state = copy.deepcopy(build_generator(config["generator"], dtype, device).state_dict())
        for model in ("qsnn", "vqc"):
            for candidate in choices[model]:
                records, final, rows, checkpoint = train_one(
                    model,
                    int(seed),
                    candidate,
                    initial_state,
                    config,
                    grids,
                    complex_dtype,
                    dtype,
                    device,
                    epochs=epochs,
                    smoke=smoke,
                    shuffled=shuffled,
                )
                all_records.extend(records)
                finals.append(final)
                condition_rows.extend(rows)
                torch.save(
                    checkpoint,
                    stage_dir / f"{model}_{candidate['name']}_seed{seed}{'_shuffled' if shuffled else ''}.pt",
                )
                # Persist after every trial so a long calibration/formal run
                # retains completed candidates if the process is interrupted.
                write_csv(
                    stage_dir / ("training_shuffled.csv" if shuffled else "training.csv"),
                    all_records,
                )
                write_csv(
                    stage_dir / ("final_shuffled.csv" if shuffled else "final.csv"), finals
                )
                write_csv(
                    stage_dir / ("conditions_shuffled.csv" if shuffled else "conditions.csv"),
                    condition_rows,
                )
                print(
                    f"{stage} seed={seed} model={model} candidate={candidate['name']} "
                    f"joint_D={final['joint_ood_mean_trace_distance']:.6f} "
                    f"joint_F={final['joint_ood_mean_fidelity']:.6f}"
                )
    write_csv(stage_dir / ("training_shuffled.csv" if shuffled else "training.csv"), all_records)
    write_csv(stage_dir / ("final_shuffled.csv" if shuffled else "final.csv"), finals)
    write_csv(stage_dir / ("conditions_shuffled.csv" if shuffled else "conditions.csv"), condition_rows)
    return all_records, finals, condition_rows


def rank_candidates(finals, keep):
    selection = {}
    for model in ("qsnn", "vqc"):
        names = sorted({row["candidate"] for row in finals if row["model"] == model})
        scored = []
        for name in names:
            values = [
                row["joint_ood_mean_trace_distance"]
                for row in finals
                if row["model"] == model and row["candidate"] == name
            ]
            scored.append({"name": name, "mean_joint_trace_distance": sum(values) / len(values)})
        selection[model] = sorted(scored, key=lambda row: row["mean_joint_trace_distance"])[:keep]
    return selection


def run_smoke(config, grids, output_dir, complex_dtype, dtype, device, seeds):
    choices = {model: [config["calibration"][model][0]] for model in ("qsnn", "vqc")}
    records, finals, _ = run_trials(
        "smoke",
        config,
        grids,
        output_dir,
        complex_dtype,
        dtype,
        device,
        seeds,
        choices,
        epochs=int(config["stages"]["smoke_epochs"]),
        smoke=True,
    )
    finite = all(math.isfinite(row["joint_ood_mean_trace_distance"]) for row in finals)
    gradients = all(
        math.isfinite(row["grad_norm_g"]) and math.isfinite(row["grad_norm_d"])
        for row in records
    )
    summary = {"passed": finite and gradients, "finite_metrics": finite, "finite_gradients": gradients}
    write_json(output_dir / "smoke" / "summary.json", summary)
    return summary


def run_calibration(config, grids, output_dir, complex_dtype, dtype, device, seeds, epochs=None):
    choices = {model: candidates_for(config, model) for model in ("qsnn", "vqc")}
    _, finals, _ = run_trials(
        "calibration", config, grids, output_dir, complex_dtype, dtype, device, seeds, choices, epochs
    )
    selection = rank_candidates(finals, keep=2)
    write_json(output_dir / "calibration" / "selection.json", selection)
    return selection


def run_validation(config, grids, output_dir, complex_dtype, dtype, device, seeds, epochs=None):
    prior = read_json(output_dir / "calibration" / "selection.json")
    choices = {
        model: candidates_for(config, model, [row["name"] for row in prior[model]])
        for model in ("qsnn", "vqc")
    }
    _, finals, _ = run_trials(
        "validation", config, grids, output_dir, complex_dtype, dtype, device, seeds, choices, epochs
    )
    selection = rank_candidates(finals, keep=1)
    write_json(output_dir / "validation" / "selection.json", selection)
    return selection


def exact_sign_flip_paired_p(qsnn, vqc):
    differences = torch.tensor(qsnn, dtype=torch.float64) - torch.tensor(vqc, dtype=torch.float64)
    observed = differences.mean().abs()
    count = differences.numel()
    if count <= 20:
        indices = torch.arange(1 << count, dtype=torch.long)
        bit_positions = torch.arange(count, dtype=torch.long)
        signs = 1.0 - 2.0 * ((indices[:, None] >> bit_positions) & 1).to(torch.float64)
        null_values = (signs * differences).mean(dim=1).abs()
        return scalar((null_values >= observed - 1e-15).to(torch.float64).mean())
    generator = torch.Generator().manual_seed(99173)
    signs = torch.randint(0, 2, (200000, count), generator=generator, dtype=torch.int64)
    signs = 1.0 - 2.0 * signs.to(torch.float64)
    return scalar(((signs * differences).mean(dim=1).abs() >= observed).to(torch.float64).mean())


def formal_summary(finals, config):
    qsnn_rows = sorted((row for row in finals if row["model"] == "qsnn"), key=lambda row: row["seed"])
    vqc_rows = sorted((row for row in finals if row["model"] == "vqc"), key=lambda row: row["seed"])
    if [row["seed"] for row in qsnn_rows] != [row["seed"] for row in vqc_rows]:
        raise ValueError("formal comparison requires paired seeds")
    key = "joint_ood_mean_trace_distance"
    qsnn = [row[key] for row in qsnn_rows]
    vqc = [row[key] for row in vqc_rows]
    p_value = exact_sign_flip_paired_p(qsnn, vqc)
    improvement = (sum(vqc) - sum(qsnn)) / max(sum(vqc), 1e-12)
    wins = sum(left < right for left, right in zip(qsnn, vqc)) / len(qsnn)
    claims = config["claims"]
    worst_consistent = mean_std([r["joint_ood_max_trace_distance"] for r in qsnn_rows])["mean"] < mean_std(
        [r["joint_ood_max_trace_distance"] for r in vqc_rows]
    )["mean"]
    covariance_consistent = mean_std([r["joint_ood_covariance_trace_error"] for r in qsnn_rows])["mean"] < mean_std(
        [r["joint_ood_covariance_trace_error"] for r in vqc_rows]
    )["mean"]
    superiority = (
        p_value < float(claims["paired_p_value"])
        and improvement >= float(claims["relative_improvement"])
        and wins > float(claims["majority_win_fraction"])
        and worst_consistent
        and covariance_consistent
    )
    metrics = [
        "joint_ood_mean_fidelity",
        "joint_ood_min_fidelity",
        "joint_ood_mean_trace_distance",
        "joint_ood_max_trace_distance",
        "joint_ood_purity_mae",
        "joint_ood_negativity_mae",
        "joint_ood_pauli_tensor_mae",
        "joint_ood_local_bloch_mae",
        "joint_ood_covariance_trace_error",
        "joint_ood_rotation_diversity_mae",
        "stress_mean_trace_distance",
        "seconds",
    ]
    return {
        "primary_metric": key,
        "paired_sign_flip_p_value": p_value,
        "relative_improvement": improvement,
        "qsnn_win_fraction": wins,
        "worst_case_consistent": worst_consistent,
        "covariance_consistent": covariance_consistent,
        "superiority_criteria_met": superiority,
        "models": {
            model: {
                metric: mean_std([row[metric] for row in finals if row["model"] == model])
                for metric in metrics
            }
            for model in ("qsnn", "vqc")
        },
    }


def baseline_controls(grids, complex_dtype):
    conditions = grids["joint_ood"]
    target = target_states(conditions, complex_dtype)
    fixed = werner_state(conditions[:, 0], dtype=complex_dtype, device=conditions.device)
    collapse = torch.eye(4, dtype=complex_dtype, device=conditions.device).expand(len(conditions), -1, -1) / 4.0
    unique_p = torch.unique(conditions[:, 0], sorted=True)
    fixed_reference = werner_state(unique_p, dtype=complex_dtype, device=conditions.device)
    collapse_reference = torch.eye(4, dtype=complex_dtype, device=conditions.device).expand(
        len(unique_p), -1, -1
    ) / 4.0
    fixed_summary, _ = evaluate_states(fixed, conditions, complex_dtype, fixed_reference)
    collapse_summary, _ = evaluate_states(
        collapse, conditions, complex_dtype, collapse_reference
    )
    return {"fixed_unrotated_werner": fixed_summary, "maximally_mixed_collapse": collapse_summary}


def plot_training(path, records):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fields = (
        ("joint_mean_trace_distance", "Joint-OOD trace distance"),
        ("validation_mean_trace_distance", "Validation trace distance"),
        ("grad_norm_g", "Generator gradient norm"),
        ("output_mass", "Discriminator output mass"),
    )
    for axis, (field, title) in zip(axes.flat, fields):
        for model, color in (("qsnn", "tab:blue"), ("vqc", "tab:orange")):
            selected = [row for row in records if row["model"] == model]
            epochs = sorted({row["epoch"] for row in selected})
            means = [
                sum(row[field] for row in selected if row["epoch"] == epoch)
                / len([row for row in selected if row["epoch"] == epoch])
                for epoch in epochs
            ]
            axis.plot(epochs, means, label=model.upper(), color=color)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_formal(path, finals):
    qsnn = sorted((row for row in finals if row["model"] == "qsnn"), key=lambda row: row["seed"])
    vqc = sorted((row for row in finals if row["model"] == "vqc"), key=lambda row: row["seed"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    key = "joint_ood_mean_trace_distance"
    axes[0].scatter([row[key] for row in vqc], [row[key] for row in qsnn], color="tab:purple")
    limit = max([row[key] for row in qsnn + vqc]) * 1.05
    axes[0].plot([0, limit], [0, limit], "k--", linewidth=1)
    axes[0].set(xlabel="VQC trace distance", ylabel="QSNN trace distance", title="Paired formal seeds")
    axes[0].grid(alpha=0.25)
    labels = ["Trace distance", "Covariance error", "Pauli tensor MAE"]
    keys = [key, "joint_ood_covariance_trace_error", "joint_ood_pauli_tensor_mae"]
    x = torch.arange(len(keys), dtype=torch.float64)
    width = 0.35
    axes[1].bar(x - width / 2, [sum(r[k] for r in qsnn) / len(qsnn) for k in keys], width, label="QSNN")
    axes[1].bar(x + width / 2, [sum(r[k] for r in vqc) / len(vqc) for k in keys], width, label="VQC")
    axes[1].set_xticks(x.tolist(), labels, rotation=18, ha="right")
    axes[1].set_title("Joint-OOD errors (lower is better)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _curve_statistics(rows, model, field):
    selected = [
        row
        for row in rows
        if row["model"] == model
        and row["grid"] == "joint_ood"
        and row.get("control", "none") == "none"
    ]
    p_values = sorted({row["p"] for row in selected})
    means, standard_deviations = [], []
    for p_value in p_values:
        per_seed = []
        for seed in sorted({row["seed"] for row in selected}):
            values = [
                row[field]
                for row in selected
                if row["seed"] == seed and row["p"] == p_value
            ]
            if values:
                per_seed.append(sum(values) / len(values))
        tensor = torch.tensor(per_seed, dtype=torch.float64)
        means.append(float(tensor.mean()))
        standard_deviations.append(
            float(tensor.std(unbiased=True)) if tensor.numel() > 1 else 0.0
        )
    return p_values, means, standard_deviations


def plot_condition_curves(path, condition_rows):
    """Four OOD condition panels, matching the earlier Werner report layout."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = (
        ("fidelity", "Joint-OOD fidelity", 1.0, None),
        ("trace_distance", "Joint-OOD trace distance", 0.0, None),
        ("generated_purity", "Purity", None, "target_purity"),
        ("generated_negativity", "Negativity", None, "target_negativity"),
    )
    for axis, (field, title, reference, target_field) in zip(axes.flat, panels):
        p_values = None
        for model, color in (("qsnn", "tab:blue"), ("vqc", "tab:orange")):
            p_values, means, deviations = _curve_statistics(condition_rows, model, field)
            lower = [mean - std for mean, std in zip(means, deviations)]
            upper = [mean + std for mean, std in zip(means, deviations)]
            axis.plot(p_values, means, marker="o", label=model.upper(), color=color)
            axis.fill_between(p_values, lower, upper, color=color, alpha=0.16)
        if reference is not None:
            axis.axhline(reference, color="black", linestyle="--", linewidth=1, label="Target")
        if target_field is not None and p_values is not None:
            target_rows = [
                row
                for row in condition_rows
                if row["model"] == "qsnn"
                and row["grid"] == "joint_ood"
                and row.get("control", "none") == "none"
            ]
            targets = []
            for p_value in p_values:
                values = [row[target_field] for row in target_rows if row["p"] == p_value]
                targets.append(sum(values) / len(values))
            axis.plot(p_values, targets, "k--", linewidth=1.4, label="Target")
        axis.set_title(title)
        axis.set_xlabel("Werner parameter p")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(config, output_dir):
    report_path = ROOT / "docs" / "reports" / "RANDOM_LOCAL_ROTATED_WERNER_QGAN_REPORT.md"
    relative_output = "../../outputs/qgan/random_local_rotated_werner"
    sections = [
        "# 随机局域旋转 Werner 态族 QSNN-QGAN 实验报告\n",
        "## 任务与假设\n",
        "目标条件态为\n\n"
        "$$\\rho(p,U)=(I\\otimes U)\\rho_W(p)(I\\otimes U^\\dagger),$$\n\n"
        "其中 $U\\in SU(2)$ 由单位四元数 $q=(q_0,q_1,q_2,q_3)$ 表示，生成器和判别器接收五维条件 "
        "$c=(p,q_0,q_1,q_2,q_3)$。训练与测试旋转池完全分离；主指标是“未见过的旋转 + 未见过的 p”上的平均迹距离。\n",
        "## 公平性协议\n",
        f"- QSNN 与 VQC 使用相同的 {config['generator']['layers']} 层纯化生成器、初始生成器参数、目标批次和训练轮数。\n"
        "- 容量预检后，公共生成器采用局域 SU(2) 等变输出层：核心线路只学习 p 依赖，随后施加条件给定的 I⊗U(q)。\n"
        "- 五维条件下，QSNN 判别器为 276 个可训练参数，VQC 判别器为 288 个，相差 4.35%。\n"
        "- 超参数只在校准种子上选择，验证种子用于冻结方案，正式种子 20–39 只运行一次配对比较。\n"
        "- 只有同时满足配对 $p<0.05$、主指标至少改善 10%、多数种子获胜、最坏情况与协变误差方向一致，才声称 QSNN 优越。\n",
        "## 数据划分\n",
        "训练 $p=[0.40,0.55,0.70,0.85,1.00]$、插值 $p=[0.475,0.625,0.775,0.925]$；"
        "训练/验证/测试分别使用 64/16/64 个 Haar 随机旋转，另设 12 个极端旋转和 $p\\le1/3$ 可分控制。\n",
    ]
    capacity_path = output_dir / "capacity" / "summary.json"
    if capacity_path.exists():
        capacity = read_json(capacity_path)
        metric = capacity["metrics"]
        sections.extend(
            [
                "## A. 生成器容量预检\n",
                f"预检结论：**{'通过' if capacity['passed'] else '未通过'}**。联合 OOD 平均保真度 "
                f"{metric['joint_ood_mean_fidelity']['mean']:.6f}，最小保真度的种子汇总见原始表，平均迹距离 "
                f"{metric['joint_ood_mean_trace_distance']['mean']:.6f}。\n\n"
                f"原始结果：[capacity_metrics.csv]({relative_output}/capacity/capacity_metrics.csv)。\n",
            ]
        )
    formal_path = output_dir / "formal" / "summary.json"
    if formal_path.exists():
        formal = read_json(formal_path)
        q = formal["models"]["qsnn"]
        v = formal["models"]["vqc"]
        sections.extend(
            [
                "## 正式盲测结果\n",
                f"QSNN 联合 OOD 平均迹距离：{q['joint_ood_mean_trace_distance']['mean']:.6f} ± "
                f"{q['joint_ood_mean_trace_distance']['std']:.6f}；VQC：{v['joint_ood_mean_trace_distance']['mean']:.6f} ± "
                f"{v['joint_ood_mean_trace_distance']['std']:.6f}。配对符号翻转检验 $p={formal['paired_sign_flip_p_value']:.6g}$，"
                f"相对改善 {100 * formal['relative_improvement']:.2f}%，QSNN 胜率 {100 * formal['qsnn_win_fraction']:.1f}%。\n\n"
                f"预注册的 QSNN 优越性判据：**{'满足' if formal['superiority_criteria_met'] else '不满足'}**。\n\n"
                "## 八个核心曲线面板\n",
                "### 训练过程（四个面板）\n",
                "1. 左上：联合 OOD 迹距离，越低越好；2. 右上：独立验证集迹距离，越低越好；"
                "3. 左下：生成器梯度范数，用于检查梯度消失或爆炸；4. 右下：判别器输出层质量，用于检查输出泄漏。\n\n"
                f"![随机旋转Werner训练曲线]({relative_output}/formal/training_curves.png)\n",
                "### 未见 p 与未见旋转上的条件曲线（四个面板）\n",
                "5. 左上：保真度；6. 右上：迹距离；7. 左下：生成态纯度与目标纯度；"
                "8. 右下：生成态负性与目标负性。实线为 20 个正式种子的均值，阴影为种子间标准差。\n\n"
                f"![随机旋转Werner条件曲线]({relative_output}/formal/condition_curves.png)\n",
                "## 补充配对统计图\n",
                f"![正式配对结果]({relative_output}/formal/formal_comparison.png)\n",
                "## 结论边界\n",
                "本实验只检验随机局域旋转 Werner 条件态族上的生成表现。若容量预检失败、输出层质量泄漏异常，或正式判据未全部满足，"
                "则不能把个别种子或单一指标的改善解释为 QSNN-GAN 的总体优越性。\n",
            ]
        )
    else:
        sections.extend(
            [
                "## 当前状态\n",
                "实验采用分阶段、可断点续跑流程。正式盲测尚未完成时，本报告只记录协议和已经产生的阶段结果，不提前给出优越性结论。\n",
            ]
        )
    report_path.write_text("\n".join(sections), encoding="utf-8")


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.capacity_steps is not None:
        config["training"]["capacity_steps"] = int(args.capacity_steps)
    if args.device is not None:
        config["runtime"]["device"] = args.device
    output_dir = args.output_dir or ROOT / config["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = config["runtime"].get("device", "cpu")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    dtype = torch.float64 if config["runtime"].get("dtype") == "float64" else torch.float32
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    grids = build_grids(config, dtype, device)
    write_json(
        output_dir / "run_metadata.json",
        {
            "config": config,
            "runtime": {"device": str(device), "torch": torch.__version__, "cuda": torch.cuda.is_available()},
        },
    )

    requested = STAGE_ORDER if args.stage == "all" else (args.stage,)
    for stage in requested:
        if stage == "report":
            formal_dir = output_dir / "formal"
            records = read_csv_rows(formal_dir / "training.csv")
            finals = read_csv_rows(formal_dir / "final.csv")
            condition_rows = read_csv_rows(formal_dir / "conditions.csv")
            plot_training(formal_dir / "training_curves.png", records)
            plot_condition_curves(formal_dir / "condition_curves.png", condition_rows)
            plot_formal(formal_dir / "formal_comparison.png", finals)
            write_report(config, output_dir)
            continue
        seeds = stage_seeds(config, stage, args.seeds if args.stage != "all" else None)
        if stage == "capacity":
            summary = run_capacity(
                config, grids, output_dir, complex_dtype, dtype, device, seeds
            )
            write_report(config, output_dir)
            if not summary["passed"] and args.stage == "all" and not args.allow_capacity_failure:
                raise SystemExit(
                    "Capacity precheck failed; improve the common generator or rerun with "
                    "--allow-capacity-failure for diagnostics."
                )
        elif stage == "smoke":
            summary = run_smoke(
                config, grids, output_dir, complex_dtype, dtype, device, seeds
            )
            if not summary["passed"]:
                raise SystemExit("Smoke test produced a non-finite metric or gradient")
        elif stage == "calibration":
            run_calibration(
                config, grids, output_dir, complex_dtype, dtype, device, seeds, args.epochs
            )
        elif stage == "validation":
            run_validation(
                config, grids, output_dir, complex_dtype, dtype, device, seeds, args.epochs
            )
        elif stage == "formal":
            selection = read_json(output_dir / "validation" / "selection.json")
            choices = {
                model: candidates_for(config, model, [selection[model][0]["name"]])
                for model in ("qsnn", "vqc")
            }
            records, finals, _ = run_trials(
                "formal",
                config,
                grids,
                output_dir,
                complex_dtype,
                dtype,
                device,
                seeds,
                choices,
                args.epochs,
            )
            summary = formal_summary(finals, config)
            summary["baselines"] = baseline_controls(grids, complex_dtype)
            if not args.skip_controls:
                _, shuffled_finals, _ = run_trials(
                    "formal",
                    config,
                    grids,
                    output_dir,
                    complex_dtype,
                    dtype,
                    device,
                    config["stages"]["control_seeds"],
                    choices,
                    args.epochs,
                    shuffled=True,
                )
                summary["shuffled_label_control"] = {
                    model: mean_std(
                        [
                            row["joint_ood_mean_trace_distance"]
                            for row in shuffled_finals
                            if row["model"] == model
                        ]
                    )
                    for model in ("qsnn", "vqc")
                }
            write_json(output_dir / "formal" / "summary.json", summary)
            plot_training(output_dir / "formal" / "training_curves.png", records)
            plot_condition_curves(
                output_dir / "formal" / "condition_curves.png",
                read_csv_rows(output_dir / "formal" / "conditions.csv"),
            )
            plot_formal(output_dir / "formal" / "formal_comparison.png", finals)
            write_report(config, output_dir)
    write_report(config, output_dir)


if __name__ == "__main__":
    main()
