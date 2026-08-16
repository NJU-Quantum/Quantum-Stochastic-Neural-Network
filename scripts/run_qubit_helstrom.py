from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.quantum_state_discrimination.bounds import (  # noqa: E402
    best_fixed_pauli_success,
    helstrom_measurement,
)
from tasks.quantum_state_discrimination.experiment import (  # noqa: E402
    TrainConfig,
    simulate_shot_success,
    train_discriminator,
)
from tasks.quantum_state_discrimination.models import (  # noqa: E402
    QubitHelstromQSNN,
    UnitaryQubitDiscriminator,
)
from tasks.quantum_state_discrimination.states import (  # noqa: E402
    make_nonorthogonal_qubit_ensemble,
)


DEFAULT_CONFIG = ROOT / "configs" / "qubit_helstrom_smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a one-qubit QSNN and compare it with the Helstrom bound."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            return {
                "real": value.real.tolist(),
                "imag": value.imag.tolist(),
            }
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def case_name(angle: float, noise: float, seed: int) -> str:
    angle_text = str(angle).replace(".", "p")
    noise_text = str(noise).replace(".", "p")
    return f"angle_{angle_text}_noise_{noise_text}_seed_{seed}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]], configured_seeds: list[int]) -> dict[str, Any]:
    gaps = [float(row["qsnn_helstrom_gap"]) for row in rows]
    leakages = [float(row["qsnn_weighted_leakage"]) for row in rows]
    min_povm_eigs = [float(row["qsnn_min_povm_eigenvalue"]) for row in rows]
    mean_gap = sum(gaps) / len(gaps)
    max_gap = max(gaps)
    max_leakage = max(leakages)
    shot_fields = sorted(
        (
            key
            for key in rows[0]
            if key.startswith("qsnn_success_") and key.endswith("_shots")
        ),
        key=lambda value: int(value.split("_")[2]),
    )
    largest_shots = int(shot_fields[-1].split("_")[2]) if shot_fields else None
    shot_mean_abs_error = None
    shot_max_condition_abs_error = None
    if shot_fields:
        shot_field = shot_fields[-1]
        absolute_errors = [
            abs(float(row[shot_field]) - float(row["qsnn_success"])) for row in rows
        ]
        shot_mean_abs_error = sum(absolute_errors) / len(absolute_errors)
        grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
        for row in rows:
            key = (float(row["angle_degrees"]), float(row["noise_strength"]))
            grouped.setdefault(key, []).append(row)
        condition_errors = []
        for group in grouped.values():
            exact = sum(float(row["qsnn_success"]) for row in group) / len(group)
            sampled = sum(float(row[shot_field]) for row in group) / len(group)
            condition_errors.append(abs(sampled - exact))
        shot_max_condition_abs_error = max(condition_errors)
    enough_seeds = len(set(configured_seeds)) >= 5
    shot_threshold_passed = bool(
        shot_max_condition_abs_error is not None
        and shot_max_condition_abs_error <= 0.02
    )
    numerical_pass = (
        mean_gap <= 0.01
        and max_gap <= 0.02
        and max_leakage <= 0.005
        and min(min_povm_eigs) >= -1e-7
        and shot_threshold_passed
    )
    return {
        "runs": len(rows),
        "unique_seeds": len(set(configured_seeds)),
        "mean_qsnn_helstrom_gap": mean_gap,
        "max_qsnn_helstrom_gap": max_gap,
        "mean_qsnn_success": sum(float(row["qsnn_success"]) for row in rows) / len(rows),
        "max_qsnn_weighted_leakage": max_leakage,
        "min_qsnn_povm_eigenvalue": min(min_povm_eigs),
        "largest_shots": largest_shots,
        "shot_mean_absolute_error": shot_mean_abs_error,
        "shot_max_condition_absolute_error": shot_max_condition_abs_error,
        "shot_threshold_passed": shot_threshold_passed,
        "numerical_thresholds_passed": numerical_pass,
        "hardware_promotion_recommended": bool(numerical_pass and enough_seeds),
        "promotion_note": (
            "All numerical thresholds and the five-seed requirement passed."
            if numerical_pass and enough_seeds
            else "Hardware promotion requires all thresholds and at least five seeds."
        ),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    experiment_config = config["experiment"]
    train_config = TrainConfig(**config["training"])
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (ROOT / experiment_config["output_dir"]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    history_dir = output_dir / "histories"
    history_dir.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    device = torch.device(args.device)
    priors = tuple(float(value) for value in experiment_config["priors"])
    shots_values = [int(value) for value in experiment_config.get("shots", [])]
    started = time.perf_counter()

    for angle in experiment_config["angles_degrees"]:
        for noise in experiment_config["noise_strengths"]:
            ensemble = make_nonorthogonal_qubit_ensemble(
                separation_degrees=float(angle),
                phase_degrees=float(experiment_config.get("phase_degrees", 0.0)),
                noise_model=experiment_config.get("noise_model", "none"),
                noise_strength=float(noise),
                priors=priors,
                device=device,
            )
            helstrom = helstrom_measurement(ensemble)
            pauli_success, pauli_measurement = best_fixed_pauli_success(ensemble)

            for seed in experiment_config["seeds"]:
                seed = int(seed)
                torch.manual_seed(seed)
                qsnn = QubitHelstromQSNN(device=device, **config["qsnn"])
                qsnn_metrics = train_discriminator(qsnn, ensemble, train_config)

                torch.manual_seed(seed)
                unitary = UnitaryQubitDiscriminator(device=device)
                unitary_metrics = train_discriminator(
                    unitary,
                    ensemble,
                    TrainConfig(
                        epochs=train_config.epochs,
                        learning_rate=train_config.learning_rate,
                        leakage_penalty=0.0,
                        grad_clip=train_config.grad_clip,
                        log_every=train_config.log_every,
                    ),
                )

                name = case_name(float(angle), float(noise), seed)
                checkpoint = {
                    "format_version": 1,
                    "model": "QubitHelstromQSNN",
                    "state_dict": qsnn.state_dict(),
                    "qsnn_config": config["qsnn"],
                    "training_config": asdict(train_config),
                    "ensemble": dict(ensemble.metadata),
                    "priors": ensemble.priors,
                    "metrics": {
                        key: value
                        for key, value in qsnn_metrics.items()
                        if key not in ("effects", "history")
                    },
                }
                torch.save(checkpoint, checkpoint_dir / f"{name}.pt")
                (history_dir / f"{name}.json").write_text(
                    json.dumps(qsnn_metrics["history"], indent=2), encoding="utf-8"
                )

                diagnostics = qsnn_metrics["povm_diagnostics"]
                row: dict[str, Any] = {
                    "angle_degrees": float(angle),
                    "phase_degrees": float(experiment_config.get("phase_degrees", 0.0)),
                    "noise_model": experiment_config.get("noise_model", "none"),
                    "noise_strength": float(noise),
                    "prior0": ensemble.priors[0],
                    "prior1": ensemble.priors[1],
                    "seed": seed,
                    "helstrom_success": helstrom.success,
                    "qsnn_success": qsnn_metrics["success"],
                    "qsnn_helstrom_gap": qsnn_metrics["helstrom_gap"],
                    "qsnn_weighted_leakage": qsnn_metrics["weighted_leakage"],
                    "qsnn_train_seconds": qsnn_metrics["training_seconds"],
                    "qsnn_parameters": qsnn_metrics["trainable_parameters"],
                    "qsnn_min_povm_eigenvalue": min(
                        diagnostics["effect0_min_eigenvalue"],
                        diagnostics["effect1_min_eigenvalue"],
                        diagnostics["leakage_effect_min_eigenvalue"],
                    ),
                    "unitary_success": unitary_metrics["success"],
                    "unitary_helstrom_gap": unitary_metrics["helstrom_gap"],
                    "unitary_train_seconds": unitary_metrics["training_seconds"],
                    "unitary_parameters": unitary_metrics["trainable_parameters"],
                    "fixed_pauli_success": pauli_success,
                    "fixed_pauli_measurement": pauli_measurement,
                }
                for shots in shots_values:
                    row[f"qsnn_success_{shots}_shots"] = simulate_shot_success(
                        qsnn_metrics["conditional_probabilities"],
                        ensemble.priors,
                        shots,
                        seed=100000 + seed + shots,
                    )
                rows.append(row)
                details.append(
                    {
                        "case": name,
                        "row": row,
                        "ensemble": {
                            **dict(ensemble.metadata),
                            "priors": ensemble.priors,
                        },
                        "qsnn": qsnn_metrics,
                        "unitary": unitary_metrics,
                    }
                )
                print(
                    f"{name}: H={helstrom.success:.6f} "
                    f"QSNN={qsnn_metrics['success']:.6f} "
                    f"gap={qsnn_metrics['helstrom_gap']:.6f} "
                    f"leak={qsnn_metrics['weighted_leakage']:.6f}"
                )

    summary = aggregate(rows, [int(seed) for seed in experiment_config["seeds"]])
    summary.update(
        {
            "wall_seconds": time.perf_counter() - started,
            "config": str(config_path),
            "output_dir": str(output_dir),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": str(device),
            },
        }
    )
    write_csv(output_dir / "results.csv", rows)
    (output_dir / "results.json").write_text(
        json.dumps(json_safe(details), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
