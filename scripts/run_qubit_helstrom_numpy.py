from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.quantum_state_discrimination.numpy_reference import (  # noqa: E402
    NumpyTrainConfig,
    best_fixed_pauli_success,
    effective_povm,
    helstrom,
    initial_qsnn_parameters,
    json_safe,
    make_ensemble,
    povm_diagnostics,
    qsnn_forward,
    simulate_shots,
    train,
    unitary_forward,
)

import numpy as np  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "qubit_helstrom_smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NumPy reference for the qubit Helstrom QSNN.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    qsnn_config = config["qsnn"]
    if bool(qsnn_config.get("coherent_during_dissipation", False)):
        raise ValueError(
            "The NumPy reference supports only coherent_during_dissipation=false; "
            "use the PyTorch runner for interleaved coherent/dissipative evolution."
        )
    training_values = dict(config["training"])
    training_values["finite_difference_step"] = float(
        training_values.get("finite_difference_step", 1e-5)
    )
    training = NumpyTrainConfig(**training_values)
    if args.output_dir is None:
        output_dir = (ROOT / experiment["output_dir"] / "numpy_reference").resolve()
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    started = time.perf_counter()

    coherent_time = float(qsnn_config["coherent_time"])
    dissipative_time = float(qsnn_config["dissipative_time"])
    target_mass = float(qsnn_config["target_initial_output_mass"])
    qsnn_callable = lambda parameters, states: qsnn_forward(
        parameters, states, coherent_time, dissipative_time
    )
    unitary_callable = lambda parameters, states: unitary_forward(
        parameters, states, coherent_time
    )

    for angle in experiment["angles_degrees"]:
        for noise in experiment["noise_strengths"]:
            ensemble = make_ensemble(
                float(angle),
                float(experiment.get("phase_degrees", 0.0)),
                experiment.get("noise_model", "none"),
                float(noise),
                experiment["priors"],
            )
            bound = helstrom(ensemble)
            pauli_success, pauli_name = best_fixed_pauli_success(ensemble)
            for seed_value in experiment["seeds"]:
                seed = int(seed_value)
                rng = np.random.default_rng(seed)
                initial = initial_qsnn_parameters(rng, dissipative_time, target_mass)
                qsnn = train(
                    initial,
                    ensemble,
                    qsnn_callable,
                    training,
                    regularize_qsnn=True,
                )
                unitary = train(
                    0.05 * np.random.default_rng(seed).normal(size=3),
                    ensemble,
                    unitary_callable,
                    NumpyTrainConfig(
                        epochs=training.epochs,
                        learning_rate=training.learning_rate,
                        leakage_penalty=0.0,
                        grad_clip=training.grad_clip,
                        log_every=training.log_every,
                        finite_difference_step=training.finite_difference_step,
                    ),
                    regularize_qsnn=False,
                )
                effects = effective_povm(qsnn["parameters"], qsnn_callable)
                diagnostics = povm_diagnostics(effects)
                row: dict[str, Any] = {
                    "angle_degrees": float(angle),
                    "phase_degrees": float(experiment.get("phase_degrees", 0.0)),
                    "noise_model": experiment.get("noise_model", "none"),
                    "noise_strength": float(noise),
                    "seed": seed,
                    "helstrom_success": float(bound["success"]),
                    "qsnn_success": qsnn["success"],
                    "qsnn_helstrom_gap": qsnn["helstrom_gap"],
                    "qsnn_weighted_leakage": qsnn["weighted_leakage"],
                    "qsnn_train_seconds": qsnn["training_seconds"],
                    "qsnn_min_povm_eigenvalue": diagnostics["minimum_effect_eigenvalue"],
                    "unitary_success": unitary["success"],
                    "unitary_helstrom_gap": unitary["helstrom_gap"],
                    "fixed_pauli_success": pauli_success,
                    "fixed_pauli_measurement": pauli_name,
                }
                for shots in experiment.get("shots", []):
                    shots = int(shots)
                    row[f"qsnn_success_{shots}_shots"] = simulate_shots(
                        qsnn["conditional_probabilities"],
                        ensemble.priors,
                        shots,
                        100000 + seed + shots,
                    )
                rows.append(row)
                details.append(
                    {
                        "ensemble": {**ensemble.metadata, "priors": ensemble.priors},
                        "row": row,
                        "qsnn": qsnn,
                        "unitary": unitary,
                        "povm_effects": effects,
                        "povm_diagnostics": diagnostics,
                    }
                )
                print(
                    f"angle={angle:g} noise={noise:g} seed={seed} "
                    f"H={bound['success']:.6f} QSNN={qsnn['success']:.6f} "
                    f"gap={qsnn['helstrom_gap']:.6f} leak={qsnn['weighted_leakage']:.6f}"
                )

    gaps = np.asarray([row["qsnn_helstrom_gap"] for row in rows])
    leakages = np.asarray([row["qsnn_weighted_leakage"] for row in rows])
    minimum_eigenvalue = min(row["qsnn_min_povm_eigenvalue"] for row in rows)
    configured_shots = sorted(int(value) for value in experiment.get("shots", []))
    largest_shots = configured_shots[-1] if configured_shots else None
    shot_mean_absolute_error = None
    shot_max_condition_absolute_error = None
    if largest_shots is not None:
        shot_field = f"qsnn_success_{largest_shots}_shots"
        shot_mean_absolute_error = float(
            np.mean(
                [
                    abs(float(row[shot_field]) - float(row["qsnn_success"]))
                    for row in rows
                ]
            )
        )
        grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
        for row in rows:
            key = (float(row["angle_degrees"]), float(row["noise_strength"]))
            grouped.setdefault(key, []).append(row)
        condition_errors = []
        for group in grouped.values():
            exact = np.mean([float(row["qsnn_success"]) for row in group])
            sampled = np.mean([float(row[shot_field]) for row in group])
            condition_errors.append(abs(float(sampled - exact)))
        shot_max_condition_absolute_error = float(max(condition_errors))
    shot_threshold_passed = bool(
        shot_max_condition_absolute_error is not None
        and shot_max_condition_absolute_error <= 0.02
    )
    numerical_pass = bool(
        gaps.mean() <= 0.01
        and gaps.max() <= 0.02
        and leakages.max() <= 0.005
        and minimum_eigenvalue >= -1e-7
        and shot_threshold_passed
    )
    enough_seeds = len(set(int(seed) for seed in experiment["seeds"])) >= 5
    summary = {
        "backend": "numpy finite-difference reference",
        "runs": len(rows),
        "unique_seeds": len(set(int(seed) for seed in experiment["seeds"])),
        "mean_qsnn_helstrom_gap": float(gaps.mean()),
        "max_qsnn_helstrom_gap": float(gaps.max()),
        "mean_qsnn_success": float(np.mean([row["qsnn_success"] for row in rows])),
        "max_qsnn_weighted_leakage": float(leakages.max()),
        "min_qsnn_povm_eigenvalue": float(minimum_eigenvalue),
        "largest_shots": largest_shots,
        "shot_mean_absolute_error": shot_mean_absolute_error,
        "shot_max_condition_absolute_error": shot_max_condition_absolute_error,
        "shot_threshold_passed": shot_threshold_passed,
        "numerical_thresholds_passed": numerical_pass,
        "hardware_promotion_recommended": bool(numerical_pass and enough_seeds),
        "promotion_note": (
            "All thresholds and the five-seed requirement passed."
            if numerical_pass and enough_seeds
            else "Reference validation is not sufficient for hardware promotion."
        ),
        "wall_seconds": time.perf_counter() - started,
        "config": str(config_path),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
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
