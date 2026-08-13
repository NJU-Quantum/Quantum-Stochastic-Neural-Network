from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "large_scale"


@dataclass(frozen=True)
class LargeScaleLayout:
    input_nodes: int = 128
    reservoir_width: int = 256
    reservoir_layers: int = 3
    sink_nodes_per_class: int = 51

    @property
    def reservoir_nodes(self) -> int:
        return self.reservoir_width * self.reservoir_layers

    @property
    def sink0_start(self) -> int:
        return self.input_nodes + self.reservoir_nodes

    @property
    def sink1_start(self) -> int:
        return self.sink0_start + self.sink_nodes_per_class

    @property
    def selector(self) -> int:
        return self.sink1_start + self.sink_nodes_per_class

    @property
    def model_spins(self) -> int:
        return self.selector + 1

    @property
    def bias_spin(self) -> int:
        return self.model_spins


LAYOUT = LargeScaleLayout()
assert LAYOUT.model_spins == 999


def make_circle_dataset(samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.arange(samples) % 2
    angles = rng.uniform(0.0, 2.0 * np.pi, samples)
    radii = np.where(labels == 0, 0.35, 0.80) + rng.normal(0.0, 0.06, samples)
    xy = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    order = rng.permutation(samples)
    return xy[order], labels[order]


def encode_inputs(xy: np.ndarray) -> np.ndarray:
    """Encode x, y, radius and xy into four 32-spin thermometer banks."""
    xy = np.asarray(xy, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    radius = np.sqrt(x * x + y * y)
    cross = x * y
    banks = (
        (x, np.linspace(-1.0, 1.0, 32)),
        (y, np.linspace(-1.0, 1.0, 32)),
        (radius, np.linspace(0.0, 1.2, 32)),
        (cross, np.linspace(-0.5, 0.5, 32)),
    )
    return np.concatenate(
        [np.where(values[:, None] >= thresholds, 1.0, -1.0) for values, thresholds in banks],
        axis=1,
    )


def sparse_projection(
    rows: int,
    cols: int,
    fan_in: int,
    rng: np.random.Generator,
) -> np.ndarray:
    matrix = np.zeros((rows, cols), dtype=np.float64)
    for row in range(rows):
        chosen = rng.choice(cols, size=min(fan_in, cols), replace=False)
        matrix[row, chosen] = rng.choice((-1.0, 1.0), size=chosen.size) / np.sqrt(chosen.size)
    return matrix


def make_reservoir(seed: int) -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    matrices = [sparse_projection(256, 128, 16, rng)]
    matrices.extend(sparse_projection(256, 256, 12, rng) for _ in range(2))
    return tuple(matrices)


def reservoir_forward(inputs: np.ndarray, matrices: Tuple[np.ndarray, ...]) -> np.ndarray:
    state = inputs
    layers = []
    for matrix in matrices:
        # NumPy/Accelerate 2.2 may leave benign IEEE flags after finite GEMM.
        with np.errstate(all="ignore"):
            projected = state @ matrix.T
        if not np.isfinite(projected).all():
            raise FloatingPointError("Reservoir projection produced a non-finite value.")
        state = np.where(projected >= 0.0, 1.0, -1.0)
        layers.append(state)
    return np.concatenate(layers, axis=1)


def train_readout(features: np.ndarray, labels: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    targets = np.where(labels > 0, 1.0, -1.0)
    with np.errstate(all="ignore"):
        gram = features @ features.T + ridge * np.eye(features.shape[0])
    if not np.isfinite(gram).all():
        raise FloatingPointError("Readout Gram matrix is non-finite.")
    dual = np.linalg.solve(gram, targets)
    with np.errstate(all="ignore"):
        weights = features.T @ dual
    if not np.isfinite(weights).all():
        raise FloatingPointError("Readout weights are non-finite.")
    scale = max(float(np.sum(np.abs(weights))), 1e-12)
    return 8.0 * weights / scale


def add_edge(matrix: np.ndarray, a: int, b: int, coefficient: float) -> None:
    matrix[a, b] += coefficient / 2.0
    matrix[b, a] += coefficient / 2.0


def build_ising(
    encoded_input: np.ndarray,
    matrices: Tuple[np.ndarray, ...],
    readout: np.ndarray,
    layout: LargeScaleLayout = LAYOUT,
) -> np.ndarray:
    """Build one sample-conditioned 1000-spin Ising energy matrix."""
    size = layout.model_spins + 1
    ising = np.zeros((size, size), dtype=np.float64)
    bias = layout.bias_spin

    # Strong state preparation fields clamp only the physical input bank.
    for node, spin in enumerate(encoded_input):
        add_edge(ising, node, bias, -15.0 * float(spin))

    previous_start = 0
    previous_width = layout.input_nodes
    layer_start = layout.input_nodes
    for projection in matrices:
        rows, cols = np.nonzero(projection)
        for row, col in zip(rows, cols):
            add_edge(
                ising,
                layer_start + int(row),
                previous_start + int(col),
                -0.24 * float(projection[row, col]),
            )
        previous_start = layer_start
        previous_width = layout.reservoir_width
        layer_start += layout.reservoir_width
    assert previous_width == layout.reservoir_width

    # Trainable multiscale semantic field from input and reservoir to selector.
    for offset, weight in enumerate(readout):
        if abs(weight) > 1e-10:
            add_edge(ising, offset, layout.selector, -float(weight))
            sink_offset = offset % layout.sink_nodes_per_class
            add_edge(ising, offset, layout.sink0_start + sink_offset, float(weight))
            add_edge(ising, offset, layout.sink1_start + sink_offset, -float(weight))

    sink0 = range(layout.sink0_start, layout.sink0_start + layout.sink_nodes_per_class)
    sink1 = range(layout.sink1_start, layout.sink1_start + layout.sink_nodes_per_class)
    for group in (tuple(sink0), tuple(sink1)):
        for index, node in enumerate(group):
            add_edge(ising, node, group[(index + 1) % len(group)], -0.70)
            add_edge(ising, node, group[(index + 7) % len(group)], -0.25)
    for node0, node1 in zip(sink0, sink1):
        add_edge(ising, node0, layout.selector, 0.03)
        add_edge(ising, node1, layout.selector, -0.03)
        add_edge(ising, node0, node1, 0.20)

    # Kaiwu maximizes s^T J s, while the construction above minimizes H(s).
    kaiwu_matrix = -ising
    kaiwu_matrix /= max(float(np.max(np.abs(kaiwu_matrix))), 1.0)
    quantized = np.rint(127.0 * kaiwu_matrix).astype(np.int16)
    return np.rint((quantized + quantized.T) / 2.0).astype(np.int16)


def energy(matrix: np.ndarray, spins: np.ndarray) -> np.ndarray:
    return -np.einsum("bi,ij,bj->b", spins, matrix, spins)


def decode_solutions(solutions: np.ndarray, layout: LargeScaleLayout = LAYOUT) -> Dict[str, Any]:
    solutions = np.asarray(solutions)
    selector = solutions[:, layout.selector] * solutions[:, layout.bias_spin]
    return {
        "pred": int(np.mean(selector > 0) >= 0.5),
        "class1_fraction": float(np.mean(selector > 0)),
        "solutions": int(solutions.shape[0]),
    }


def solve_sa(matrix: np.ndarray, seed: int) -> np.ndarray:
    import kaiwu as kw

    optimizer = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=10000,
        alpha=0.995,
        cutoff_temperature=1.0,
        iterations_per_t=2,
        size_limit=1000,
        rand_seed=seed,
        process_num=1,
    )
    return np.asarray(optimizer.solve(matrix), dtype=np.float64)


def solve_cim(matrix: np.ndarray, task_name: str, output_dir: Path) -> Tuple[np.ndarray, str]:
    import kaiwu as kw

    load_dotenv(REPO_ROOT / ".env", override=False)
    access_key = os.environ["WUYUE_ACCESS_KEY_ID"]
    secret_key = os.environ["WUYUE_ACCESS_KEY_SECRET"]
    device_id = os.environ.get("WUYUE_PHOTONIC_DEVICE_ID", "WuYue-QPU-Qboson-1000")
    kw.common.CheckpointManager.save_dir = str(output_dir / "kaiwu_tasks")
    optimizer = kw.cim.CIMOptimizer(
        task_name=task_name,
        wait=True,
        access_key=access_key,
        secret_key=secret_key,
        device_id=device_id,
    )
    solutions = np.asarray(optimizer.solve(matrix), dtype=np.float64)
    task_files = sorted((output_dir / "kaiwu_tasks").glob(f"{task_name}_*_task_id.txt"))
    task_id = task_files[-1].read_text(encoding="utf-8").strip() if task_files else ""
    return solutions, task_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured 1000-spin photonic QSNN milestone.")
    parser.add_argument("--mode", choices=("build", "sa", "cim"), default="build")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sa-eval-samples", type=int, default=8)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    xy, labels = make_circle_dataset(args.samples, args.seed)
    split = int(0.8 * args.samples)
    encoded = encode_inputs(xy)
    projections = make_reservoir(args.seed)
    reservoir = reservoir_forward(encoded, projections)
    semantic = np.concatenate((encoded, reservoir), axis=1)
    readout = train_readout(semantic[:split], labels[:split])
    with np.errstate(all="ignore"):
        classical_scores = semantic[split:] @ readout
    if not np.isfinite(classical_scores).all():
        raise FloatingPointError("Classical readout produced a non-finite score.")
    classical_pred = (classical_scores > 0).astype(int)
    sample = split + (args.sample_index % max(args.samples - split, 1))
    matrix = build_ising(encoded[sample], projections, readout)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "large_qsnn_ising.npy", matrix)
    np.savez_compressed(
        args.output_dir / "large_qsnn_model.npz",
        readout=readout,
        projection0=projections[0],
        projection1=projections[1],
        projection2=projections[2],
    )
    report: Dict[str, Any] = {
        "architecture": "structured coherent-reservoir dissipative-attractor Ising QSNN",
        "layout": asdict(LAYOUT),
        "model_spins": LAYOUT.model_spins,
        "bias_spins": 1,
        "ising_shape": list(matrix.shape),
        "nonzero_matrix_entries": int(np.count_nonzero(matrix)),
        "independent_couplings": int(np.count_nonzero(np.triu(matrix, 1))),
        "coefficient_format": "signed-int8 stored as int16",
        "coefficient_min": int(matrix.min()),
        "coefficient_max": int(matrix.max()),
        "classical_reservoir_test_accuracy": float(np.mean(classical_pred == labels[split:])),
        "conditioned_sample_index": int(sample),
        "conditioned_xy": xy[sample].tolist(),
        "conditioned_label": int(labels[sample]),
        "timestamp": int(time.time()),
    }
    if args.mode in ("sa", "cim"):
        cases = []
        eval_count = min(args.sa_eval_samples, args.samples - split)
        for offset in range(eval_count):
            eval_sample = split + offset
            eval_matrix = build_ising(encoded[eval_sample], projections, readout)
            solutions = solve_sa(eval_matrix, args.seed + offset)
            decoded = decode_solutions(solutions)
            relative_inputs = solutions[:, : LAYOUT.input_nodes] * solutions[:, [LAYOUT.bias_spin]]
            cases.append(
                {
                    "sample_index": int(eval_sample),
                    "label": int(labels[eval_sample]),
                    "pred": decoded["pred"],
                    "class1_fraction": decoded["class1_fraction"],
                    "input_match_fraction": float(np.mean(relative_inputs == encoded[eval_sample])),
                    "best_energy": float(np.min(energy(eval_matrix, solutions))),
                }
            )
        sa_accuracy = float(np.mean([case["pred"] == case["label"] for case in cases]))
        input_match = float(np.mean([case["input_match_fraction"] for case in cases]))
        report["sa"] = {
            "evaluated_samples": eval_count,
            "accuracy": sa_accuracy,
            "mean_input_match_fraction": input_match,
            "passed": bool(sa_accuracy >= 0.75 and input_match >= 0.93),
            "cases": cases,
        }
        if args.mode == "cim" and not report["sa"]["passed"]:
            raise RuntimeError("1000-spin SA preflight failed; CIM submission blocked.")
    if args.mode == "cim":
        task_name = args.task_name or f"LargeQSNN1000_{int(time.time())}"[-24:]
        solutions, task_id = solve_cim(matrix, task_name, args.output_dir)
        report["cim"] = decode_solutions(solutions)
        report["cim"].update(
            {
                "best_energy": float(np.min(energy(matrix, solutions))),
                "device_id": os.environ.get("WUYUE_PHOTONIC_DEVICE_ID", "WuYue-QPU-Qboson-1000"),
                "task_name": task_name,
                "task_id": task_id,
            }
        )
    report_path = args.output_dir / "large_qsnn_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
