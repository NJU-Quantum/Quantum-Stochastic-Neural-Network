from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv
from scipy.optimize import minimize


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"


@dataclass(frozen=True)
class VQCConfig:
    qubits: int = 9
    data_qubits: Tuple[int, int, int] = (2, 3, 4)
    readout_qubit: int = 4
    train_samples: int = 96
    test_samples: int = 48
    noise: float = 0.06
    seed: int = 23


CONFIG = VQCConfig()


def make_circle_dataset(config: VQCConfig = CONFIG) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the same inner/outer-ring task used by the photonic model."""
    total = config.train_samples + config.test_samples
    rng = np.random.default_rng(config.seed)
    labels = np.arange(total, dtype=np.int64) % 2
    angles = rng.uniform(0.0, 2.0 * np.pi, total)
    radius = np.where(labels == 0, 0.35, 0.80) + rng.normal(0.0, config.noise, total)
    xy = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    order = rng.permutation(total)
    xy, labels = xy[order], labels[order]
    split = np.where(np.arange(total) < config.train_samples, "train", "test")
    return xy, labels, split


def processed_features(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack((x, y, x * x + y * y, x * y))


def export_dataset(
    xy: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Dict[str, str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "circle_raw.csv"
    processed_path = data_dir / "circle_processed.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("sample_id", "x", "y", "label", "split"))
        for index, ((x, y), label, part) in enumerate(zip(xy, labels, split)):
            writer.writerow((index, f"{x:.12f}", f"{y:.12f}", int(label), part))
    features = processed_features(xy)
    with processed_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("sample_id", "x", "y", "radius_squared", "xy", "label", "split"))
        for index, (row, label, part) in enumerate(zip(features, labels, split)):
            writer.writerow((index, *(f"{value:.12f}" for value in row), int(label), part))
    return {"raw": str(raw_path), "processed": str(processed_path)}


def clip_unit(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def configure_wuyue() -> None:
    from wuyue.utils import set_ml_backend

    set_ml_backend("numpy", dtype="float64")


def build_vqc_circuit(
    xy: Sequence[float],
    params: Sequence[float],
    config: VQCConfig = CONFIG,
    measure: bool = True,
) -> Any:
    """Build a WuYue-native VQC using the Baihua-accepted 2-3-4 chain."""
    configure_wuyue()
    from wuyue.circuit import QuantumCircuit
    from wuyue.element.gate import CX, MEASURE, RY, RZ
    from wuyue.register.classicalregister import ClassicalRegister
    from wuyue.register.quantumregister import QuantumRegister

    p = np.asarray(params, dtype=np.float64)
    if p.shape != (8,):
        raise ValueError("VQC requires exactly 8 trainable parameters.")
    x, y = (clip_unit(v) for v in xy)
    r2 = min((x * x + y * y) / 1.2, 1.0)
    cross = clip_unit(2.0 * x * y)
    q0, q1, q2 = config.data_qubits
    qreg = QuantumRegister(config.qubits)
    creg = ClassicalRegister(config.qubits)
    circuit = QuantumCircuit(qreg, creg, name="WuYue_VQC_Baihua")

    # Angle encoding and data re-uploading expose radial and angular structure.
    circuit.add(RY, qreg[q0], paras=0.5 * math.pi * (x + 1.0))
    circuit.add(RY, qreg[q1], paras=0.5 * math.pi * (y + 1.0))
    circuit.add(RY, qreg[q2], paras=p[0] * r2 + p[1])
    circuit.add(RZ, qreg[q0], paras=p[2] * cross)
    circuit.add(CX, qreg[q1], control=qreg[q0])
    circuit.add(RY, qreg[q0], paras=p[3])
    circuit.add(RY, qreg[q1], paras=p[4])
    circuit.add(CX, qreg[q1], control=qreg[q0])
    # CX-RZ-CX realizes a trainable ZZ interaction on the Baihua 3-4 edge.
    circuit.add(CX, qreg[q2], control=qreg[q1])
    circuit.add(RZ, qreg[q2], paras=p[5])
    circuit.add(CX, qreg[q2], control=qreg[q1])
    circuit.add(RY, qreg[q2], paras=p[6] + p[7] * r2)
    if measure:
        circuit.all_add(MEASURE)
    return circuit


def _apply_single(state: np.ndarray, gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
    tensor = state.reshape((2,) * n)
    axis = n - 1 - qubit
    moved = np.moveaxis(tensor, axis, 0).reshape(2, -1)
    moved = gate @ moved
    return np.moveaxis(moved.reshape((2,) + (2,) * (n - 1)), 0, axis).reshape(-1)


def _apply_cx(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    result = state.copy()
    for index in range(1 << n):
        if ((index >> control) & 1) and not ((index >> target) & 1):
            pair = index | (1 << target)
            result[index], result[pair] = state[pair], state[index]
    return result


def fast_probability(xy: Sequence[float], params: Sequence[float], config: VQCConfig = CONFIG) -> float:
    """Exact statevector equivalent used for efficient classical optimization."""
    p = np.asarray(params, dtype=np.float64)
    x, y = (clip_unit(v) for v in xy)
    r2 = min((x * x + y * y) / 1.2, 1.0)
    cross = clip_unit(2.0 * x * y)
    q0, q1, q2 = config.data_qubits
    state = np.zeros(1 << config.qubits, dtype=np.complex128)
    state[0] = 1.0

    def ry(theta: float) -> np.ndarray:
        return np.array([[math.cos(theta / 2), -math.sin(theta / 2)],
                         [math.sin(theta / 2), math.cos(theta / 2)]], dtype=np.complex128)

    def rz(theta: float) -> np.ndarray:
        return np.diag((np.exp(-0.5j * theta), np.exp(0.5j * theta)))

    for qubit, gate in (
        (q0, ry(0.5 * math.pi * (x + 1.0))),
        (q1, ry(0.5 * math.pi * (y + 1.0))),
        (q2, ry(p[0] * r2 + p[1])),
        (q0, rz(p[2] * cross)),
    ):
        state = _apply_single(state, gate, qubit, config.qubits)
    state = _apply_cx(state, q0, q1, config.qubits)
    for qubit, theta in ((q0, p[3]), (q1, p[4])):
        state = _apply_single(state, ry(theta), qubit, config.qubits)
    state = _apply_cx(state, q0, q1, config.qubits)
    state = _apply_cx(state, q1, q2, config.qubits)
    state = _apply_single(state, rz(p[5]), q2, config.qubits)
    state = _apply_cx(state, q1, q2, config.qubits)
    state = _apply_single(state, ry(p[6] + p[7] * r2), q2, config.qubits)
    probs = np.abs(state) ** 2
    return float(sum(prob for index, prob in enumerate(probs) if (index >> config.readout_qubit) & 1))


def wuyue_probability(xy: Sequence[float], params: Sequence[float], config: VQCConfig = CONFIG) -> float:
    from wuyue.backend import Backend

    circuit = build_vqc_circuit(xy, params, config=config, measure=False)
    simulator = Backend.get_device("Full amplitude")
    simulator.apply(circuit)
    probs = np.asarray(simulator.get_probs(), dtype=np.float64)
    return float(sum(prob for index, prob in enumerate(probs) if (index >> config.readout_qubit) & 1))


def binary_cross_entropy(params: np.ndarray, xy: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.array([fast_probability(sample, params) for sample in xy])
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)))


def train_vqc(xy: np.ndarray, labels: np.ndarray, maxiter: int = 120) -> Dict[str, Any]:
    # Radial warm start maps the expected inner/outer radii near |0>/|1>.
    initial = np.array((6.0, -0.72, 0.1, 0.0, 0.0, 0.05, 0.0, 0.0), dtype=np.float64)
    started = time.perf_counter()
    result = minimize(
        binary_cross_entropy,
        initial,
        args=(xy, labels),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-5},
    )
    return {
        "params": result.x.tolist(),
        "initial_loss": binary_cross_entropy(initial, xy, labels),
        "final_loss": float(result.fun),
        "iterations": int(result.nit),
        "evaluations": int(result.nfev),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "train_seconds": time.perf_counter() - started,
    }


def evaluate(xy: np.ndarray, labels: np.ndarray, params: Sequence[float]) -> Dict[str, Any]:
    started = time.perf_counter()
    probabilities = np.array([fast_probability(sample, params) for sample in xy])
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "loss": binary_cross_entropy(np.asarray(params), xy, labels),
        "probabilities": probabilities.tolist(),
        "predictions": predictions.tolist(),
        "labels": labels.astype(int).tolist(),
        "seconds": time.perf_counter() - started,
    }


def verify_wuyue_backend(
    xy: np.ndarray,
    params: Sequence[float],
    sample_indices: Iterable[int] = (0, 1, 2, 3, 4, 5),
) -> Dict[str, Any]:
    rows = []
    for index in sample_indices:
        fast = fast_probability(xy[index], params)
        sdk = wuyue_probability(xy[index], params)
        rows.append({"index": int(index), "fast_p1": fast, "wuyue_p1": sdk, "abs_error": abs(fast - sdk)})
    maximum = max(row["abs_error"] for row in rows)
    return {"samples": rows, "max_abs_error": maximum, "passed": maximum < 1e-10}


def decode_counts(counts: Dict[str, Any], readout_qubit: int = CONFIG.readout_qubit) -> Dict[str, Any]:
    parsed = {str(key): int(value) for key, value in counts.items()}
    shots = max(sum(parsed.values()), 1)
    class1 = 0
    for bitstring, count in parsed.items():
        bits = bitstring.replace(" ", "")[::-1]
        if len(bits) > readout_qubit and bits[readout_qubit] == "1":
            class1 += count
    p1 = class1 / shots
    return {"shots": shots, "p0": 1.0 - p1, "p1": p1, "pred": int(p1 >= 0.5)}


def cloud_runner() -> Any:
    from wuyue.plugin.runner import Runner

    load_dotenv(REPO_ROOT / ".env", override=False)
    access_key = os.environ.get("WUYUE_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("WUYUE_ACCESS_KEY_SECRET", "").strip()
    if not access_key or not secret_key:
        raise RuntimeError("WuYue AK/SK are missing from .env.")
    return Runner(access_key=access_key, secret_key=secret_key, auto_retry=True)


def submit_cloud(
    circuit: Any,
    device_id: str,
    shots: int,
    timeout: int,
    task_prefix: str,
) -> Dict[str, Any]:
    result = cloud_runner().run(
        circuit,
        qubits=CONFIG.qubits,
        shots=shots,
        task_name=f"{task_prefix}_{int(time.time())}"[-24:],
        device_id=device_id,
        timeout=timeout,
        calculate_type=1,
        circuit_optimization=True,
        qubit_mapping=True,
        gate_decomposition=True,
    )
    counts = result.get_counts() or {}
    return {
        "device_id": result.get_device_id(),
        "task_id": result.get_task_id(),
        "success": bool(result.get_success()),
        "status": result.get_status(),
        "counts": counts,
        "decoded": decode_counts(counts) if counts else None,
        "probabilities": result.get_prob(),
        "raw_result": result.get_raw_result(),
    }


def poll_task(task_id: str, device_id: str, timeout: int) -> Dict[str, Any]:
    runner = cloud_runner()
    started = time.time()
    while True:
        response = runner.get_task_result(task_id)
        if response.code != 1:
            raise RuntimeError(f"Task query failed: {response.error_code}{response.msg}")
        data = response.data
        if data.task_status in (5, 6):
            counts = ast.literal_eval(data.out_counts or "{}")
            return {
                "device_id": device_id,
                "task_id": task_id,
                "success": data.task_status == 5,
                "status": "计算成功" if data.task_status == 5 else "计算失败",
                "counts": counts,
                "decoded": decode_counts(counts) if counts else None,
                "probabilities": ast.literal_eval(data.out_probs or "{}"),
                "raw_result": data.out_data,
            }
        if time.time() - started >= timeout:
            return {
                "device_id": device_id,
                "task_id": task_id,
                "success": False,
                "status_code": data.task_status,
                "status": "timeout",
            }
        time.sleep(5)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="WuYue-native trainable VQC for Baihua.")
    parser.add_argument("--mode", choices=("train", "local", "cloud-simulator", "baihua", "poll"), default="train")
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    xy, labels, split = make_circle_dataset()
    data_files = export_dataset(xy, labels, split)
    model_path = args.output_dir / "vqc_model.json"
    if args.mode == "train" or not model_path.exists():
        training = train_vqc(xy[: CONFIG.train_samples], labels[: CONFIG.train_samples], args.maxiter)
        params = training["params"]
    else:
        stored = json.loads(model_path.read_text(encoding="utf-8"))
        training = stored["training"]
        params = stored["training"]["params"]

    train_metrics = evaluate(xy[: CONFIG.train_samples], labels[: CONFIG.train_samples], params)
    test_metrics = evaluate(xy[CONFIG.train_samples :], labels[CONFIG.train_samples :], params)
    verification = verify_wuyue_backend(xy, params)
    payload: Dict[str, Any] = {
        "algorithm": "WuYue-native variational quantum classifier",
        "config": asdict(CONFIG),
        "gate_set": ["RY", "RZ", "CX", "MEASURE"],
        "baihua_edges": [[2, 3], [3, 4]],
        "data_files": data_files,
        "training": training,
        "train": train_metrics,
        "test": test_metrics,
        "wuyue_backend_verification": verification,
    }
    save_json(model_path, payload)

    sample = CONFIG.train_samples + (args.sample_index % CONFIG.test_samples)
    circuit = build_vqc_circuit(xy[sample], params, measure=True)
    payload["representative_sample"] = {
        "index": sample,
        "xy": xy[sample].tolist(),
        "label": int(labels[sample]),
        "exact_p1": fast_probability(xy[sample], params),
        "qasm": circuit.QASM(),
    }
    if args.mode == "cloud-simulator":
        device = args.device_id or os.environ.get("WUYUE_SIMULATOR_DEVICE_ID", "WuYue-QPUSim-FullAmpSim")
        payload["cloud"] = submit_cloud(circuit, device, args.shots, args.timeout, "WYVQC_SIM")
    elif args.mode == "baihua":
        device = args.device_id or "WuYue-QPU-Baihua"
        if not verification["passed"] or test_metrics["accuracy"] < 0.80:
            raise RuntimeError("Local VQC gate failed; Baihua submission blocked.")
        try:
            payload["cloud"] = submit_cloud(circuit, device, args.shots, args.timeout, "WYVQC_BH")
        except RuntimeError as exc:
            payload["cloud"] = {
                "device_id": device,
                "success": False,
                "status": "submission_rejected",
                "error": str(exc),
                "attempted_at": int(time.time()),
            }
    elif args.mode == "poll":
        if not args.task_id or not args.device_id:
            raise ValueError("poll mode requires --task-id and --device-id")
        payload["cloud"] = poll_task(args.task_id, args.device_id, args.timeout or 600)
    report_path = args.output_dir / f"vqc_{args.mode}_report.json"
    save_json(report_path, payload)
    print(json.dumps({
        "report": str(report_path),
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "wuyue_verified": verification["passed"],
        "cloud": payload.get("cloud"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
