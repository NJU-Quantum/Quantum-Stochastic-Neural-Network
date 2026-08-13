from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MATRIX = SCRIPT_DIR / "outputs" / "photonic_qsnn_ising.npy"
DEFAULT_REPORT = SCRIPT_DIR / "outputs" / "kaiwu_cim_repro.json"


def load_matrix(path: Path) -> np.ndarray:
    matrix = np.asarray(np.load(path), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Ising matrix must be square.")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError("Ising matrix must be finite and symmetric.")
    return matrix


def energies(matrix: np.ndarray, solutions: np.ndarray) -> np.ndarray:
    solutions = np.asarray(solutions, dtype=np.float64)
    return np.einsum("bi,ij,bj->b", solutions, matrix, solutions)


def summarize(kind: str, matrix: np.ndarray, solutions: np.ndarray) -> Dict[str, Any]:
    values = energies(matrix, solutions)
    return {
        "solver": kind,
        "solutions": int(solutions.shape[0]),
        "variables": int(solutions.shape[1]),
        "best_energy": float(values.min()),
        "mean_energy": float(values.mean()),
        "best_solution": solutions[int(values.argmin())].astype(int).tolist(),
    }


def load_cloud_credentials() -> tuple[str, str, str]:
    load_dotenv(REPO_ROOT / ".env", override=False)
    access_key = os.environ.get("WUYUE_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("WUYUE_ACCESS_KEY_SECRET", "").strip()
    device_id = os.environ.get(
        "WUYUE_PHOTONIC_DEVICE_ID", "WuYue-QPU-Qboson-1000"
    ).strip()
    if not access_key or not secret_key:
        raise RuntimeError(
            "WUYUE_ACCESS_KEY_ID and WUYUE_ACCESS_KEY_SECRET must be set in .env."
        )
    return access_key, secret_key, device_id


def solve_sa(matrix: np.ndarray) -> np.ndarray:
    import kaiwu as kw

    optimizer = kw.classical.SimulatedAnnealingOptimizer(
        alpha=0.999,
        size_limit=100,
    )
    return np.asarray(optimizer.solve(matrix), dtype=np.float64)


def solve_cim(matrix: np.ndarray, task_name: str) -> tuple[np.ndarray, str, str]:
    import kaiwu as kw

    access_key, secret_key, device_id = load_cloud_credentials()
    kw.common.CheckpointManager.save_dir = str(SCRIPT_DIR / "outputs" / "kaiwu_tasks")
    optimizer = kw.cim.CIMOptimizer(
        task_name=task_name,
        wait=True,
        access_key=access_key,
        secret_key=secret_key,
        device_id=device_id,
    )
    reducer_cls = getattr(getattr(kw, "preprocess", None), "PrecisionReducer", None)
    if reducer_cls is None:
        reducer_cls = kw.cim.PrecisionReducer
    optimizer = reducer_cls(optimizer, precision=8)
    solutions = np.asarray(optimizer.solve(matrix), dtype=np.float64)
    task_files = sorted(
        (SCRIPT_DIR / "outputs" / "kaiwu_tasks").glob(f"{task_name}_*_task_id.txt"),
        key=lambda path: path.stat().st_mtime,
    )
    task_id = task_files[-1].read_text(encoding="utf-8").strip() if task_files else ""
    return solutions, device_id, task_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible Kaiwu SA/CIM QSNN solve.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--mode", choices=("sa", "cim", "both"), default="both")
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    report: Dict[str, Any] = {
        "matrix": str(args.matrix.resolve()),
        "shape": list(matrix.shape),
        "timestamp": int(time.time()),
        "results": {},
    }
    if args.mode in ("sa", "both"):
        report["results"]["sa"] = summarize("Kaiwu SA", matrix, solve_sa(matrix))
    if args.mode in ("cim", "both"):
        task_name = args.task_name or f"QSNN_CIM_{int(time.time())}"[-24:]
        solutions, device_id, task_id = solve_cim(matrix, task_name)
        report["results"]["cim"] = summarize(
            "Kaiwu CIM",
            matrix,
            solutions,
        )
        report["results"]["cim"]["device_id"] = device_id
        report["results"]["cim"]["task_name"] = task_name
        report["results"]["cim"]["task_id"] = task_id
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
