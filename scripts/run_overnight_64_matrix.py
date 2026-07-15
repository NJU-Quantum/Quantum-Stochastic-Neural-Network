"""Run the overnight 64-dimensional QGAN comparison sequentially on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/qgan/overnight_64_matrix",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def command_for(name: str, config: str, output_root: Path, args, extra=()):
    return [
        sys.executable,
        str(ROOT / "scripts/train_qgan.py"),
        "--config",
        str(ROOT / config),
        "--device",
        "cuda",
        "--epochs",
        str(args.epochs),
        "--samples-per-class",
        str(args.samples_per_class),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(output_root / name),
        *extra,
    ]


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    jobs = [
        ("qsnn_full", "configs/mnist0_64.yaml", ()),
        ("vqc", "configs/mnist0_64_vqc.yaml", ()),
        ("qsnn_l_only", "configs/mnist0_64.yaml", ("--ablation", "l_only")),
        ("qsnn_h_only", "configs/mnist0_64.yaml", ("--ablation", "h_only")),
    ]
    commands = [command_for(name, config, output_root, args, extra) for name, config, extra in jobs]
    if args.dry_run:
        for command in commands:
            print(subprocess.list2cmdline(command))
        return

    status = {
        "state": "running",
        "pid": os.getpid(),
        "started_at": timestamp(),
        "finished_at": None,
        "current_job": None,
        "jobs": {
            name: {"state": "pending", "started_at": None, "finished_at": None, "returncode": None}
            for name, _, _ in jobs
        },
    }
    write_status(status_path, status)
    failures = []
    for (name, _config, _extra), command in zip(jobs, commands):
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        status["current_job"] = name
        status["jobs"][name].update(state="running", started_at=timestamp())
        write_status(status_path, status)
        with (run_dir / "train.log").open("w", encoding="utf-8", buffering=1) as log:
            log.write(f"COMMAND: {subprocess.list2cmdline(command)}\n")
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
            )
        job_state = "completed" if result.returncode == 0 else "failed"
        status["jobs"][name].update(
            state=job_state,
            finished_at=timestamp(),
            returncode=result.returncode,
        )
        if result.returncode != 0:
            failures.append(name)
        write_status(status_path, status)

    summary_command = [
        sys.executable,
        str(ROOT / "scripts/summarize_qgan_matrix.py"),
        "--root",
        str(output_root),
    ]
    with (output_root / "summary.log").open("w", encoding="utf-8", buffering=1) as log:
        summary = subprocess.run(
            summary_command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
    if summary.returncode != 0:
        failures.append("summary")
    status.update(
        state="completed" if not failures else "completed_with_errors",
        current_job=None,
        finished_at=timestamp(),
        failures=failures,
    )
    write_status(status_path, status)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
