"""Run the focused Autoencoder-64 QSNN-full versus VQC comparison sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/qgan/ae64_comparison"))
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_status(path: Path, **values) -> None:
    current = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(values)
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def main():
    args = parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "runner_status.json"
    write_status(
        status_path,
        status="running",
        stage="starting",
        started_at=timestamp(),
        python=sys.executable,
    )

    shared = ["--device", args.device]
    if args.epochs is not None:
        shared += ["--epochs", str(args.epochs)]
    if args.samples_per_class is not None:
        shared += ["--samples-per-class", str(args.samples_per_class)]
    if args.max_steps_per_epoch is not None:
        shared += ["--max-steps-per-epoch", str(args.max_steps_per_epoch)]

    runs = [
        ("qsnn_full", ROOT / "configs/mnist0_ae64_qsnn.yaml"),
        ("vqc", ROOT / "configs/mnist0_ae64_vqc.yaml"),
    ]
    try:
        for name, config in runs:
            run_dir = output_root / name
            if (run_dir / "metrics.csv").exists():
                raise FileExistsError(
                    f"Refusing to overwrite an existing formal run: {run_dir}"
                )
            write_status(status_path, stage=name, stage_started_at=timestamp())
            command = [
                sys.executable,
                str(ROOT / "scripts/train_qgan.py"),
                "--config",
                str(config),
                "--output-dir",
                str(run_dir),
                *shared,
            ]
            run_logged(command, output_root / f"{name}.log")

        write_status(status_path, stage="summary", stage_started_at=timestamp())
        run_logged(
            [
                sys.executable,
                str(ROOT / "scripts/summarize_qgan_matrix.py"),
                "--root",
                str(output_root),
            ],
            output_root / "summary.log",
        )
    except Exception as error:
        write_status(
            status_path,
            status="failed",
            failed_at=timestamp(),
            error=f"{type(error).__name__}: {error}",
        )
        raise

    write_status(
        status_path,
        status="completed",
        stage="completed",
        completed_at=timestamp(),
    )


if __name__ == "__main__":
    main()
