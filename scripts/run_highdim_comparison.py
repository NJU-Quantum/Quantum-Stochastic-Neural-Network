"""Run 128D, 256D, and full-resolution QSNN/VQC comparisons on a GPU pool."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

AE_CONFIGS = {
    128: ROOT / "configs/autoencoder_mnist0_128.yaml",
    256: ROOT / "configs/autoencoder_mnist0_256.yaml",
}
AE_CHECKPOINTS = {
    128: ROOT / "outputs/autoencoder/mnist0_ae128_balanced/checkpoint_best.pt",
    256: ROOT / "outputs/autoencoder/mnist0_ae256_balanced/checkpoint_best.pt",
}
QGAN_CONFIGS = {
    (128, "qsnn_full"): ROOT / "configs/mnist0_ae128_qsnn.yaml",
    (128, "vqc"): ROOT / "configs/mnist0_ae128_vqc.yaml",
    (256, "qsnn_full"): ROOT / "configs/mnist0_ae256_qsnn.yaml",
    (256, "vqc"): ROOT / "configs/mnist0_ae256_vqc.yaml",
    (784, "qsnn_full"): ROOT / "configs/mnist0_full784_qsnn.yaml",
    (784, "vqc"): ROOT / "configs/mnist0_full784_vqc.yaml",
}


@dataclass
class Task:
    name: str
    command: list[str]
    log_path: Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[128, 256, 784])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("qsnn_full", "vqc"),
        default=["qsnn_full", "vqc"],
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0"],
        help="Worker devices, e.g. --devices cuda:0 cuda:1 cuda:2 cuda:3",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/qgan/highdim_comparison"),
    )
    parser.add_argument(
        "--autoencoder-root",
        type=Path,
        help="Optional alternate root containing dim128/ and dim256/ Autoencoder runs",
    )
    parser.add_argument("--ae-epochs", type=int)
    parser.add_argument("--qgan-epochs", type=int)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--skip-autoencoders", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_status(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_args(args) -> None:
    unsupported = sorted(set(args.dimensions) - {128, 256, 784})
    if unsupported:
        raise ValueError(f"Unsupported dimensions: {unsupported}; choose from 128, 256, 784")
    if not args.devices:
        raise ValueError("At least one device is required")
    if len(set(args.devices)) != len(args.devices):
        raise ValueError("Worker devices must be unique")
    for value in (args.ae_epochs, args.qgan_epochs, args.samples_per_class):
        if value is not None and value <= 0:
            raise ValueError("Epoch and sample overrides must be positive")


def prepare_mnist_download() -> None:
    from torchvision.datasets import MNIST

    MNIST(root=str(ROOT / "datasets"), train=True, download=True)


def make_ae_tasks(args, log_root: Path, checkpoints: dict[int, Path]) -> list[Task]:
    tasks = []
    if args.skip_autoencoders:
        missing = [
            str(checkpoints[dim])
            for dim in args.dimensions
            if dim in checkpoints and not checkpoints[dim].exists()
        ]
        if missing and not args.dry_run:
            raise FileNotFoundError(
                "--skip-autoencoders was set, but checkpoints are missing: " + ", ".join(missing)
            )
        return tasks
    for dim in args.dimensions:
        if dim not in AE_CONFIGS or checkpoints[dim].exists():
            continue
        output_dir = checkpoints[dim].parent
        if output_dir.exists() and any(output_dir.iterdir()) and not args.dry_run:
            raise FileExistsError(
                f"Incomplete Autoencoder directory exists: {output_dir}. "
                "Move it aside or finish that run before starting the formal matrix."
            )
        command = [
            sys.executable,
            str(ROOT / "scripts/train_autoencoder.py"),
            "--config",
            str(AE_CONFIGS[dim]),
            "--output-dir",
            str(output_dir),
        ]
        if args.ae_epochs is not None:
            command += ["--epochs", str(args.ae_epochs)]
        if args.samples_per_class is not None:
            command += ["--samples-per-class", str(args.samples_per_class)]
        tasks.append(Task(f"ae{dim}", command, log_root / f"ae{dim}.log"))
    return tasks


def make_qgan_tasks(
    args,
    output_root: Path,
    log_root: Path,
    autoencoder_checkpoints: dict[int, Path],
) -> list[Task]:
    tasks = []
    for dim in args.dimensions:
        for model in args.models:
            run_dir = output_root / f"dim{dim}" / model
            metrics_path = run_dir / "metrics.csv"
            checkpoint_path = run_dir / "checkpoint_latest.pt"
            command = [
                sys.executable,
                str(ROOT / "scripts/train_qgan.py"),
                "--config",
                str(QGAN_CONFIGS[(dim, model)]),
                "--output-dir",
                str(run_dir),
            ]
            if dim in autoencoder_checkpoints:
                checkpoint = autoencoder_checkpoints[dim]
                if not checkpoint.exists() and not args.dry_run:
                    raise FileNotFoundError(f"Autoencoder checkpoint not found: {checkpoint}")
                command += ["--autoencoder-checkpoint", str(checkpoint)]
            if metrics_path.exists():
                if not args.resume:
                    raise FileExistsError(
                        f"Formal run already exists: {run_dir}. Use --resume to continue it."
                    )
                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Cannot resume {run_dir}: checkpoint_latest.pt is missing"
                    )
                command += ["--resume", str(checkpoint_path)]
            elif run_dir.exists() and any(run_dir.iterdir()) and not args.dry_run:
                raise FileExistsError(
                    f"Non-empty run directory has no resumable metrics: {run_dir}"
                )
            if args.qgan_epochs is not None:
                command += ["--epochs", str(args.qgan_epochs)]
            if args.samples_per_class is not None:
                command += ["--samples-per-class", str(args.samples_per_class)]
            if args.max_steps_per_epoch is not None:
                command += ["--max-steps-per-epoch", str(args.max_steps_per_epoch)]
            tasks.append(
                Task(
                    f"dim{dim}_{model}",
                    command,
                    log_root / f"dim{dim}_{model}.log",
                )
            )
    return tasks


def run_pool(
    tasks: list[Task],
    devices: list[str],
    status_path: Path,
    status: dict,
    *,
    dry_run: bool,
) -> None:
    if dry_run:
        for index, task in enumerate(tasks):
            device = devices[index % len(devices)]
            print(f"[{task.name}] {' '.join(task.command + ['--device', device])}")
        return

    pending = list(tasks)
    active: dict[str, tuple[Task, subprocess.Popen, object]] = {}
    while pending or active:
        for device in devices:
            if not pending or device in active:
                continue
            task = pending.pop(0)
            task.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = task.log_path.open("w", encoding="utf-8")
            command = [*task.command, "--device", device]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[device] = (task, process, log)
            status["tasks"][task.name] = {
                "status": "running",
                "device": device,
                "pid": process.pid,
                "started_at": timestamp(),
                "log": str(task.log_path),
                "command": command,
            }
            write_status(status_path, status)

        failed = None
        for device, (task, process, log) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log.close()
            del active[device]
            status["tasks"][task.name].update(
                {
                    "status": "completed" if return_code == 0 else "failed",
                    "return_code": return_code,
                    "finished_at": timestamp(),
                }
            )
            write_status(status_path, status)
            if return_code != 0:
                failed = (task, return_code)
                break
        if failed is not None:
            for _device, (_task, process, log) in active.items():
                process.terminate()
                log.close()
            raise subprocess.CalledProcessError(failed[1], failed[0].command)
        if pending or active:
            time.sleep(1.0)


def summarize(output_root: Path, dimensions: list[int]) -> None:
    for dim in dimensions:
        run_root = output_root / f"dim{dim}"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/summarize_qgan_matrix.py"),
                "--root",
                str(run_root),
            ],
            cwd=ROOT,
            check=True,
        )


def main():
    args = parse_args()
    validate_args(args)
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    if args.autoencoder_root is None:
        autoencoder_checkpoints = dict(AE_CHECKPOINTS)
    else:
        autoencoder_root = (
            args.autoencoder_root
            if args.autoencoder_root.is_absolute()
            else ROOT / args.autoencoder_root
        )
        autoencoder_checkpoints = {
            dim: autoencoder_root / f"dim{dim}" / "checkpoint_best.pt"
            for dim in AE_CONFIGS
        }
    log_root = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "runner_status.json"
    status = {
        "status": "running",
        "started_at": timestamp(),
        "python": sys.executable,
        "dimensions": args.dimensions,
        "models": args.models,
        "devices": args.devices,
        "tasks": {},
    }
    write_status(status_path, status)
    try:
        if args.download and not args.dry_run:
            prepare_mnist_download()
        ae_tasks = make_ae_tasks(args, log_root, autoencoder_checkpoints)
        run_pool(
            ae_tasks,
            args.devices,
            status_path,
            status,
            dry_run=args.dry_run,
        )
        qgan_tasks = make_qgan_tasks(
            args,
            output_root,
            log_root,
            autoencoder_checkpoints,
        )
        run_pool(
            qgan_tasks,
            args.devices,
            status_path,
            status,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            summarize(output_root, args.dimensions)
    except Exception as error:
        status.update(
            {
                "status": "failed",
                "failed_at": timestamp(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        write_status(status_path, status)
        raise
    status.update(
        {
            "status": "dry_run" if args.dry_run else "completed",
            "completed_at": timestamp(),
        }
    )
    write_status(status_path, status)


if __name__ == "__main__":
    main()
