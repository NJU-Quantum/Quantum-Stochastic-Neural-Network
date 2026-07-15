"""Reproducible QGAN checkpoint save and restore helpers."""

from __future__ import annotations

import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import torch


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        git_directory = Path(__file__).resolve().parents[1] / ".git"
        try:
            head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref: "):
                return (git_directory / head[5:]).read_text(encoding="utf-8").strip()
            return head or None
        except OSError:
            return None


def runtime_metadata() -> dict[str, Any]:
    """Return the environment details needed to interpret a saved run."""
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git_commit(),
    }
    if torch.cuda.is_available():
        metadata.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": tuple(torch.cuda.get_device_capability(0)),
            }
        )
    return metadata


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def save_checkpoint(
    path: str | Path,
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    *,
    epoch: int,
    step: int,
    config: dict[str, Any],
    scheduler_g=None,
    scheduler_d=None,
    preprocessing_version: str = "probability_amplitude_v1",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save all train and random states required for an exact continuation."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "epoch": int(epoch),
        "step": int(step),
        "config": config,
        "preprocessing_version": preprocessing_version,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "scheduler_g": None if scheduler_g is None else scheduler_g.state_dict(),
        "scheduler_d": None if scheduler_d is None else scheduler_d.state_dict(),
        "rng_state": _rng_state(),
        "runtime": runtime_metadata(),
        "extra": extra or {},
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    *,
    scheduler_g=None,
    scheduler_d=None,
    map_location=None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint into already constructed model and optimizer objects."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format_version')}")
    generator.load_state_dict(payload["generator"])
    discriminator.load_state_dict(payload["discriminator"])
    optimizer_g.load_state_dict(payload["optimizer_g"])
    optimizer_d.load_state_dict(payload["optimizer_d"])
    if scheduler_g is not None and payload["scheduler_g"] is not None:
        scheduler_g.load_state_dict(payload["scheduler_g"])
    if scheduler_d is not None and payload["scheduler_d"] is not None:
        scheduler_d.load_state_dict(payload["scheduler_d"])
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    return payload
