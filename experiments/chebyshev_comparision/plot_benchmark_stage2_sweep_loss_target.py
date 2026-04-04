import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


def unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_payload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_latest_result(filename: str) -> Path:
    ensure_parent(DEFAULT_RESULTS_DIR / filename)
    candidates = sorted(DEFAULT_RESULTS_DIR.glob(filename.replace(".json", "*.json")))
    if not candidates:
        return DEFAULT_RESULTS_DIR / filename
    return max(candidates, key=lambda p: p.stat().st_mtime)


def plot_raw(payload: dict, out_path: Path) -> Path:
    rows = payload["raw_results"]
    grouped = {"rk4": [], "split": []}
    for row in rows:
        grouped[row["stage2_method"]].append(row)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {"rk4": "tab:blue", "split": "tab:orange"}
    markers = {"rk4": "o", "split": "s"}

    for method in ["rk4", "split"]:
        rows_ = sorted(grouped[method], key=lambda r: r["stage2_steps"])
        xs = [r["train_seconds"] for r in rows_]
        ys = [r["final_loss"] for r in rows_]
        labels = [str(r["stage2_steps"]) for r in rows_]
        ax.scatter(xs, ys, color=colors[method], marker=markers[method], s=60, label=method)
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.axhline(payload["target_loss"], linestyle="--", color="gray", linewidth=1.0, label="target_loss")
    ax.set_title("Stage-2 Sweep Raw Results")
    ax.set_xlabel("train_seconds")
    ax.set_ylabel("final_loss")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    ensure_parent(out_path)
    out_path = unique_output_path(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_summary(payload: dict, out_path: Path) -> Path:
    best_rk4 = payload.get("best_rk4")
    best_split = payload.get("best_split")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    methods = ["rk4", "split"]
    times = [
        best_rk4["train_seconds"] if best_rk4 else float("nan"),
        best_split["train_seconds"] if best_split else float("nan"),
    ]
    losses = [
        best_rk4["final_loss"] if best_rk4 else float("nan"),
        best_split["final_loss"] if best_split else float("nan"),
    ]
    steps = [
        best_rk4["stage2_steps"] if best_rk4 else None,
        best_split["stage2_steps"] if best_split else None,
    ]

    bars = axes[0].bar(methods, times, color=["tab:blue", "tab:orange"])
    axes[0].set_title("Best Feasible Runtime")
    axes[0].set_ylabel("train_seconds")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, loss, step in zip(bars, losses, steps):
        axes[0].annotate(
            f"loss={loss:.4f}\nsteps={step}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
        )

    if best_rk4 and best_split:
        ratio = best_rk4["train_seconds"] / best_split["train_seconds"]
        axes[1].bar(["rk4/split"], [ratio], color="tab:green")
        axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        axes[1].set_title("Matched-Loss Time Ratio")
        axes[1].set_ylabel("time_ratio_rk4_to_split")
        axes[1].grid(axis="y", alpha=0.25)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    ensure_parent(out_path)
    out_path = unique_output_path(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize benchmark_stage2_sweep_loss_target results.")
    parser.add_argument(
        "--input",
        default=str(resolve_latest_result("benchmark_stage2_sweep_loss_target_results.json")),
    )
    parser.add_argument(
        "--out-prefix",
        default=str(SCRIPT_DIR / "results" / "benchmark_stage2_sweep_loss_target"),
    )
    args = parser.parse_args()

    payload = load_payload(Path(args.input))
    prefix = Path(args.out_prefix)
    raw_path = prefix.with_name(prefix.name + "_raw.png")
    summary_path = prefix.with_name(prefix.name + "_summary.png")

    raw_saved = plot_raw(payload, raw_path)
    summary_saved = plot_summary(payload, summary_path)

    print(f"saved: {raw_saved}")
    print(f"saved: {summary_saved}")


if __name__ == "__main__":
    main()
