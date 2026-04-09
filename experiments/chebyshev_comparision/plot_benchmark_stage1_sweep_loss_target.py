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


def method_label(row: dict) -> str:
    method = row["stage1_method"]
    if method == "exact_state":
        return "exact_state"
    if method == "chebyshev":
        return f"chebyshev ({row['chebyshev_order']})"
    return f"suzuki ({row['stage1_suzuki_steps']})"


def plot_raw(payload: dict, out_path: Path) -> Path:
    rows = payload["raw_results"]
    grouped = {"exact_state": [], "chebyshev": [], "suzuki": []}
    for row in rows:
        grouped[row["stage1_method"]].append(row)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {
        "exact_state": "tab:blue",
        "chebyshev": "tab:orange",
        "suzuki": "tab:green",
    }
    markers = {
        "exact_state": "o",
        "chebyshev": "s",
        "suzuki": "^",
    }

    for method in ["exact_state", "chebyshev", "suzuki"]:
        rows_ = grouped[method]
        if not rows_:
            continue
        if method == "chebyshev":
            rows_ = sorted(rows_, key=lambda r: r["chebyshev_order"])
        elif method == "suzuki":
            rows_ = sorted(rows_, key=lambda r: r["stage1_suzuki_steps"])

        xs = [r["train_seconds"] for r in rows_]
        ys = [r["final_loss"] for r in rows_]
        if method == "exact_state":
            labels = ["baseline" for _ in rows_]
        elif method == "chebyshev":
            labels = [str(r["chebyshev_order"]) for r in rows_]
        else:
            labels = [str(r["stage1_suzuki_steps"]) for r in rows_]

        ax.scatter(xs, ys, color=colors[method], marker=markers[method], s=60, label=method)
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.axhline(payload["target_loss"], linestyle="--", color="gray", linewidth=1.0, label="target_loss")
    ax.set_title("Stage-1 Sweep Raw Results")
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
    best_exact = payload.get("best_exact_state")
    best_chebyshev = payload.get("best_chebyshev")
    best_suzuki = payload.get("best_suzuki")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    methods = ["exact_state", "chebyshev", "suzuki"]
    rows = [best_exact, best_chebyshev, best_suzuki]
    times = [row["train_seconds"] if row else float("nan") for row in rows]
    losses = [row["final_loss"] if row else float("nan") for row in rows]
    labels = [method_label(row) if row else "none" for row in rows]

    bars = axes[0].bar(methods, times, color=["tab:blue", "tab:orange", "tab:green"])
    axes[0].set_title("Best Feasible Runtime")
    axes[0].set_ylabel("train_seconds")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, loss, label in zip(bars, losses, labels):
        axes[0].annotate(
            f"loss={loss:.4f}\n{label}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
        )

    ratio_names = []
    ratio_values = []
    if best_exact and best_chebyshev:
        ratio_names.append("exact/cheb")
        ratio_values.append(best_exact["train_seconds"] / best_chebyshev["train_seconds"])
    if best_exact and best_suzuki:
        ratio_names.append("exact/suzuki")
        ratio_values.append(best_exact["train_seconds"] / best_suzuki["train_seconds"])

    if ratio_values:
        axes[1].bar(ratio_names, ratio_values, color=["tab:purple", "tab:red"][: len(ratio_values)])
        axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        axes[1].set_title("Matched-Loss Time Ratios")
        axes[1].set_ylabel("time_ratio_exact_to_method")
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
    parser = argparse.ArgumentParser(description="Visualize benchmark_stage1_sweep_loss_target results.")
    parser.add_argument(
        "--input",
        default=str(resolve_latest_result("benchmark_stage1_sweep_loss_target_results.json")),
    )
    parser.add_argument(
        "--out-prefix",
        default=str(SCRIPT_DIR / "results" / "benchmark_stage1_sweep_loss_target"),
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
