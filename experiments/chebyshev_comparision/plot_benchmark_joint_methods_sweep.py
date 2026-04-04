import argparse
import json
import math
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


def load_payload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_latest_result(filename: str) -> Path:
    ensure_parent(DEFAULT_RESULTS_DIR / filename)
    candidates = sorted(DEFAULT_RESULTS_DIR.glob(filename.replace(".json", "*.json")))
    if not candidates:
        return DEFAULT_RESULTS_DIR / filename
    return max(candidates, key=lambda p: p.stat().st_mtime)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def case_title(case: dict) -> str:
    return f"N={case['N']}, T_u/T_d={case['tu_td_ratio']}"


def plot_raw_results(payload: dict, out_path: Path) -> None:
    cases = payload["cases"]
    if not cases:
        return

    cols = min(3, len(cases))
    rows = math.ceil(len(cases) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)

    for ax in axes.flat[len(cases):]:
        ax.axis("off")

    style = {
        "baseline": {"color": "tab:blue", "marker": "o"},
        "optimized": {"color": "tab:orange", "marker": "s"},
    }

    for ax, case in zip(axes.flat, cases):
        grouped = {"baseline": [], "optimized": []}
        for row in case["raw_results"]:
            grouped[row["strategy"]].append(row)

        for strategy in ["baseline", "optimized"]:
            rows_ = sorted(grouped[strategy], key=lambda r: r["stage2_steps"])
            xs = [r["train_seconds"] for r in rows_]
            ys = [r["final_loss"] for r in rows_]
            labels = [str(r["stage2_steps"]) for r in rows_]
            ax.scatter(
                xs,
                ys,
                label=strategy,
                color=style[strategy]["color"],
                marker=style[strategy]["marker"],
                s=55,
                alpha=0.9,
            )
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

        ax.axhline(
            case["target_loss"],
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label="target_loss",
        )
        ax.set_title(case_title(case))
        ax.set_xlabel("train_seconds")
        ax.set_ylabel("final_loss")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle("Joint Strategy Sweep: Raw Results", fontsize=14)
    fig.tight_layout()
    ensure_parent(out_path)
    out_path = unique_output_path(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_best_summary(payload: dict, out_path: Path) -> None:
    cases = payload["cases"]
    if not cases:
        return

    labels = []
    baseline_times = []
    optimized_times = []
    ratios = []

    for case in cases:
        labels.append(f"N={case['N']}\nr={case['tu_td_ratio']}")
        best_baseline = case.get("best_baseline")
        best_optimized = case.get("best_optimized")

        baseline_times.append(best_baseline["train_seconds"] if best_baseline else float("nan"))
        optimized_times.append(best_optimized["train_seconds"] if best_optimized else float("nan"))
        if best_baseline and best_optimized:
            ratios.append(best_baseline["train_seconds"] / best_optimized["train_seconds"])
        else:
            ratios.append(float("nan"))

    x = list(range(len(labels)))
    width = 0.36

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, 1.8 * len(labels)), 9))

    ax1.bar([i - width / 2 for i in x], baseline_times, width=width, label="baseline", color="tab:blue")
    ax1.bar([i + width / 2 for i in x], optimized_times, width=width, label="optimized", color="tab:orange")
    ax1.set_ylabel("best matched-loss train_seconds")
    ax1.set_title("Best Feasible Runtime at Matched Loss Tolerance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend()

    ax2.bar(x, ratios, color="tab:green")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax2.set_ylabel("time_ratio_baseline_to_optimized")
    ax2.set_title("Matched-Loss Time Ratio")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    ensure_parent(out_path)
    out_path = unique_output_path(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize benchmark_joint_methods_sweep_loss_target results."
    )
    parser.add_argument(
        "--input",
        default=str(resolve_latest_result("benchmark_joint_methods_sweep_loss_target_results.json")),
        help="Input JSON result file generated by benchmark_joint_methods_sweep_loss_target.py",
    )
    parser.add_argument(
        "--out-prefix",
        default=str(SCRIPT_DIR / "results" / "benchmark_joint_methods_sweep"),
        help="Prefix for output PNG files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_prefix = Path(args.out_prefix)
    payload = load_payload(input_path)

    raw_path = out_prefix.with_name(out_prefix.name + "_raw.png")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.png")

    raw_saved = plot_raw_results(payload, raw_path)
    summary_saved = plot_best_summary(payload, summary_path)

    print(f"saved: {raw_saved}")
    print(f"saved: {summary_saved}")


if __name__ == "__main__":
    main()
