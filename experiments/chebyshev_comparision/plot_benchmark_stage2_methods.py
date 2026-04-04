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


def plot_runtime_sections(payload: dict, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    forward = payload.get("forward_full", [])
    if forward:
        Ns = [row["N"] for row in forward]
        rk4 = [row["measurements"][0]["avg_forward_seconds"] for row in forward]
        split = [row["measurements"][1]["avg_forward_seconds"] for row in forward]
        axes[0].plot(Ns, rk4, marker="o", label="rk4")
        axes[0].plot(Ns, split, marker="s", label="split")
        axes[0].set_title("Full Forward Runtime")
        axes[0].set_xlabel("N")
        axes[0].set_ylabel("avg_forward_seconds")
        axes[0].grid(alpha=0.25)
        axes[0].legend()
    else:
        axes[0].axis("off")

    stage2 = payload.get("stage2_only", [])
    if stage2:
        Ns = [row["N"] for row in stage2]
        rk4 = [row["measurements"][0]["avg_stage2_seconds"] for row in stage2]
        split = [row["measurements"][1]["avg_stage2_seconds"] for row in stage2]
        axes[1].plot(Ns, rk4, marker="o", label="rk4")
        axes[1].plot(Ns, split, marker="s", label="split")
        axes[1].set_title("Stage-2 Only Runtime")
        axes[1].set_xlabel("N")
        axes[1].set_ylabel("avg_stage2_seconds")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
    else:
        axes[1].axis("off")

    training = payload.get("training")
    if training:
        methods = [row["method"] for row in training["measurements"]]
        times = [row["train_seconds"] for row in training["measurements"]]
        losses = [row["final_loss"] for row in training["measurements"]]
        x = range(len(methods))
        bars = axes[2].bar(x, times, color=["tab:blue", "tab:orange"])
        axes[2].set_title(f"Training Runtime (N={training['N']})")
        axes[2].set_xticks(list(x))
        axes[2].set_xticklabels(methods)
        axes[2].set_ylabel("train_seconds")
        axes[2].grid(axis="y", alpha=0.25)
        for bar, loss in zip(bars, losses):
            axes[2].annotate(
                f"loss={loss:.4f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8,
            )
    else:
        axes[2].axis("off")

    fig.suptitle("Stage-2 Methods Benchmark", fontsize=14)
    fig.tight_layout()
    ensure_parent(out_path)
    out_path = unique_output_path(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_error_summary(payload: dict, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    forward = payload.get("forward_full", [])
    if forward:
        Ns = [row["N"] for row in forward]
        prob = [row["max_prob_diff"] for row in forward]
        rho = [row["max_rho_diff"] for row in forward]
        axes[0].plot(Ns, prob, marker="o", color="tab:red")
        axes[0].set_title("Full Forward Max Prob Diff")
        axes[0].set_xlabel("N")
        axes[0].set_ylabel("max_prob_diff")
        axes[0].set_yscale("log")
        axes[0].grid(alpha=0.25)

        axes[1].plot(Ns, rho, marker="o", color="tab:purple")
        axes[1].set_title("Full Forward Max Rho Diff")
        axes[1].set_xlabel("N")
        axes[1].set_ylabel("max_rho_diff")
        axes[1].set_yscale("log")
        axes[1].grid(alpha=0.25)
    else:
        axes[0].axis("off")
        axes[1].axis("off")

    fig.tight_layout()
    ensure_parent(out_path)
    out_path = unique_output_path(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize benchmark_stage2_methods results.")
    parser.add_argument(
        "--input",
        default=str(resolve_latest_result("benchmark_stage2_methods_results.json")),
    )
    parser.add_argument(
        "--out-prefix",
        default=str(SCRIPT_DIR / "results" / "benchmark_stage2_methods"),
    )
    args = parser.parse_args()

    payload = load_payload(Path(args.input))
    prefix = Path(args.out_prefix)
    runtime_path = prefix.with_name(prefix.name + "_runtime.png")
    error_path = prefix.with_name(prefix.name + "_error.png")

    runtime_saved = plot_runtime_sections(payload, runtime_path)
    error_saved = plot_error_summary(payload, error_path)

    print(f"saved: {runtime_saved}")
    print(f"saved: {error_saved}")


if __name__ == "__main__":
    main()
