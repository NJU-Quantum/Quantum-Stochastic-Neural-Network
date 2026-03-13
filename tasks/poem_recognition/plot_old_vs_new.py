import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: str) -> dict:
    cols = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.array(v, dtype=np.float64) for k, v in cols.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot old vs new comparison CSV.")
    parser.add_argument("--csv", required=True, help="Path to old_vs_new_comparison.csv")
    parser.add_argument("--out", required=True, help="Output png path")
    args = parser.parse_args()

    data = load_csv(args.csv)
    x = data["update"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    axes = axes.flatten()

    pairs = [
        ("loss", "old_loss_mean", "new_loss_mean"),
        ("test1", "old_test1_mean", "new_test1_mean"),
        ("test2", "old_test2_mean", "new_test2_mean"),
        ("test3", "old_test3_mean", "new_test3_mean"),
        ("test4", "old_test4_mean", "new_test4_mean"),
    ]

    for i, (title, old_k, new_k) in enumerate(pairs):
        ax = axes[i]
        ax.plot(x, data[old_k], color="dimgray", linewidth=2.2, label="old")
        ax.plot(x, data[new_k], color="royalblue", linewidth=2.2, label="new (QSNNText)")
        ax.set_title(title)
        ax.set_xlabel("update")
        if title == "loss":
            ax.set_ylabel("mean loss")
        else:
            ax.set_ylabel("mean score")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    # Hide the extra subplot slot.
    axes[-1].axis("off")

    fig.suptitle("Old vs QSNNText Retest Comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(args.out)


if __name__ == "__main__":
    main()
