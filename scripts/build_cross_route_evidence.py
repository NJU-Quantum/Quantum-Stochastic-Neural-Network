from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "hardware" / "wuyue_vqc" / "data" / "circle_processed.csv"
VQC = ROOT / "hardware" / "wuyue_vqc" / "outputs" / "vqc_cloud-simulator_report.json"
BAIHUA = ROOT / "hardware" / "wuyue_vqc" / "outputs" / "vqc_baihua_report.json"
PHOTONIC = ROOT / "hardware" / "kaiwu_qsnn_photonic" / "outputs" / "large_scale" / "large_qsnn_report.json"
OUT = ROOT / "docs" / "competition" / "evidence"
FIG = ROOT / "docs" / "competition" / "figures"


def load_dataset():
    rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
    raw = np.array([[float(row["x"]), float(row["y"])] for row in rows])
    processed = np.array([
        [float(row["x"]), float(row["y"]), float(row["radius_squared"]), float(row["xy"])]
        for row in rows
    ])
    labels = np.array([int(row["label"]) for row in rows])
    train = np.array([row["split"] == "train" for row in rows])
    return raw, processed, labels, train


def timed_fit(name, estimator, x_train, y_train, x_test, y_test):
    start = time.perf_counter()
    estimator.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    probability = estimator.predict_proba(x_test)[:, 1]
    predict_seconds = time.perf_counter() - start
    prediction = (probability >= 0.5).astype(int)
    return {
        "method": name,
        "train_seconds": train_seconds,
        "test_predict_seconds": predict_seconds,
        "test_accuracy": accuracy_score(y_test, prediction),
        "test_log_loss": log_loss(y_test, probability),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    raw, processed, labels, train = load_dataset()
    classical = [
        timed_fit(
            "LogisticRegression(x,y)",
            make_pipeline(StandardScaler(), LogisticRegression(random_state=23)),
            raw[train], labels[train], raw[~train], labels[~train],
        ),
        timed_fit(
            "LogisticRegression(x,y,r2,xy)",
            make_pipeline(StandardScaler(), LogisticRegression(random_state=23)),
            processed[train], labels[train], processed[~train], labels[~train],
        ),
        timed_fit(
            "RBF-SVM(x,y)",
            make_pipeline(
                StandardScaler(),
                CalibratedClassifierCV(SVC(kernel="rbf"), method="sigmoid", cv=3, ensemble=False),
            ),
            raw[train], labels[train], raw[~train], labels[~train],
        ),
    ]
    vqc = json.loads(VQC.read_text(encoding="utf-8"))
    baihua = json.loads(BAIHUA.read_text(encoding="utf-8"))
    photonic = json.loads(PHOTONIC.read_text(encoding="utf-8"))
    summary = {
        "dataset": {
            "samples": int(labels.size),
            "train_samples": int(train.sum()),
            "test_samples": int((~train).sum()),
            "seed": 23,
        },
        "classical": classical,
        "generic_quantum": {
            "algorithm": vqc["algorithm"],
            "train_accuracy": vqc["train"]["accuracy"],
            "test_accuracy": vqc["test"]["accuracy"],
            "training_seconds": vqc["training"]["train_seconds"],
            "parameters": len(vqc["training"]["params"]),
            "wuyue_backend_max_abs_error": vqc["wuyue_backend_verification"]["max_abs_error"],
            "cloud_simulator": vqc.get("cloud"),
            "baihua_submission": baihua["cloud"],
        },
        "photonic": {
            "spins": photonic["ising_shape"][0],
            "independent_couplings": photonic["independent_couplings"],
            "classical_reservoir_test_accuracy": photonic["classical_reservoir_test_accuracy"],
            "sa_accuracy_samples": photonic["sa"]["evaluated_samples"],
            "sa_accuracy": photonic["sa"]["accuracy"],
            "cim_task_id": photonic["cim"]["task_id"],
            "cim_prediction": photonic["cim"]["pred"],
            "expected_label": photonic["conditioned_label"],
            "cim_solutions": photonic["cim"]["solutions"],
            "cim_best_energy": photonic["cim"]["best_energy"],
            "sa_best_energy_same_sample": photonic["sa"]["cases"][0]["best_energy"],
        },
        "comparison_warning": "Photonic, VQC and classical entries do not share identical hardware budgets or sample coverage; no quantum advantage claim is made.",
    }
    (OUT / "cross_route_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "classical_baselines.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=classical[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(classical)

    colors = np.where(labels == 0, "#167D8D", "#D65A4A")
    fig, ax = plt.subplots(figsize=(6.8, 5.3))
    ax.scatter(raw[train, 0], raw[train, 1], c=colors[train], s=22, alpha=0.78, edgecolors="none")
    ax.scatter(raw[~train, 0], raw[~train, 1], c=colors[~train], s=48, marker="x", linewidths=1.2)
    ax.set(xlabel="x", ylabel="y", title="Shared inner/outer-ring binary dataset")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "dataset_distribution.png", dpi=180)
    plt.close(fig)

    labels_bar = ["LR raw", "LR + r2", "RBF-SVM", "WuYue VQC", "Photonic SA\n(8 samples)"]
    values = [row["test_accuracy"] for row in classical] + [vqc["test"]["accuracy"], photonic["sa"]["accuracy"]]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bars = ax.bar(labels_bar, values, color=("#687A8F", "#4B8F8C", "#D89B45", "#735DA5", "#C9503E"))
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title("Observed accuracy (coverage differs by route)")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "route_accuracy_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar(("SA", "CIM"), (photonic["sa"]["cases"][0]["best_energy"], photonic["cim"]["best_energy"]), color=("#687A8F", "#C9503E"))
    ax.set_ylabel("Best physical Hamiltonian (lower is better)")
    ax.set_title("Same 1000-spin conditioned sample")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "sa_cim_energy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.axis("off")
    lanes = (
        (0.80, "Coherent photonic", ("Data encoding", "1000-spin Ising", "Kaiwu SA gate", "Qboson CIM"), "#C9503E"),
        (0.50, "Universal gate", ("Angle encoding", "Trainable VQC", "WuYue simulator", "Baihua"), "#735DA5"),
        (0.20, "Classical", ("Feature matrix", "LR / RBF-SVM", "CPU evaluation", "Metrics"), "#4B8F8C"),
    )
    for y, label, steps, color in lanes:
        ax.text(0.01, y, label, va="center", fontsize=11, fontweight="bold", color=color)
        for i, step in enumerate(steps):
            x = 0.19 + i * 0.20
            ax.text(x, y, step, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=color, linewidth=1.5))
            if i < len(steps) - 1:
                ax.annotate("", xy=(x + 0.14, y), xytext=(x + 0.065, y),
                            arrowprops=dict(arrowstyle="->", color=color, linewidth=1.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Cross-technology validation workflow", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(FIG / "cross_route_workflow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
