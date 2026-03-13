import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import make_circles
from models import QSNN2D


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_from_probs(probs: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(probs, dim=1)
    return (pred == y).float().mean().item()


def run_one_ratio(
    ratio_u: float,
    total_time: float,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str,
    n_in: int,
    epochs: int,
    lr: float,
    stage2_steps: int,
):
    t_u = total_time * ratio_u
    t_d = total_time - t_u

    model = QSNN2D(
        N_in=n_in,
        T_u=t_u,
        T_d=t_d,
        init_h=0.1,
        init_g=0.1,
        device=device,
        stage2_steps=stage2_steps,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        probs, _ = model(x_train)
        loss = (1.0 - probs[torch.arange(x_train.shape[0], device=device), y_train]).mean()
        loss.backward()
        opt.step()

        with torch.no_grad():
            model.eval()
            p_train, _ = model(x_train)
            train_loss = (1.0 - p_train[torch.arange(x_train.shape[0], device=device), y_train]).mean().item()
            train_acc = accuracy_from_probs(p_train, y_train)

            p_test, _ = model(x_test)
            test_loss = (1.0 - p_test[torch.arange(x_test.shape[0], device=device), y_test]).mean().item()
            test_acc = accuracy_from_probs(p_test, y_test)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

    best_test_acc = float(max(history["test_acc"]))
    best_epoch = int(np.argmax(history["test_acc"])) + 1

    return {
        "ratio_u": ratio_u,
        "ratio_d": 1.0 - ratio_u,
        "T_u": t_u,
        "T_d": t_d,
        "final_train_loss": history["train_loss"][-1],
        "final_test_loss": history["test_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
    }


def main():
    set_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_time = 2.0
    ratios = [i / 10 for i in range(11)]  # [0,1] endpoints included

    # 用户要求的组合
    Ns = [20, 40, 60]
    stage2_steps_list = [12, 20]

    # 统一训练配置
    epochs = 20
    lr = 3e-2

    # 固定数据以保证跨配置可比性
    X, y = make_circles(n=240, noise=0.05, factor=0.5, seed=0)
    perm = torch.randperm(X.shape[0])
    X = X[perm]
    y = y[perm]

    split = int(0.7 * X.shape[0])
    x_train, y_train = X[:split].to(device), y[:split].to(device)
    x_test, y_test = X[split:].to(device), y[split:].to(device)

    print(f"device={device}, total_time={total_time}, epochs={epochs}, ratios={ratios}")

    all_rows = []

    t0 = time.perf_counter()
    for N in Ns:
        n_in = N - 2
        for s2 in stage2_steps_list:
            print(f"\n=== Sweep N={N}, N_in={n_in}, stage2_steps={s2} ===")
            cfg_t0 = time.perf_counter()
            for r in ratios:
                r0 = time.perf_counter()
                res = run_one_ratio(
                    ratio_u=r,
                    total_time=total_time,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    device=device,
                    n_in=n_in,
                    epochs=epochs,
                    lr=lr,
                    stage2_steps=s2,
                )
                dt = time.perf_counter() - r0
                all_rows.append(
                    {
                        "N": N,
                        "N_in": n_in,
                        "stage2_steps": s2,
                        **res,
                    }
                )
                print(
                    f"N={N}, s2={s2}, ratio_u={r:.1f}, T_u={res['T_u']:.2f}, T_d={res['T_d']:.2f}, "
                    f"final_test_acc={res['final_test_acc']:.4f}, final_test_loss={res['final_test_loss']:.4f}, "
                    f"best_test_acc={res['best_test_acc']:.4f}, time={dt:.1f}s"
                )
            print(f"cfg done in {time.perf_counter() - cfg_t0:.1f}s")

    total_dt = time.perf_counter() - t0
    print(f"\nall sweeps done in {total_dt:.1f}s")

    out_csv = Path("tu_td_grid_N20_40_60_s2_12_20.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "N",
                "N_in",
                "stage2_steps",
                "ratio_u",
                "ratio_d",
                "T_u",
                "T_d",
                "final_train_loss",
                "final_test_loss",
                "final_train_acc",
                "final_test_acc",
                "best_test_acc",
                "best_epoch",
            ]
        )
        for row in all_rows:
            writer.writerow(
                [
                    row["N"],
                    row["N_in"],
                    row["stage2_steps"],
                    row["ratio_u"],
                    row["ratio_d"],
                    row["T_u"],
                    row["T_d"],
                    row["final_train_loss"],
                    row["final_test_loss"],
                    row["final_train_acc"],
                    row["final_test_acc"],
                    row["best_test_acc"],
                    row["best_epoch"],
                ]
            )

    # 绘制 2x3 综合图：每个配置一张（acc/loss vs ratio）
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)

    cfgs = [(20, 12), (40, 12), (60, 12), (20, 20), (40, 20), (60, 20)]
    for idx, (N, s2) in enumerate(cfgs):
        ax = axes[idx // 3, idx % 3]
        rows = [r for r in all_rows if r["N"] == N and r["stage2_steps"] == s2]
        rows = sorted(rows, key=lambda x: x["ratio_u"])

        ru = [r["ratio_u"] for r in rows]
        acc = [r["final_test_acc"] for r in rows]
        loss = [r["final_test_loss"] for r in rows]

        l1 = ax.plot(ru, acc, marker="o", label="Final Test Acc", color="tab:blue")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"N={N}, stage2_steps={s2}")
        ax.grid(alpha=0.25)

        ax2 = ax.twinx()
        l2 = ax2.plot(ru, loss, marker="s", linestyle="--", label="Final Test Loss", color="tab:red")

        if idx // 3 == 1:
            ax.set_xlabel("T_u / (T_u + T_d)")
        if idx % 3 == 0:
            ax.set_ylabel("Accuracy")

    # 统一图例
    lines = [
        plt.Line2D([0], [0], color="tab:blue", marker="o", label="Final Test Acc"),
        plt.Line2D([0], [0], color="tab:red", marker="s", linestyle="--", label="Final Test Loss"),
    ]
    fig.legend(handles=lines, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_fig = Path("tu_td_grid_N20_40_60_s2_12_20.png")
    fig.savefig(out_fig, dpi=170)

    # 输出每个配置最优点
    print("\n=== BEST PER CONFIG (by final_test_acc) ===")
    for N, s2 in cfgs:
        rows = [r for r in all_rows if r["N"] == N and r["stage2_steps"] == s2]
        best = max(rows, key=lambda x: x["final_test_acc"])
        print(
            f"N={N}, s2={s2}: best ratio_u={best['ratio_u']:.1f}, ratio_d={best['ratio_d']:.1f}, "
            f"T_u={best['T_u']:.2f}, T_d={best['T_d']:.2f}, final_test_acc={best['final_test_acc']:.4f}, "
            f"final_test_loss={best['final_test_loss']:.4f}"
        )

    print(f"saved: {out_csv}, {out_fig}")


if __name__ == "__main__":
    main()
