import os
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import make_circles
from models import QSNN2D

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def predict_grid(model, grid_n=120, chunk_size=512, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    gx = torch.linspace(0, 1, grid_n, device=device)
    gy = torch.linspace(0, 1, grid_n, device=device)

    grid_points = torch.stack(
        torch.meshgrid(gx, gy, indexing="xy"), dim=-1
    ).reshape(-1, 2)

    preds = []
    with torch.no_grad():
        for i in range(0, grid_points.shape[0], chunk_size):
            batch = grid_points[i:i + chunk_size]
            p, _ = model(batch)
            preds.append(torch.argmax(p, dim=1).cpu())

    pred = torch.cat(preds, dim=0).reshape(len(gy), len(gx))

    return pred


def train_one_n(
    N,
    X,
    y,
    device,
    steps=100,
    lr=3e-2,
    T_u=1.0,
    T_d=1.0,
    init_h=0.1,
    init_g=0.1,
    stage2_steps=12,
):
    # 固定总神经元 N，模型输入神经元 N_in = N - 2
    N_in = N - 2

    Xd, yd = X.to(device), y.to(device)

    model = QSNN2D(
        N_in=N_in,
        T_u=T_u,
        T_d=T_d,
        init_h=init_h,
        init_g=init_g,
        device=device,
        stage2_steps=stage2_steps,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for it in range(steps):
        opt.zero_grad()
        _, logits = model(Xd)
        loss = torch.nn.functional.cross_entropy(logits, yd)
        loss.backward()
        opt.step()

        losses.append(loss.item())

    return model, losses


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 待比较的总神经元规模
    N_list = [100, 200, 300, 400, 500]

    # 使用同一份数据，公平比较不同 N
    X, y = make_circles(n=100, noise=0.05, factor=0.5, seed=0)

    steps = 100
    lr = 3e-2

    results = {}

    for N in N_list:
        print(f"\n=== Training N={N} ===")
        model, losses = train_one_n(
            N=N,
            X=X,
            y=y,
            device=device,
            steps=steps,
            lr=lr,
            T_u=1.0,
            T_d=1.0,
            init_h=0.1,
            init_g=0.1,
            stage2_steps=12,
        )
        model.eval()
        model_cpu = model.to("cpu")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results[N] = {
            "model": model_cpu,
            "losses": losses,
            "final_loss": losses[-1],
        }
        print(f"N={N} final loss={losses[-1]:.6f}")

    # 2 行 5 列：第一行损失曲线，第二行决策边界
    fig, axes = plt.subplots(2, len(N_list), figsize=(4 * len(N_list), 8))

    Xn = X.numpy()
    yn = y.numpy()

    for col, N in enumerate(N_list):
        model = results[N]["model"]
        losses = results[N]["losses"]

        # 第一行：loss
        ax_loss = axes[0, col]
        ax_loss.plot(losses, lw=1.8)
        ax_loss.set_title(f"N={N} loss")
        ax_loss.set_xlabel("iteration")
        ax_loss.set_ylabel("cross entropy")
        ax_loss.grid(alpha=0.25)

        # 第二行：boundary
        ax_bd = axes[1, col]
        zz = predict_grid(model, grid_n=80, chunk_size=512, device=device)
        ax_bd.imshow(
            zz.numpy(),
            origin="lower",
            extent=(0, 1, 0, 1),
            alpha=0.35,
            cmap="coolwarm",
            aspect="auto",
        )
        ax_bd.scatter(Xn[yn == 0, 0], Xn[yn == 0, 1], s=14, c="red", label="class0")
        ax_bd.scatter(Xn[yn == 1, 0], Xn[yn == 1, 1], s=14, c="blue", label="class1")
        ax_bd.set_xlim(0, 1)
        ax_bd.set_ylim(0, 1)
        ax_bd.set_title(f"N={N} boundary")

        if col == len(N_list) - 1:
            ax_bd.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "train_2d_multiN_boundary.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nsaved: {out_path}")

    print("\nSummary (final loss):")
    for N in N_list:
        print(f"N={N:3d}  final_loss={results[N]['final_loss']:.6f}")


if __name__ == "__main__":
    main()
