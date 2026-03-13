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


def plot_boundary(model, X, y, title=""):
    device = model.device
    gx = torch.linspace(0, 1, 20)
    gy = torch.linspace(0, 1, 20)
    zz = torch.zeros((len(gy), len(gx)))

    grid_points = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), dim=-1).reshape(-1, 2).to(device)
    with torch.no_grad():
        p, _ = model(grid_points)
        pred = torch.argmax(p, dim=1).cpu().reshape(len(gy), len(gx))
        zz[:, :] = pred

    plt.imshow(zz.numpy(), origin="lower", extent=(0, 1, 0, 1), alpha=0.35, cmap="coolwarm")

    Xn = X.numpy()
    yn = y.numpy()
    plt.scatter(Xn[yn == 0, 0], Xn[yn == 0, 1], s=18, c="red", label="class0")
    plt.scatter(Xn[yn == 1, 0], Xn[yn == 1, 1], s=18, c="blue", label="class1")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 固定总神经元 N=100 -> N_in=98
    N = 100
    N_in = N - 2

    X, y = make_circles(n=100, noise=0.05, factor=0.5, seed=0)
    Xd, yd = X.to(device), y.to(device)

    model = QSNN2D(
        N_in=N_in,
        T_u=1.0,
        T_d=1.0,
        init_h=0.1,
        init_g=0.1,
        device=device,
        stage2_steps=12,
    )
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)

    steps = 100
    losses = []
    for it in range(steps):
        opt.zero_grad()
        probs, _ = model(Xd)
        loss = (1.0 - probs[torch.arange(Xd.shape[0], device=device), yd]).mean()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if (it + 1) % 10 == 0:
            print(f"it {it+1:4d} loss={loss.item():.6f}")

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training loss")
    plt.xlabel("iteration")
    plt.ylabel("1-p(correct)")

    plt.subplot(1, 2, 2)
    plot_boundary(model, X, y, title="QSNN decision boundary")

    plt.tight_layout()
    out_path = "experiments/tu_td_sweeps/train_2d_n100.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
