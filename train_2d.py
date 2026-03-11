# train_2d.py
import torch
import matplotlib.pyplot as plt
from models import QSNN2D
from data import make_circles, make_moons
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def plot_boundary(model, X, y, title=""):
    device = model.device
    # grid
    gx = torch.linspace(0, 1, 20)
    gy = torch.linspace(0, 1, 20)
    zz = torch.zeros((len(gy), len(gx)))
    with torch.no_grad():
        for iy, yy in enumerate(gy):
            for ix, xx in enumerate(gx):
                p, _ = model(torch.tensor([xx, yy], device=device))
                zz[iy, ix] = torch.argmax(p).cpu()
    plt.imshow(zz.numpy(), origin="lower", extent=(0,1,0,1), alpha=0.35, cmap="coolwarm")

    Xn = X.numpy()
    yn = y.numpy()
    plt.scatter(Xn[yn==0,0], Xn[yn==0,1], s=18, c="red", label="class0")
    plt.scatter(Xn[yn==1,0], Xn[yn==1,1], s=18, c="blue", label="class1")
    plt.xlim(0,1); plt.ylim(0,1)
    plt.title(title)
    plt.legend()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_gpu = False  # 想用CPU就 False
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # choose one:
    X, y = make_circles(n=100, noise=0.05, factor=0.5, seed=0)
    # X, y = make_moons(n=200, noise=0.08, seed=0)

    model = QSNN2D(N_in=12, T_u=1.0, T_d=1.0, init_h=0.1, init_g=0.1, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)

    Xd, yd = X.to(device), y.to(device)

    steps = 100
    losses = []
    for it in range(steps):
        opt.zero_grad()
        loss = 0.0
        for i in range(Xd.shape[0]):
            probs, _ = model(Xd[i])
            loss = loss + (1.0 - probs[yd[i]])
        loss = loss / Xd.shape[0]
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if (it+1) % 1 == 0:
            print(f"it {it+1:4d} loss={loss.item():.6f}")
    print("Training finished.")

    plt.figure(figsize=(11,4))
    plt.subplot(1,2,1)
    plt.plot(losses)
    plt.title("Training loss")
    plt.xlabel("iteration"); plt.ylabel("1-p(correct)")

    plt.subplot(1,2,2)
    plot_boundary(model, X, y, title="QSNN decision boundary")
    plt.tight_layout()
    plt.savefig("train_2d.png", dpi=150)

if __name__ == "__main__":
    main()