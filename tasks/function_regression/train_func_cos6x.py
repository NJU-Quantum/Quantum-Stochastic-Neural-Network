# train_func_cos6x.py
import torch
import numpy as np
from models import QSNNFunction
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def target(x):
    return (1.0 + torch.cos(6.0 * x)) / 6.0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model = QSNNFunction(N_in=10, T=1.0, init_scale=0.05, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)

    M = 25
    x_train = torch.linspace(0.3, 1.3, M, device=device)
    y_train = target(x_train)

    losses = []
    steps = 1000  
    for it in range(steps):
        opt.zero_grad()
        yhat, _ = model(x_train)
        loss = torch.mean((yhat - y_train) ** 2)
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if (it + 1) % 200 == 0:
            print(f"it {it+1:5d}  loss={loss.item():.6e}")

    # plot
    with torch.no_grad():
        xs = torch.linspace(0.3, 1.3, 300, device=device)
        yh, _ = model(xs)
        yh = yh.cpu().numpy()
        yt = target(xs).cpu().numpy()

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.semilogy(losses)
    plt.title("Loss (MSE)")
    plt.xlabel("iteration")
    plt.ylabel("loss")

    plt.subplot(1,2,2)
    plt.plot(xs.cpu().numpy(), yt, "k--", label="target")
    plt.scatter(x_train.cpu().numpy(), y_train.cpu().numpy(), s=25, label="train")
    plt.plot(xs.cpu().numpy(), yh, "r", label="QSNN")
    plt.title("f(x)=(1+cos(6x))/6")
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_func_cos6x.png", dpi=150)

if __name__ == "__main__":
    main()