import time
import traceback
import torch

from models import QSNNFunction, QSNN2D
from data import make_circles


def try_qsnn_function(n_in: int, device: str, batch: int = 16):
    model = QSNNFunction(N_in=n_in, T=1.0, init_scale=0.05, device=device)
    x = torch.linspace(0.3, 1.3, batch, device=device)
    target = torch.zeros(batch, device=device)

    t0 = time.perf_counter()
    yhat, _ = model(x)
    loss = torch.mean((yhat - target) ** 2)
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def try_qsnn_2d(n_in: int, device: str, batch: int = 32):
    model = QSNN2D(N_in=n_in, T_u=1.0, T_d=1.0, init_h=0.1, init_g=0.1, device=device)
    X, y = make_circles(n=batch, noise=0.05, factor=0.5, seed=0)
    X = X.to(device)
    y = y.to(device)

    t0 = time.perf_counter()
    probs, _ = model(X)
    loss = (1.0 - probs[torch.arange(X.shape[0], device=device), y]).mean()
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def sweep(name, try_fn, start, step, max_n_in, per_trial_seconds, device):
    best = None
    stop_reason = "reached_limit"
    n = start
    while n <= max_n_in:
        try:
            dt = try_fn(n, device)
            print(f"[{name}] N_in={n:3d} ok  time={dt:.3f}s")
            if dt > per_trial_seconds:
                stop_reason = f"too_slow(>{per_trial_seconds}s)"
                break
            best = (n, dt)
            n += step
        except Exception as e:
            print(f"[{name}] N_in={n:3d} fail: {type(e).__name__}: {e}")
            stop_reason = "exception"
            break

    return best, stop_reason, n


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"device={device}")

    func_best, func_reason, func_stop_n = sweep(
        "QSNNFunction",
        try_qsnn_function,
        start=8,
        step=4,
        max_n_in=200,
        per_trial_seconds=15.0,
        device=device,
    )

    cls_best, cls_reason, cls_stop_n = sweep(
        "QSNN2D",
        try_qsnn_2d,
        start=8,
        step=2,
        max_n_in=120,
        per_trial_seconds=15.0,
        device=device,
    )

    print("\n=== SUMMARY ===")
    if func_best is None:
        print(f"QSNNFunction: no successful dimension; stopped at N_in={func_stop_n}, reason={func_reason}")
    else:
        n_in, dt = func_best
        print(
            f"QSNNFunction: max practical N_in={n_in} (N={n_in+1}), last_ok_time={dt:.3f}s, "
            f"stop_at_N_in={func_stop_n}, reason={func_reason}"
        )

    if cls_best is None:
        print(f"QSNN2D: no successful dimension; stopped at N_in={cls_stop_n}, reason={cls_reason}")
    else:
        n_in, dt = cls_best
        print(
            f"QSNN2D: max practical N_in={n_in} (N={n_in+2}), last_ok_time={dt:.3f}s, "
            f"stop_at_N_in={cls_stop_n}, reason={cls_reason}"
        )


if __name__ == "__main__":
    main()
