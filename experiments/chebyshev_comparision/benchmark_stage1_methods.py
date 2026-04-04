import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import make_circles
from models import QSNN2D
import qsw


def unique_output_path(path_str: str) -> Path:
    path = Path(path_str)
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


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def clone_weights(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    dst.load_state_dict(src.state_dict())


def build_model(
    N: int,
    device: torch.device,
    stage1_method: str,
    stage2_steps: int,
    stage1_suzuki_steps: int,
    stage1_suzuki_order: int,
    chebyshev_order: int,
    chebyshev_tol: float,
) -> QSNN2D:
    return QSNN2D(
        N_in=N - 2,
        T_u=1.0,
        T_d=1.0,
        init_h=0.1,
        init_g=0.1,
        device=device,
        stage2_steps=stage2_steps,
        stage1_method=stage1_method,
        stage1_suzuki_steps=stage1_suzuki_steps,
        stage1_suzuki_order=stage1_suzuki_order,
        chebyshev_order=chebyshev_order,
        chebyshev_tol=chebyshev_tol,
    )


def benchmark_forward_full(
    N: int,
    batch_size: int,
    warmup: int,
    runs: int,
    device: torch.device,
    stage2_steps: int,
    stage1_suzuki_steps: int,
    stage1_suzuki_order: int,
    chebyshev_order: int,
    chebyshev_tol: float,
) -> dict:
    xy = torch.rand(batch_size, 2, device=device)
    torch.manual_seed(0)

    exact = build_model(
        N, device, "exact", stage2_steps, stage1_suzuki_steps, stage1_suzuki_order, chebyshev_order, chebyshev_tol
    )
    cheb = build_model(
        N, device, "chebyshev", stage2_steps, stage1_suzuki_steps, stage1_suzuki_order, chebyshev_order, chebyshev_tol
    )
    suzuki = build_model(
        N, device, "suzuki", stage2_steps, stage1_suzuki_steps, stage1_suzuki_order, chebyshev_order, chebyshev_tol
    )
    clone_weights(cheb, exact)
    clone_weights(suzuki, exact)

    measurements = []
    outputs = {}
    for method, model in [("exact", exact), ("chebyshev", cheb), ("suzuki", suzuki)]:
        model.eval()
        with torch.no_grad():
            for _ in range(warmup):
                probs, rho = model(xy)
            sync_if_needed(device)
            t0 = time.perf_counter()
            for _ in range(runs):
                probs, rho = model(xy)
            sync_if_needed(device)
            elapsed = time.perf_counter() - t0
        label = "exact_state" if method == "exact" else method
        measurements.append(
            {
                "method": label,
                "avg_forward_seconds": elapsed / runs,
            }
        )
        outputs[method] = (probs, rho)

    prob_diffs = {
        "chebyshev": (outputs["exact"][0] - outputs["chebyshev"][0]).abs().max().item(),
        "suzuki": (outputs["exact"][0] - outputs["suzuki"][0]).abs().max().item(),
    }
    rho_diffs = {
        "chebyshev": (outputs["exact"][1] - outputs["chebyshev"][1]).abs().max().item(),
        "suzuki": (outputs["exact"][1] - outputs["suzuki"][1]).abs().max().item(),
    }

    exact_t = measurements[0]["avg_forward_seconds"]
    cheb_t = measurements[1]["avg_forward_seconds"]
    suzuki_t = measurements[2]["avg_forward_seconds"]
    return {
        "N": N,
        "batch_size": batch_size,
        "measurements": measurements,
        "time_ratio_exact_to_chebyshev": exact_t / cheb_t,
        "time_ratio_exact_to_suzuki": exact_t / suzuki_t,
        "max_prob_diff_to_exact": prob_diffs,
        "max_rho_diff_to_exact": rho_diffs,
    }


def benchmark_stage1_only(
    N: int,
    batch_size: int,
    warmup: int,
    runs: int,
    device: torch.device,
    stage1_suzuki_steps: int,
    stage1_suzuki_order: int,
    chebyshev_order: int,
    chebyshev_tol: float,
) -> dict:
    xy = torch.rand(batch_size, 2, device=device)
    torch.manual_seed(0)
    model = build_model(
        N, device, "exact", stage2_steps=12, stage1_suzuki_steps=stage1_suzuki_steps, stage1_suzuki_order=stage1_suzuki_order, chebyshev_order=chebyshev_order, chebyshev_tol=chebyshev_tol
    )

    x, y = xy[:, 0], xy[:, 1]
    psi0 = model.encode_state(x, y)
    Hf = model.Hu_raw.to(torch.complex64)
    Hu = 0.5 * (Hf + Hf.mH)
    H = torch.zeros((model.N, model.N), device=device, dtype=torch.complex64)
    H[: model.N_in, : model.N_in] = Hu

    def run_exact():
        U = torch.matrix_exp((-1j) * H * model.T_u)
        psi_u = U.unsqueeze(0) @ psi0
        return psi_u @ psi_u.mH

    def run_chebyshev():
        psi_u = qsw.evolve_state_chebyshev(
            psi0, H, model.T_u, max_order=chebyshev_order, tol=chebyshev_tol
        )
        return psi_u @ psi_u.mH

    def run_suzuki():
        psi_u = qsw.evolve_state_suzuki(
            psi0, H, model.T_u, steps=stage1_suzuki_steps, order=stage1_suzuki_order
        )
        return psi_u @ psi_u.mH

    measurements = []
    outputs = {}
    for method, fn in [("exact", run_exact), ("chebyshev", run_chebyshev), ("suzuki", run_suzuki)]:
        with torch.no_grad():
            for _ in range(warmup):
                rho = fn()
            sync_if_needed(device)
            t0 = time.perf_counter()
            for _ in range(runs):
                rho = fn()
            sync_if_needed(device)
            elapsed = time.perf_counter() - t0
        label = "exact_state" if method == "exact" else method
        measurements.append(
            {
                "method": label,
                "avg_stage1_seconds": elapsed / runs,
            }
        )
        outputs[method] = rho

    rho_diffs = {
        "chebyshev": (outputs["exact"] - outputs["chebyshev"]).abs().max().item(),
        "suzuki": (outputs["exact"] - outputs["suzuki"]).abs().max().item(),
    }
    exact_t = measurements[0]["avg_stage1_seconds"]
    cheb_t = measurements[1]["avg_stage1_seconds"]
    suzuki_t = measurements[2]["avg_stage1_seconds"]
    return {
        "N": N,
        "batch_size": batch_size,
        "measurements": measurements,
        "time_ratio_exact_to_chebyshev": exact_t / cheb_t,
        "time_ratio_exact_to_suzuki": exact_t / suzuki_t,
        "max_rho_diff_to_exact": rho_diffs,
    }


def benchmark_training(
    N: int,
    steps: int,
    device: torch.device,
    stage2_steps: int,
    stage1_suzuki_steps: int,
    stage1_suzuki_order: int,
    chebyshev_order: int,
    chebyshev_tol: float,
) -> dict:
    X, y = make_circles(n=100, noise=0.05, factor=0.5, seed=0)
    Xd, yd = X.to(device), y.to(device)

    results = []
    for method in ["exact", "chebyshev", "suzuki"]:
        torch.manual_seed(0)
        model = build_model(
            N, device, method, stage2_steps, stage1_suzuki_steps, stage1_suzuki_order, chebyshev_order, chebyshev_tol
        )
        opt = torch.optim.Adam(model.parameters(), lr=3e-2)

        sync_if_needed(device)
        t0 = time.perf_counter()
        last_loss = None
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            probs, _ = model(Xd)
            loss = torch.nn.functional.nll_loss(torch.log(probs), yd)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
        sync_if_needed(device)
        elapsed = time.perf_counter() - t0

        label = "exact_state" if method == "exact" else method
        results.append(
            {
                "method": label,
                "train_seconds": elapsed,
                "final_loss": last_loss,
            }
        )

    return {
        "N": N,
        "steps": steps,
        "measurements": results,
        "time_ratio_exact_to_chebyshev": (
            results[0]["train_seconds"] / results[1]["train_seconds"]
        ),
        "time_ratio_exact_to_suzuki": (
            results[0]["train_seconds"] / results[2]["train_seconds"]
        ),
    }


def print_section(title: str, rows: List[Dict], time_key: str) -> None:
    print(f"\n=== {title} ===")
    for row in rows:
        print(f"N={row['N']}")
        for m in row["measurements"]:
            print(f"  {m['method']:10s} {time_key}={m[time_key]:.6f}")
        print(
            f"  time_ratio_exact_to_chebyshev="
            f"{row['time_ratio_exact_to_chebyshev']:.3f}"
        )
        if "time_ratio_exact_to_suzuki" in row:
            print(
                f"  time_ratio_exact_to_suzuki="
                f"{row['time_ratio_exact_to_suzuki']:.3f}"
            )
        if "max_prob_diff_to_exact" in row:
            print(
                "  max_prob_diff_to_exact="
                f"{row['max_prob_diff_to_exact']}"
            )
            print(
                "  max_rho_diff_to_exact="
                f"{row['max_rho_diff_to_exact']}"
            )
        elif "max_rho_diff_to_exact" in row:
            print(f"  max_rho_diff_to_exact={row['max_rho_diff_to_exact']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark QSNN2D stage-1 exact_state vs chebyshev vs suzuki."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--forward-ns", default="100,200")
    parser.add_argument("--stage1-ns", default="100,200,300")
    parser.add_argument("--train-n", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--forward-runs", type=int, default=10)
    parser.add_argument("--stage1-runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--stage2-steps", type=int, default=12)
    parser.add_argument("--stage1-suzuki-steps", type=int, default=12)
    parser.add_argument("--stage1-suzuki-order", type=int, default=2)
    parser.add_argument("--chebyshev-order", type=int, default=128)
    parser.add_argument("--chebyshev-tol", type=float, default=1e-10)
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default=str(SCRIPT_DIR / "results" / "benchmark_stage1_methods_results.json"),
    )
    args = parser.parse_args()

    device = choose_device(args.device)
    payload = {
        "device": str(device),
        "config": {
            "forward_ns": parse_int_list(args.forward_ns),
            "stage1_ns": parse_int_list(args.stage1_ns),
            "train_n": args.train_n,
            "batch_size": args.batch_size,
            "forward_runs": args.forward_runs,
            "stage1_runs": args.stage1_runs,
            "warmup": args.warmup,
            "train_steps": args.train_steps,
            "stage2_steps": args.stage2_steps,
            "stage1_suzuki_steps": args.stage1_suzuki_steps,
            "stage1_suzuki_order": args.stage1_suzuki_order,
            "chebyshev_order": args.chebyshev_order,
            "chebyshev_tol": args.chebyshev_tol,
        },
        "forward_full": [],
        "stage1_only": [],
        "training": None,
    }

    print(f"device: {device}")

    if not args.skip_forward:
        for N in parse_int_list(args.forward_ns):
            payload["forward_full"].append(
                benchmark_forward_full(
                    N=N,
                    batch_size=args.batch_size,
                    warmup=args.warmup,
                    runs=args.forward_runs,
                    device=device,
                    stage2_steps=args.stage2_steps,
                    stage1_suzuki_steps=args.stage1_suzuki_steps,
                    stage1_suzuki_order=args.stage1_suzuki_order,
                    chebyshev_order=args.chebyshev_order,
                    chebyshev_tol=args.chebyshev_tol,
                )
            )
        print_section(
            "Forward Benchmark (Full Model)",
            payload["forward_full"],
            "avg_forward_seconds",
        )

    if not args.skip_stage1:
        for N in parse_int_list(args.stage1_ns):
            payload["stage1_only"].append(
                benchmark_stage1_only(
                    N=N,
                    batch_size=args.batch_size,
                    warmup=args.warmup,
                    runs=args.stage1_runs,
                    device=device,
                    stage1_suzuki_steps=args.stage1_suzuki_steps,
                    stage1_suzuki_order=args.stage1_suzuki_order,
                    chebyshev_order=args.chebyshev_order,
                    chebyshev_tol=args.chebyshev_tol,
                )
            )
        print_section(
            "Stage-1 Only Benchmark",
            payload["stage1_only"],
            "avg_stage1_seconds",
        )

    if not args.skip_train:
        payload["training"] = benchmark_training(
            N=args.train_n,
            steps=args.train_steps,
            device=device,
            stage2_steps=args.stage2_steps,
            stage1_suzuki_steps=args.stage1_suzuki_steps,
            stage1_suzuki_order=args.stage1_suzuki_order,
            chebyshev_order=args.chebyshev_order,
            chebyshev_tol=args.chebyshev_tol,
        )
        print("\n=== Training Benchmark ===")
        print(
            f"N={payload['training']['N']} "
            f"steps={payload['training']['steps']}"
        )
        for item in payload["training"]["measurements"]:
            print(
                f"  {item['method']:10s} "
                f"train_seconds={item['train_seconds']:.6f} "
                f"final_loss={item['final_loss']:.6f}"
            )
        print(
            "  time_ratio_exact_to_chebyshev="
            f"{payload['training']['time_ratio_exact_to_chebyshev']:.3f}"
        )
        print(
            "  time_ratio_exact_to_suzuki="
            f"{payload['training']['time_ratio_exact_to_suzuki']:.3f}"
        )

    out_path = unique_output_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
