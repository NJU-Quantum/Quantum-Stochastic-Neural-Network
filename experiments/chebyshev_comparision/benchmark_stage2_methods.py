import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import make_circles
from models import QSNN2D


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
    stage2_method: str,
    stage2_steps: int,
) -> QSNN2D:
    return QSNN2D(
        N_in=N - 2,
        T_u=1.0,
        T_d=1.0,
        init_h=0.1,
        init_g=0.1,
        device=device,
        stage1_method="exact",
        stage2_method=stage2_method,
        stage2_steps=stage2_steps,
    )


def benchmark_forward_full(
    N: int,
    batch_size: int,
    warmup: int,
    runs: int,
    device: torch.device,
    stage2_steps: int,
) -> Dict:
    xy = torch.rand(batch_size, 2, device=device)
    torch.manual_seed(0)

    rk4 = build_model(N, device, "rk4", stage2_steps)
    split = build_model(N, device, "split", stage2_steps)
    clone_weights(split, rk4)

    measurements = []
    outputs = {}
    for method, model in [("rk4", rk4), ("split", split)]:
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
        measurements.append(
            {
                "method": method,
                "avg_forward_seconds": elapsed / runs,
            }
        )
        outputs[method] = (probs, rho)

    prob_diff = (outputs["rk4"][0] - outputs["split"][0]).abs().max().item()
    rho_diff = (outputs["rk4"][1] - outputs["split"][1]).abs().max().item()
    rk4_t = measurements[0]["avg_forward_seconds"]
    split_t = measurements[1]["avg_forward_seconds"]
    return {
        "N": N,
        "batch_size": batch_size,
        "measurements": measurements,
        "time_ratio_rk4_to_split": rk4_t / split_t,
        "max_prob_diff": prob_diff,
        "max_rho_diff": rho_diff,
    }


def benchmark_stage2_only(
    N: int,
    batch_size: int,
    warmup: int,
    runs: int,
    device: torch.device,
    stage2_steps: int,
) -> Dict:
    xy = torch.rand(batch_size, 2, device=device)
    torch.manual_seed(0)

    rk4_model = build_model(N, device, "rk4", stage2_steps)
    split_model = build_model(N, device, "split", stage2_steps)
    clone_weights(split_model, rk4_model)

    x, y = xy[:, 0], xy[:, 1]
    psi0 = rk4_model.encode_state(x, y)
    Hf = rk4_model.Hu_raw.to(torch.complex64)
    Hu = 0.5 * (Hf + Hf.mH)
    H = torch.zeros((rk4_model.N, rk4_model.N), device=device, dtype=torch.complex64)
    H[: rk4_model.N_in, : rk4_model.N_in] = Hu
    psi_u = torch.matrix_exp((-1j) * H * rk4_model.T_u).unsqueeze(0) @ psi0
    rho_u = psi_u @ psi_u.mH
    gamma = rk4_model.gamma.to(torch.complex64)

    def run_rk4():
        import qsw
        return qsw.evolve_qsnn2d_stage2_structured(
            rho_u, H, gamma, rk4_model.T_d, rk4_model.N_in, steps=stage2_steps
        )

    def run_split():
        import qsw
        return qsw.evolve_qsnn2d_stage2_split(
            rho_u, H, gamma, rk4_model.T_d, rk4_model.N_in, steps=stage2_steps
        )

    measurements = []
    outputs = {}
    for method, fn in [("rk4", run_rk4), ("split", run_split)]:
        with torch.no_grad():
            for _ in range(warmup):
                rho = fn()
            sync_if_needed(device)
            t0 = time.perf_counter()
            for _ in range(runs):
                rho = fn()
            sync_if_needed(device)
            elapsed = time.perf_counter() - t0
        measurements.append(
            {
                "method": method,
                "avg_stage2_seconds": elapsed / runs,
            }
        )
        outputs[method] = rho

    rho_diff = (outputs["rk4"] - outputs["split"]).abs().max().item()
    rk4_t = measurements[0]["avg_stage2_seconds"]
    split_t = measurements[1]["avg_stage2_seconds"]
    return {
        "N": N,
        "batch_size": batch_size,
        "measurements": measurements,
        "time_ratio_rk4_to_split": rk4_t / split_t,
        "max_rho_diff": rho_diff,
    }


def benchmark_training(
    N: int,
    steps: int,
    device: torch.device,
    stage2_steps: int,
) -> Dict:
    X, y = make_circles(n=100, noise=0.05, factor=0.5, seed=0)
    Xd, yd = X.to(device), y.to(device)

    results = []
    for method in ["rk4", "split"]:
        torch.manual_seed(0)
        model = build_model(N, device, method, stage2_steps)
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

        results.append(
            {
                "method": method,
                "train_seconds": elapsed,
                "final_loss": last_loss,
            }
        )

    return {
        "N": N,
        "steps": steps,
        "measurements": results,
        "time_ratio_rk4_to_split": results[0]["train_seconds"] / results[1]["train_seconds"],
    }


def print_section(title: str, rows: List[Dict], time_key: str) -> None:
    print(f"\n=== {title} ===")
    for row in rows:
        print(f"N={row['N']}")
        for m in row["measurements"]:
            print(f"  {m['method']:10s} {time_key}={m[time_key]:.6f}")
        print(f"  time_ratio_rk4_to_split={row['time_ratio_rk4_to_split']:.3f}")
        if "max_prob_diff" in row:
            print(
                f"  max_prob_diff={row['max_prob_diff']:.3e} "
                f"max_rho_diff={row['max_rho_diff']:.3e}"
            )
        elif "max_rho_diff" in row:
            print(f"  max_rho_diff={row['max_rho_diff']:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark QSNN2D stage-2 rk4 vs split."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--forward-ns", default="100,200")
    parser.add_argument("--stage2-ns", default="100,200,300")
    parser.add_argument("--train-n", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--forward-runs", type=int, default=10)
    parser.add_argument("--stage2-runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--stage2-steps", type=int, default=12)
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/chebyshev_comparision/benchmark_stage2_methods_results.json",
    )
    args = parser.parse_args()

    device = choose_device(args.device)
    payload = {
        "device": str(device),
        "config": {
            "forward_ns": parse_int_list(args.forward_ns),
            "stage2_ns": parse_int_list(args.stage2_ns),
            "train_n": args.train_n,
            "batch_size": args.batch_size,
            "forward_runs": args.forward_runs,
            "stage2_runs": args.stage2_runs,
            "warmup": args.warmup,
            "train_steps": args.train_steps,
            "stage2_steps": args.stage2_steps,
        },
        "forward_full": [],
        "stage2_only": [],
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
                )
            )
        print_section(
            "Forward Benchmark (Full Model)",
            payload["forward_full"],
            "avg_forward_seconds",
        )

    if not args.skip_stage2:
        for N in parse_int_list(args.stage2_ns):
            payload["stage2_only"].append(
                benchmark_stage2_only(
                    N=N,
                    batch_size=args.batch_size,
                    warmup=args.warmup,
                    runs=args.stage2_runs,
                    device=device,
                    stage2_steps=args.stage2_steps,
                )
            )
        print_section(
            "Stage-2 Only Benchmark",
            payload["stage2_only"],
            "avg_stage2_seconds",
        )

    if not args.skip_train:
        payload["training"] = benchmark_training(
            N=args.train_n,
            steps=args.train_steps,
            device=device,
            stage2_steps=args.stage2_steps,
        )
        print("\n=== Training Benchmark ===")
        print(f"N={payload['training']['N']} steps={payload['training']['steps']}")
        for item in payload["training"]["measurements"]:
            print(
                f"  {item['method']:10s} "
                f"train_seconds={item['train_seconds']:.6f} "
                f"final_loss={item['final_loss']:.6f}"
            )
        print(
            "  time_ratio_rk4_to_split="
            f"{payload['training']['time_ratio_rk4_to_split']:.3f}"
        )

    out_path = unique_output_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
