import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import make_circles
from models import QSNN2D


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def ratio_to_times(ratio: float, total_time: float) -> Tuple[float, float]:
    td = total_time / (1.0 + ratio)
    tu = total_time - td
    return tu, td


def build_model(
    N: int,
    device: torch.device,
    tu: float,
    td: float,
    stage1_method: str,
    stage2_method: str,
    stage2_steps: int,
    chebyshev_order: int,
    chebyshev_tol: float,
) -> QSNN2D:
    return QSNN2D(
        N_in=N - 2,
        T_u=tu,
        T_d=td,
        init_h=0.1,
        init_g=0.1,
        device=device,
        stage1_method=stage1_method,
        stage2_method=stage2_method,
        stage2_steps=stage2_steps,
        chebyshev_order=chebyshev_order,
        chebyshev_tol=chebyshev_tol,
    )


def run_training_case(
    N: int,
    tu_td_ratio: float,
    total_time: float,
    strategy_name: str,
    stage1_method: str,
    stage2_method: str,
    stage2_steps: int,
    train_steps: int,
    device: torch.device,
    seed: int,
    chebyshev_order: int,
    chebyshev_tol: float,
) -> Dict:
    torch.manual_seed(seed)
    X, y = make_circles(n=100, noise=0.05, factor=0.5, seed=seed)
    Xd, yd = X.to(device), y.to(device)
    tu, td = ratio_to_times(tu_td_ratio, total_time)

    model = build_model(
        N=N,
        device=device,
        tu=tu,
        td=td,
        stage1_method=stage1_method,
        stage2_method=stage2_method,
        stage2_steps=stage2_steps,
        chebyshev_order=chebyshev_order,
        chebyshev_tol=chebyshev_tol,
    )
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)

    sync_if_needed(device)
    t0 = time.perf_counter()
    last_loss = None
    for _ in range(train_steps):
        opt.zero_grad(set_to_none=True)
        probs, _ = model(Xd)
        loss = torch.nn.functional.nll_loss(torch.log(probs), yd)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
    sync_if_needed(device)
    elapsed = time.perf_counter() - t0

    return {
        "N": N,
        "tu_td_ratio": tu_td_ratio,
        "T_u": tu,
        "T_d": td,
        "strategy": strategy_name,
        "stage1_method": stage1_method,
        "stage2_method": stage2_method,
        "stage2_steps": stage2_steps,
        "train_seconds": elapsed,
        "final_loss": last_loss,
        "seed": seed,
    }


def find_row(rows: List[Dict], strategy: str, stage2_steps: int) -> Optional[Dict]:
    for row in rows:
        if row["strategy"] == strategy and row["stage2_steps"] == stage2_steps:
            return row
    return None


def find_best_feasible(rows: List[Dict], strategy: str, target_loss: float, eps: float) -> Optional[Dict]:
    feasible = [
        row for row in rows
        if row["strategy"] == strategy and row["final_loss"] <= target_loss + eps
    ]
    if not feasible:
        return None
    feasible.sort(key=lambda row: (row["train_seconds"], row["stage2_steps"]))
    return feasible[0]


def print_case_header(N: int, ratio: float, tu: float, td: float) -> None:
    print("\n==================================================")
    print(f"N={N}  T_u/T_d={ratio:.3f}  T_u={tu:.6f}  T_d={td:.6f}")


def print_raw_table(rows: List[Dict]) -> None:
    print("=== Raw Results ===")
    for row in rows:
        print(
            f"strategy={row['strategy']:18s} "
            f"stage2_steps={row['stage2_steps']:2d} "
            f"train_seconds={row['train_seconds']:.6f} "
            f"final_loss={row['final_loss']:.6f}"
        )


def print_summary(
    target_loss: float,
    eps: float,
    best_baseline: Optional[Dict],
    best_optimized: Optional[Dict],
) -> None:
    print("=== Loss-Target Summary ===")
    print(f"target_loss={target_loss:.6f}")
    print(f"tolerance_eps={eps:.6f}")

    if best_baseline is None:
        print("best_baseline: none")
    else:
        print(
            "best_baseline: "
            f"stage2_steps={best_baseline['stage2_steps']} "
            f"train_seconds={best_baseline['train_seconds']:.6f} "
            f"final_loss={best_baseline['final_loss']:.6f}"
        )

    if best_optimized is None:
        print("best_optimized: none")
    else:
        print(
            "best_optimized: "
            f"stage2_steps={best_optimized['stage2_steps']} "
            f"train_seconds={best_optimized['train_seconds']:.6f} "
            f"final_loss={best_optimized['final_loss']:.6f}"
        )

    if best_baseline is not None and best_optimized is not None:
        ratio = best_baseline["train_seconds"] / best_optimized["train_seconds"]
        print(f"time_ratio_baseline_to_optimized={ratio:.3f}")
        if best_optimized["train_seconds"] < best_baseline["train_seconds"]:
            print("conclusion: optimized strategy is faster at matched loss tolerance")
        else:
            print("conclusion: optimized strategy is not faster at matched loss tolerance")
    else:
        print("conclusion: insufficient feasible configurations for one or both strategies")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline (exact_state+rk4) vs optimized (chebyshev+split) at matched loss across N and T_u/T_d ratios."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--N-list", default="100,200")
    parser.add_argument("--tu-td-ratios", default="0.5,1.0,2.0")
    parser.add_argument("--total-time", type=float, default=2.0)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage2-steps-list", default="8,12,16,20,24,32")
    parser.add_argument("--eps", type=float, default=0.002)
    parser.add_argument("--chebyshev-order", type=int, default=128)
    parser.add_argument("--chebyshev-tol", type=float, default=1e-10)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    device = choose_device(args.device)
    n_list = parse_int_list(args.N_list)
    ratio_list = parse_float_list(args.tu_td_ratios)
    stage2_steps_list = parse_int_list(args.stage2_steps_list)

    print(f"device: {device}")
    print(f"N_list={n_list}")
    print(f"tu_td_ratios={ratio_list}")
    print(f"stage2_steps_list={stage2_steps_list}")

    all_cases = []

    for N in n_list:
        for ratio in ratio_list:
            tu, td = ratio_to_times(ratio, args.total_time)
            print_case_header(N, ratio, tu, td)

            rows = []
            for strategy_name, stage1_method, stage2_method in [
                ("baseline", "exact", "rk4"),
                ("optimized", "chebyshev", "split"),
            ]:
                for stage2_steps in stage2_steps_list:
                    row = run_training_case(
                        N=N,
                        tu_td_ratio=ratio,
                        total_time=args.total_time,
                        strategy_name=strategy_name,
                        stage1_method=stage1_method,
                        stage2_method=stage2_method,
                        stage2_steps=stage2_steps,
                        train_steps=args.train_steps,
                        device=device,
                        seed=args.seed,
                        chebyshev_order=args.chebyshev_order,
                        chebyshev_tol=args.chebyshev_tol,
                    )
                    rows.append(row)

            print_raw_table(rows)

            baseline_ref = find_row(rows, "baseline", 12)
            if baseline_ref is None:
                raise ValueError("Baseline exact_state+rk4 with stage2_steps=12 not found.")

            target_loss = baseline_ref["final_loss"]
            best_baseline = find_best_feasible(rows, "baseline", target_loss, args.eps)
            best_optimized = find_best_feasible(rows, "optimized", target_loss, args.eps)
            print_summary(target_loss, args.eps, best_baseline, best_optimized)

            all_cases.append(
                {
                    "N": N,
                    "tu_td_ratio": ratio,
                    "T_u": tu,
                    "T_d": td,
                    "target_loss": target_loss,
                    "raw_results": rows,
                    "best_baseline": best_baseline,
                    "best_optimized": best_optimized,
                }
            )

    payload = {
        "device": str(device),
        "config": {
            "N_list": n_list,
            "tu_td_ratios": ratio_list,
            "total_time": args.total_time,
            "train_steps": args.train_steps,
            "seed": args.seed,
            "stage2_steps_list": stage2_steps_list,
            "eps": args.eps,
            "chebyshev_order": args.chebyshev_order,
            "chebyshev_tol": args.chebyshev_tol,
            "baseline_strategy": {
                "stage1_method": "exact",
                "stage2_method": "rk4",
                "reference_stage2_steps": 12,
            },
            "optimized_strategy": {
                "stage1_method": "chebyshev",
                "stage2_method": "split",
            },
        },
        "cases": all_cases,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
