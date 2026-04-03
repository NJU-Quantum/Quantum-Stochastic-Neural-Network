import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import make_circles
from models import QSNN2D


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


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


def run_training_case(
    N: int,
    stage2_method: str,
    stage2_steps: int,
    train_steps: int,
    device: torch.device,
    seed: int,
) -> Dict:
    torch.manual_seed(seed)
    X, y = make_circles(n=100, noise=0.05, factor=0.5, seed=seed)
    Xd, yd = X.to(device), y.to(device)

    model = build_model(N, device, stage2_method, stage2_steps)
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
        "stage2_method": stage2_method,
        "stage2_steps": stage2_steps,
        "train_seconds": elapsed,
        "final_loss": last_loss,
        "seed": seed,
    }


def find_best_feasible(rows: List[Dict], method: str, target_loss: float, eps: float) -> Optional[Dict]:
    feasible = [
        row for row in rows
        if row["stage2_method"] == method and row["final_loss"] <= target_loss + eps
    ]
    if not feasible:
        return None
    feasible.sort(key=lambda row: (row["train_seconds"], row["stage2_steps"]))
    return feasible[0]


def print_raw_table(rows: List[Dict]) -> None:
    print("\n=== Raw Results ===")
    for row in rows:
        print(
            f"method={row['stage2_method']:5s} "
            f"stage2_steps={row['stage2_steps']:2d} "
            f"train_seconds={row['train_seconds']:.6f} "
            f"final_loss={row['final_loss']:.6f}"
        )


def print_summary(target_loss: float, eps: float, best_rk4: Optional[Dict], best_split: Optional[Dict]) -> None:
    print("\n=== Loss-Target Summary ===")
    print(f"target_loss={target_loss:.6f}")
    print(f"tolerance_eps={eps:.6f}")

    if best_rk4 is None:
        print("best_rk4: none")
    else:
        print(
            "best_rk4: "
            f"stage2_steps={best_rk4['stage2_steps']} "
            f"train_seconds={best_rk4['train_seconds']:.6f} "
            f"final_loss={best_rk4['final_loss']:.6f}"
        )

    if best_split is None:
        print("best_split: none")
    else:
        print(
            "best_split: "
            f"stage2_steps={best_split['stage2_steps']} "
            f"train_seconds={best_split['train_seconds']:.6f} "
            f"final_loss={best_split['final_loss']:.6f}"
        )

    if best_rk4 is not None and best_split is not None:
        ratio = best_rk4["train_seconds"] / best_split["train_seconds"]
        print(f"time_ratio_best_rk4_to_best_split={ratio:.3f}")
        if best_split["train_seconds"] < best_rk4["train_seconds"]:
            print("conclusion: split is faster at matched loss tolerance")
        else:
            print("conclusion: split is not faster at matched loss tolerance")
    else:
        print("conclusion: insufficient feasible configurations for one or both methods")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep stage2_method x stage2_steps and compare matched-loss training time."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage2-steps-list", default="8,12,16,20,24,32")
    parser.add_argument("--eps", type=float, default=0.002)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    device = choose_device(args.device)
    step_list = parse_int_list(args.stage2_steps_list)

    print(f"device: {device}")
    print(f"N={args.N} train_steps={args.train_steps} seed={args.seed}")
    print(f"stage2_steps_list={step_list}")

    rows = []
    for stage2_method in ["rk4", "split"]:
        for stage2_steps in step_list:
            row = run_training_case(
                N=args.N,
                stage2_method=stage2_method,
                stage2_steps=stage2_steps,
                train_steps=args.train_steps,
                device=device,
                seed=args.seed,
            )
            rows.append(row)

    print_raw_table(rows)

    baseline = None
    for row in rows:
        if row["stage2_method"] == "rk4" and row["stage2_steps"] == 12:
            baseline = row
            break

    if baseline is None:
        raise ValueError("Baseline rk4 with stage2_steps=12 not found in sweep results.")

    target_loss = baseline["final_loss"]
    best_rk4 = find_best_feasible(rows, "rk4", target_loss, args.eps)
    best_split = find_best_feasible(rows, "split", target_loss, args.eps)

    print_summary(target_loss, args.eps, best_rk4, best_split)

    payload = {
        "device": str(device),
        "config": {
            "N": args.N,
            "train_steps": args.train_steps,
            "seed": args.seed,
            "stage2_steps_list": step_list,
            "eps": args.eps,
            "baseline_method": "rk4",
            "baseline_stage2_steps": 12,
        },
        "raw_results": rows,
        "target_loss": target_loss,
        "best_rk4": best_rk4,
        "best_split": best_split,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
