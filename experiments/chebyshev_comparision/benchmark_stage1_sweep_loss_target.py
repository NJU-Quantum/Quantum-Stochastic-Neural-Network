import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
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


def build_model(
    N: int,
    device: torch.device,
    stage1_method: str,
    stage2_method: str,
    stage2_steps: int,
    chebyshev_order: int,
    chebyshev_tol: float,
    stage1_suzuki_steps: int,
    stage1_suzuki_order: int,
) -> QSNN2D:
    return QSNN2D(
        N_in=N - 2,
        T_u=1.0,
        T_d=1.0,
        init_h=0.1,
        init_g=0.1,
        device=device,
        stage1_method=stage1_method,
        stage2_method=stage2_method,
        stage2_steps=stage2_steps,
        chebyshev_order=chebyshev_order,
        chebyshev_tol=chebyshev_tol,
        stage1_suzuki_steps=stage1_suzuki_steps,
        stage1_suzuki_order=stage1_suzuki_order,
    )


def run_training_case(
    N: int,
    stage1_method: str,
    stage2_method: str,
    stage2_steps: int,
    train_steps: int,
    device: torch.device,
    seed: int,
    chebyshev_order: int,
    chebyshev_tol: float,
    stage1_suzuki_steps: int,
    stage1_suzuki_order: int,
) -> Dict:
    torch.manual_seed(seed)
    X, y = make_circles(n=100, noise=0.05, factor=0.5, seed=seed)
    Xd, yd = X.to(device), y.to(device)

    model = build_model(
        N=N,
        device=device,
        stage1_method=stage1_method,
        stage2_method=stage2_method,
        stage2_steps=stage2_steps,
        chebyshev_order=chebyshev_order,
        chebyshev_tol=chebyshev_tol,
        stage1_suzuki_steps=stage1_suzuki_steps,
        stage1_suzuki_order=stage1_suzuki_order,
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

    row = {
        "stage1_method": "exact_state" if stage1_method == "exact" else stage1_method,
        "train_seconds": elapsed,
        "final_loss": last_loss,
        "seed": seed,
        "stage2_method": stage2_method,
        "stage2_steps": stage2_steps,
        "chebyshev_order": None,
        "stage1_suzuki_steps": None,
        "stage1_suzuki_order": None,
    }
    if stage1_method == "chebyshev":
        row["chebyshev_order"] = chebyshev_order
    if stage1_method == "suzuki":
        row["stage1_suzuki_steps"] = stage1_suzuki_steps
        row["stage1_suzuki_order"] = stage1_suzuki_order
    return row


def find_best_feasible(rows: List[Dict], method: str, target_loss: float, eps: float) -> Optional[Dict]:
    feasible = [
        row for row in rows
        if row["stage1_method"] == method and row["final_loss"] <= target_loss + eps
    ]
    if not feasible:
        return None
    feasible.sort(
        key=lambda row: (
            row["train_seconds"],
            row["chebyshev_order"] if row["chebyshev_order"] is not None else 0,
            row["stage1_suzuki_steps"] if row["stage1_suzuki_steps"] is not None else 0,
        )
    )
    return feasible[0]


def method_config_label(row: Dict) -> str:
    if row["stage1_method"] == "chebyshev":
        return f"order={row['chebyshev_order']}"
    if row["stage1_method"] == "suzuki":
        return f"steps={row['stage1_suzuki_steps']}, order={row['stage1_suzuki_order']}"
    return "baseline"


def print_raw_table(rows: List[Dict]) -> None:
    print("\n=== Raw Results ===")
    for row in rows:
        print(
            f"method={row['stage1_method']:11s} "
            f"{method_config_label(row):18s} "
            f"train_seconds={row['train_seconds']:.6f} "
            f"final_loss={row['final_loss']:.6f}"
        )


def print_summary(
    target_loss: float,
    eps: float,
    best_exact: Optional[Dict],
    best_chebyshev: Optional[Dict],
    best_suzuki: Optional[Dict],
) -> None:
    print("\n=== Loss-Target Summary ===")
    print(f"target_loss={target_loss:.6f}")
    print(f"tolerance_eps={eps:.6f}")

    for label, row in [
        ("best_exact_state", best_exact),
        ("best_chebyshev", best_chebyshev),
        ("best_suzuki", best_suzuki),
    ]:
        if row is None:
            print(f"{label}: none")
        else:
            print(
                f"{label}: "
                f"{method_config_label(row)} "
                f"train_seconds={row['train_seconds']:.6f} "
                f"final_loss={row['final_loss']:.6f}"
            )

    if best_exact is not None and best_chebyshev is not None:
        ratio = best_exact["train_seconds"] / best_chebyshev["train_seconds"]
        print(f"time_ratio_best_exact_to_best_chebyshev={ratio:.3f}")
        if best_chebyshev["train_seconds"] < best_exact["train_seconds"]:
            print("conclusion_chebyshev: faster at matched loss tolerance")
        else:
            print("conclusion_chebyshev: not faster at matched loss tolerance")

    if best_exact is not None and best_suzuki is not None:
        ratio = best_exact["train_seconds"] / best_suzuki["train_seconds"]
        print(f"time_ratio_best_exact_to_best_suzuki={ratio:.3f}")
        if best_suzuki["train_seconds"] < best_exact["train_seconds"]:
            print("conclusion_suzuki: faster at matched loss tolerance")
        else:
            print("conclusion_suzuki: not faster at matched loss tolerance")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep stage1 methods at matched loss and compare training time."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage2-method", default="rk4", choices=["rk4", "split"])
    parser.add_argument("--stage2-steps", type=int, default=12)
    parser.add_argument("--chebyshev-order-list", default="32,64,96,128")
    parser.add_argument("--chebyshev-tol", type=float, default=1e-10)
    parser.add_argument("--stage1-suzuki-steps-list", default="4,8,12,16,24,32")
    parser.add_argument("--stage1-suzuki-order", type=int, default=2)
    parser.add_argument("--eps", type=float, default=0.002)
    parser.add_argument(
        "--out",
        type=str,
        default=str(SCRIPT_DIR / "results" / "benchmark_stage1_sweep_loss_target_results.json"),
    )
    args = parser.parse_args()

    device = choose_device(args.device)
    chebyshev_order_list = parse_int_list(args.chebyshev_order_list)
    suzuki_steps_list = parse_int_list(args.stage1_suzuki_steps_list)

    print(f"device: {device}")
    print(f"N={args.N} train_steps={args.train_steps} seed={args.seed}")
    print(f"stage2_method={args.stage2_method} stage2_steps={args.stage2_steps}")
    print(f"chebyshev_order_list={chebyshev_order_list}")
    print(
        "stage1_suzuki_steps_list="
        f"{suzuki_steps_list} stage1_suzuki_order={args.stage1_suzuki_order}"
    )

    rows = []

    rows.append(
        run_training_case(
            N=args.N,
            stage1_method="exact",
            stage2_method=args.stage2_method,
            stage2_steps=args.stage2_steps,
            train_steps=args.train_steps,
            device=device,
            seed=args.seed,
            chebyshev_order=128,
            chebyshev_tol=args.chebyshev_tol,
            stage1_suzuki_steps=12,
            stage1_suzuki_order=args.stage1_suzuki_order,
        )
    )

    for chebyshev_order in chebyshev_order_list:
        rows.append(
            run_training_case(
                N=args.N,
                stage1_method="chebyshev",
                stage2_method=args.stage2_method,
                stage2_steps=args.stage2_steps,
                train_steps=args.train_steps,
                device=device,
                seed=args.seed,
                chebyshev_order=chebyshev_order,
                chebyshev_tol=args.chebyshev_tol,
                stage1_suzuki_steps=12,
                stage1_suzuki_order=args.stage1_suzuki_order,
            )
        )

    for suzuki_steps in suzuki_steps_list:
        rows.append(
            run_training_case(
                N=args.N,
                stage1_method="suzuki",
                stage2_method=args.stage2_method,
                stage2_steps=args.stage2_steps,
                train_steps=args.train_steps,
                device=device,
                seed=args.seed,
                chebyshev_order=128,
                chebyshev_tol=args.chebyshev_tol,
                stage1_suzuki_steps=suzuki_steps,
                stage1_suzuki_order=args.stage1_suzuki_order,
            )
        )

    print_raw_table(rows)

    baseline = rows[0]
    target_loss = baseline["final_loss"]
    best_exact = find_best_feasible(rows, "exact_state", target_loss, args.eps)
    best_chebyshev = find_best_feasible(rows, "chebyshev", target_loss, args.eps)
    best_suzuki = find_best_feasible(rows, "suzuki", target_loss, args.eps)

    print_summary(target_loss, args.eps, best_exact, best_chebyshev, best_suzuki)

    payload = {
        "device": str(device),
        "config": {
            "N": args.N,
            "train_steps": args.train_steps,
            "seed": args.seed,
            "stage2_method": args.stage2_method,
            "stage2_steps": args.stage2_steps,
            "chebyshev_order_list": chebyshev_order_list,
            "chebyshev_tol": args.chebyshev_tol,
            "stage1_suzuki_steps_list": suzuki_steps_list,
            "stage1_suzuki_order": args.stage1_suzuki_order,
            "eps": args.eps,
            "baseline_method": "exact_state",
        },
        "raw_results": rows,
        "target_loss": target_loss,
        "best_exact_state": best_exact,
        "best_chebyshev": best_chebyshev,
        "best_suzuki": best_suzuki,
    }

    out_path = unique_output_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
