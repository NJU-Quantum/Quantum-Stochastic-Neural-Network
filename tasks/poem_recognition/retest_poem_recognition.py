import argparse
import csv
import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch

from models import QSNNText


TRAIN_YES = [
    "There is a gold sun at dawn",
    "I love stay all day in the sun",
    "He went for gold that day",
    "He love gold but have nothing",
    "He loves the dawn of a day",
    "I love the lovely sun",
]

TRAIN_NO = [
    "sun gold day i",
    "day nothing dawn",
    "gold goes love",
    "stay love go sun",
    "day gold nothing",
    "stay dawn love go",
]

DEFAULT_RETEST = [
    ("so dawn goes down to day", 1),
    ("nothing gold can stay", 1),
    ("i love to stay here until the dawn", 1),
    ("i love to go out for love", 1),
    ("sun gold day i", 0),
    ("gold goes love", 0),
]

LEGACY_TEST_SENTENCES = [
    "so dawn goes down to day",
    "nothing gold can stay",
    "i love to stay here until the dawn",
    "i love to go out for love",
]


def parse_case(text: str) -> Tuple[str, int]:
    if "|||" not in text:
        raise ValueError("Case must be 'sentence|||label'.")
    sent, label = text.rsplit("|||", 1)
    label = label.strip()
    if label not in {"0", "1"}:
        raise ValueError("Label must be 0 or 1.")
    return sent.strip(), int(label)


def load_cases_from_file(path: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(parse_case(line))
    return out


def collect_cases(args: argparse.Namespace) -> List[Tuple[str, int]]:
    if args.case_file:
        return load_cases_from_file(args.case_file)
    if args.case:
        return [parse_case(c) for c in args.case]
    return list(DEFAULT_RETEST)


def load_legacy_matrix(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"Empty legacy file: {path}")
    arr = np.array(eval(text), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Legacy file must be 2D list: {path}")
    return arr


def write_legacy_files(
    out_dir: str,
    prefix: str,
    loss_samples: List[List[float]],
    sent_samples: List[List[List[float]]],
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    loss_path = os.path.join(out_dir, f"{prefix}_loss.txt")
    with open(loss_path, "w", encoding="utf-8") as f:
        f.write(str(loss_samples))
    paths.append(loss_path)

    for i, samples in enumerate(sent_samples, start=1):
        p = os.path.join(out_dir, f"{prefix}_test{i}.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(str(samples))
        paths.append(p)

    return paths


def save_comparison_csv(
    csv_path: str,
    old_loss: np.ndarray,
    old_tests: List[np.ndarray],
    new_loss: np.ndarray,
    new_tests: List[np.ndarray],
) -> None:
    steps = min(old_loss.shape[1], new_loss.shape[1])
    header = [
        "update",
        "old_loss_mean",
        "new_loss_mean",
        "delta_loss",
        "old_test1_mean",
        "new_test1_mean",
        "delta_test1",
        "old_test2_mean",
        "new_test2_mean",
        "delta_test2",
        "old_test3_mean",
        "new_test3_mean",
        "delta_test3",
        "old_test4_mean",
        "new_test4_mean",
        "delta_test4",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for u in range(steps):
            old_loss_mean = float(np.mean(old_loss[:, u]))
            new_loss_mean = float(np.mean(new_loss[:, u]))
            row = [u, old_loss_mean, new_loss_mean, new_loss_mean - old_loss_mean]
            for i in range(4):
                old_m = float(np.mean(old_tests[i][:, u]))
                new_m = float(np.mean(new_tests[i][:, u]))
                row.extend([old_m, new_m, new_m - old_m])
            w.writerow(row)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Retest poem/sentence recognition using QSNNText and export legacy-compatible outputs."
    )
    p.add_argument("--lr", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--legacy-updates", type=int, default=200)
    p.add_argument("--legacy-samples", type=int, default=15)
    p.add_argument("--legacy-output-dir", type=str, default=".")
    p.add_argument("--legacy-prefix", type=str, default="retest_qsnntext")
    p.add_argument("--old-result-dir", type=str)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--use-nltk", action="store_true")
    p.add_argument(
        "--case",
        action="append",
        help="Custom retest case in format: sentence|||label (0 or 1). Repeatable.",
    )
    p.add_argument(
        "--case-file",
        type=str,
        help="Path to a text file, one case per line, format: sentence|||label",
    )
    return p


def lr_schedule(base_lr: float, update_num: int, update_num_cap: int = 15) -> float:
    if update_num <= 100:
        return base_lr / (1.0 + update_num / update_num_cap)
    return base_lr / (1.0 + 100.0 / update_num_cap)


def train_one_sample(
    seed: int,
    device: str,
    lr0: float,
    updates: int,
    use_nltk: bool,
) -> Tuple[List[float], List[List[float]], float]:
    set_seed(seed)

    model = QSNNText(device=device, use_nltk=use_nltk)
    if use_nltk and not model.use_nltk:
        raise RuntimeError(
            "--use-nltk is enabled, but QSNNText fell back to simple tokenizer. "
            "Please verify NLTK corpora installation."
        )
    model.train()

    train_texts = TRAIN_YES + TRAIN_NO
    train_labels = torch.tensor([1] * len(TRAIN_YES) + [0] * len(TRAIN_NO), device=device, dtype=torch.long)

    # Quantum features are fixed for this model setup; cache once for speed.
    with torch.no_grad():
        train_features = model.encode_sentences(train_texts).detach()
        legacy_test_features = model.encode_sentences(LEGACY_TEST_SENTENCES).detach()

    # Initialize trajectory with current parameters.
    loss_hist: List[float] = []
    test_hist: List[List[float]] = [[], [], [], []]

    def snapshot() -> None:
        logits = model.classifier(train_features)
        probs = torch.softmax(logits, dim=-1)
        p_correct = probs.gather(1, train_labels.view(-1, 1)).squeeze(1)
        legacy_cost = 1.0 - p_correct.mean()
        loss_hist.append(float(legacy_cost.item()))

        t_logits = model.classifier(legacy_test_features)
        t_probs = torch.softmax(t_logits, dim=-1)[:, 1]
        for i in range(4):
            test_hist[i].append(float(t_probs[i].item()))

    snapshot()

    for u in range(updates):
        lr = lr_schedule(lr0, u)
        logits = model.classifier(train_features)
        probs = torch.softmax(logits, dim=-1)
        p_correct = probs.gather(1, train_labels.view(-1, 1)).squeeze(1)
        loss = 1.0 - p_correct.mean()

        model.classifier.weight.grad = None
        model.classifier.bias.grad = None
        loss.backward()

        with torch.no_grad():
            model.classifier.weight -= lr * model.classifier.weight.grad
            model.classifier.bias -= lr * model.classifier.bias.grad

        snapshot()

    final_train_acc = float((probs.argmax(dim=-1) == train_labels).float().mean().item())
    return loss_hist, test_hist, final_train_acc


def evaluate_retest(
    model: QSNNText,
    cases: Sequence[Tuple[str, int]],
    device: str,
) -> Tuple[float, List[Tuple[str, int, float, int]]]:
    texts = [x[0] for x in cases]
    labels = torch.tensor([x[1] for x in cases], device=device, dtype=torch.long)

    with torch.no_grad():
        feats = model.encode_sentences(texts)
        logits = model.classifier(feats)
        probs = torch.softmax(logits, dim=-1)
        p_yes = probs[:, 1]
        pred = probs.argmax(dim=-1)

    acc = float((pred == labels).float().mean().item())
    rows = []
    for i, s in enumerate(texts):
        rows.append((s, int(labels[i].item()), float(p_yes[i].item()), int(pred[i].item())))
    return acc, rows


def main() -> None:
    args = build_parser().parse_args()
    device = args.device
    retest_cases = collect_cases(args)

    # One upfront sanity check so users can confirm whether NLTK mode is truly active.
    check_model = QSNNText(device=device, use_nltk=args.use_nltk)
    if args.use_nltk and not check_model.use_nltk:
        raise RuntimeError(
            "Requested --use-nltk, but NLTK mode is not active. "
            "Install/download NLTK resources first."
        )
    print(f"NLTK active: {check_model.use_nltk}")

    loss_samples: List[List[float]] = []
    sent_samples: List[List[List[float]]] = [[], [], [], []]
    train_acc_samples: List[float] = []

    for k in range(args.legacy_samples):
        losses, tests, train_acc = train_one_sample(
            seed=args.seed + k,
            device=device,
            lr0=args.lr,
            updates=args.legacy_updates,
            use_nltk=args.use_nltk,
        )
        loss_samples.append(losses)
        for i in range(4):
            sent_samples[i].append(tests[i])
        train_acc_samples.append(train_acc)

    # Build one final model (same seed) for custom retest report.
    final_model = QSNNText(device=device, use_nltk=args.use_nltk)
    set_seed(args.seed)
    with torch.no_grad():
        final_model.classifier.weight.uniform_(-1.0, 1.0)
        final_model.classifier.bias.uniform_(-1.0, 1.0)
        train_features = final_model.encode_sentences(TRAIN_YES + TRAIN_NO).detach()
        labels = torch.tensor([1] * len(TRAIN_YES) + [0] * len(TRAIN_NO), device=device, dtype=torch.long)

    for u in range(args.legacy_updates):
        lr = lr_schedule(args.lr, u)
        logits = final_model.classifier(train_features)
        probs = torch.softmax(logits, dim=-1)
        p_correct = probs.gather(1, labels.view(-1, 1)).squeeze(1)
        loss = 1.0 - p_correct.mean()
        final_model.classifier.weight.grad = None
        final_model.classifier.bias.grad = None
        loss.backward()
        with torch.no_grad():
            final_model.classifier.weight -= lr * final_model.classifier.weight.grad
            final_model.classifier.bias -= lr * final_model.classifier.bias.grad

    retest_acc, rows = evaluate_retest(final_model, retest_cases, device)

    print("=== Poem Recognition Retest (QSNNText) ===")
    print(f"Train samples: {len(TRAIN_YES) + len(TRAIN_NO)} | Retest samples: {len(retest_cases)}")
    print(f"Train accuracy (mean over samples): {float(np.mean(train_acc_samples)):.4f}")
    print(f"Retest accuracy: {retest_acc:.4f}")
    print()
    print("sentence\tlabel\tp_yes\tpred")
    for sent, label, p_yes, pred in rows:
        print(f"{sent}\t{label}\t{p_yes:.4f}\t{pred}")

    written = write_legacy_files(
        out_dir=args.legacy_output_dir,
        prefix=args.legacy_prefix,
        loss_samples=loss_samples,
        sent_samples=sent_samples,
    )

    print()
    print("Legacy-format files:")
    for p in written:
        print(p)

    if args.old_result_dir:
        old_loss_path = os.path.join(args.old_result_dir, "classicalNN_loss.txt")
        old_test_paths = [
            os.path.join(args.old_result_dir, f"classicalNN_test{i}.txt") for i in range(1, 5)
        ]

        old_loss = load_legacy_matrix(old_loss_path)
        old_tests = [load_legacy_matrix(p) for p in old_test_paths]
        new_loss = np.array(loss_samples, dtype=np.float64)
        new_tests = [np.array(s, dtype=np.float64) for s in sent_samples]

        cmp_path = os.path.join(args.legacy_output_dir, "old_vs_new_comparison.csv")
        save_comparison_csv(
            csv_path=cmp_path,
            old_loss=old_loss,
            old_tests=old_tests,
            new_loss=new_loss,
            new_tests=new_tests,
        )
        print("Comparison table:")
        print(cmp_path)


if __name__ == "__main__":
    main()
