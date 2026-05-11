import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from lm_tokenizer import create_tokenizer

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Head(nn.Module):
    def __init__(self, embed_dim: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [Head(embed_dim, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.sa = MultiHeadAttention(embed_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(
            *[Block(embed_dim, n_heads, block_size, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        if t > self.block_size:
            raise ValueError(f"Sequence length {t} exceeds block_size {self.block_size}")
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(t, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(b * t, -1), targets.reshape(b * t))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
):
    model.eval()
    out = {}
    for split_name, split_data in [("train", train_ids), ("val", val_ids)]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            xb, yb = get_batch(split_data, batch_size, block_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss
        out[split_name] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny Transformer language model on TS_modern_english.txt")
    parser.add_argument("--input", type=str, default="tasks/poem_recognition/TS_modern_english.txt")
    parser.add_argument("--outdir", type=str, default="outputs/poem_recognition/ts_modern_lm")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--level", type=str, choices=["char", "word", "bpe"], default="word")
    parser.add_argument("--bpe-merges", type=int, default=1500)
    parser.add_argument("--bpe-min-pair-freq", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--gen-tokens", type=int, default=120)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8")
    tokenizer = create_tokenizer(
        level=args.level,
        bpe_merges=args.bpe_merges,
        bpe_min_pair_freq=args.bpe_min_pair_freq,
    )
    tokenizer.fit(text)
    ids_list = tokenizer.encode(text)

    ids = torch.tensor(ids_list, dtype=torch.long)
    n_train = int(0.9 * len(ids))
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]
    vocab_size = tokenizer.vocab_size()

    # Validate token IDs are within bounds
    max_id = ids.max().item()
    min_id = ids.min().item()
    invalid_tokens = (ids >= vocab_size).sum().item()
    print(f"device={device}")
    print(
        f"level={args.level} tokens={len(ids_list)} vocab_size={vocab_size} "
        f"train={len(train_ids)} val={len(val_ids)}"
    )
    print(f"Token ID range: [{min_id}, {max_id}], Invalid tokens (>= vocab_size): {invalid_tokens}")

    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_curve = []
    val_curve = []
    steps = []

    for step in range(1, args.max_iters + 1):
        xb, yb = get_batch(train_ids, args.batch_size, args.block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1 or step == args.max_iters:
            metrics = estimate_loss(
                model=model,
                train_ids=train_ids,
                val_ids=val_ids,
                eval_iters=args.eval_iters,
                batch_size=args.batch_size,
                block_size=args.block_size,
                device=device,
            )
            steps.append(step)
            train_curve.append(metrics["train"])
            val_curve.append(metrics["val"])
            print(f"step={step:5d} train_loss={metrics['train']:.4f} val_loss={metrics['val']:.4f}")

    # Save artifacts
    model_path = outdir / "model.pt"
    vocab_path = outdir / "vocab.json"
    loss_png = outdir / "loss_curve.png"
    sample_path = outdir / "sample_generation.txt"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "block_size": args.block_size,
                "embed_dim": args.embed_dim,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
            },
        },
        model_path,
    )

    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(tokenizer.to_serializable(), f, ensure_ascii=False, indent=2)

    if _HAS_MPL:
        plt.figure(figsize=(8, 4))
        plt.plot(steps, train_curve, label="train")
        plt.plot(steps, val_curve, label="val")
        plt.xlabel("step")
        plt.ylabel("cross-entropy")
        plt.title("TS Modern English LM Training")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_png, dpi=160)
        plt.close()

    # Sample generation from a random short context in validation split.
    start_pos = max(0, len(val_ids) // 3)
    context = val_ids[start_pos : start_pos + min(8, len(val_ids))]
    if len(context) < 1:
        context = torch.tensor([1], dtype=torch.long)
    context = context.unsqueeze(0).to(device)
    gen_ids = model.generate(context, max_new_tokens=args.gen_tokens)[0].cpu()
    sample_text = tokenizer.decode_ids(gen_ids.tolist())
    sample_path.write_text(sample_text, encoding="utf-8")

    print(f"saved: {model_path}")
    print(f"saved: {vocab_path}")
    if _HAS_MPL:
        print(f"saved: {loss_png}")
    else:
        print("skip: matplotlib unavailable, loss plot not generated")
    print(f"saved: {sample_path}")


if __name__ == "__main__":
    main()
