from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import qsw


CHINESE_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
KEEP_PUNCT = set("，。！？；：、,.!?;:（）()《》“”‘’【】—-")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_poem_text(text: str) -> str:
    # 统一文本形态：只保留中文字符和常见标点，
    # 并把换行折叠成句号，方便后续做字符级编码。
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chars: list[str] = []
    for ch in text:
        if CHINESE_CHAR_RE.match(ch) or ch in KEEP_PUNCT:
            chars.append(ch)
        elif ch == "\n":
            chars.append("。")
    merged = "".join(chars)
    merged = re.sub(r"[。]{2,}", "。", merged)
    return merged.strip("。 ")


def extract_text_from_record(record: object) -> str | None:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return None

    for key in ("text", "content", "body"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    paragraphs = record.get("paragraphs")
    if isinstance(paragraphs, list):
        parts = [str(part).strip() for part in paragraphs if str(part).strip()]
        if parts:
            return "".join(parts)
    return None


def iter_texts_from_json(path: Path) -> Iterable[str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        for item in data:
            text = extract_text_from_record(item)
            if text:
                yield text
        return

    if isinstance(data, dict):
        for key in ("poems", "data", "items", "records"):
            value = data.get(key)
            if isinstance(value, list):
                for item in value:
                    text = extract_text_from_record(item)
                    if text:
                        yield text
                return
        text = extract_text_from_record(data)
        if text:
            yield text


def iter_texts_from_jsonl(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = line
            text = extract_text_from_record(record)
            if text:
                yield text


def iter_texts_from_txt(path: Path) -> Iterable[str]:
    raw = path.read_text(encoding="utf-8-sig")
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", raw) if block.strip()]
    if not blocks:
        return
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) >= 3 and len(lines[0]) <= 20 and len(lines[1]) <= 20:
            yield "".join(lines[2:])
        else:
            yield "".join(lines)


def iter_corpus_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for sub in sorted(path.rglob("*")):
        if sub.is_file() and sub.suffix.lower() in {".json", ".jsonl", ".txt"}:
            yield sub


def load_corpus_texts(path: Path, limit: int | None = None) -> list[str]:
    # 把 json / jsonl / txt / 目录 统一整理成训练所需的纯文本列表。
    texts: list[str] = []
    for sub in iter_corpus_paths(path):
        suffix = sub.suffix.lower()
        if suffix == ".json":
            iterator = iter_texts_from_json(sub)
        elif suffix == ".jsonl":
            iterator = iter_texts_from_jsonl(sub)
        elif suffix == ".txt":
            iterator = iter_texts_from_txt(sub)
        else:
            continue

        for text in iterator:
            cleaned = normalize_poem_text(text)
            if cleaned:
                texts.append(cleaned)
                if limit is not None and len(texts) >= limit:
                    return texts
    return texts


def synthetic_negative_from_text(text: str, partner: str, rng: random.Random) -> str:
    # 当没有现成负样本时，用诗句自身做扰动，快速合成“非诗句”样本。
    units = [u for u in re.split(r"[，。！？；：、,.!?;:]+", text) if u]
    partner_units = [u for u in re.split(r"[，。！？；：、,.!?;:]+", partner) if u]
    if not units:
        return text[::-1]

    mode = rng.choice(["rotate", "swap_halves", "stride", "cross_mix"])
    if mode == "rotate" and len(units) >= 2:
        k = rng.randrange(1, len(units))
        mixed = units[k:] + units[:k]
        return "，".join(mixed) + "。"
    if mode == "swap_halves":
        chars = list(text)
        mid = len(chars) // 2
        return "".join(chars[mid:] + chars[:mid])
    if mode == "stride":
        chars = [c for c in text if c not in " "]
        return "".join(chars[::2] + chars[1::2])

    left = units[: max(1, len(units) // 2)]
    right = partner_units[max(0, len(partner_units) // 2) :] or partner_units
    if not right:
        right = units[::-1]
    return "，".join(left + right) + "。"


def build_synthetic_negative_corpus(texts: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    negatives: list[str] = []
    n = len(texts)
    for idx, text in enumerate(texts):
        partner = texts[(idx + rng.randrange(1, max(n, 2))) % n] if n > 1 else text[::-1]
        negatives.append(synthetic_negative_from_text(text, partner, rng))
    return negatives


def build_vocab(
    texts: list[str],
    max_size: int,
    min_freq: int,
    reserved_tokens: list[str],
    ngram: int = 1,
) -> tuple[dict[str, int], list[tuple[str, int]]]:
    # 这里建立的是字符/双字词表；真正送入 QSNN 的不是 one-hot，
    # 而是后面投影到更低维 latent space 的嵌入表示。
    counter: Counter[str] = Counter()
    for text in texts:
        chars = list(text)
        if ngram == 1:
            counter.update(chars)
        else:
            counter.update("".join(chars[i : i + ngram]) for i in range(max(0, len(chars) - ngram + 1)))

    items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
    items.sort(key=lambda item: (-item[1], item[0]))

    keep = max(0, max_size - len(reserved_tokens))
    items = items[:keep]
    vocab = {tok: idx for idx, tok in enumerate(reserved_tokens)}
    for token, _ in items:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab, items


@dataclass
class Example:
    text: str
    label: int


class PoetryBinaryDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        char_vocab: dict[str, int],
        bigram_vocab: dict[str, int],
        max_len: int,
    ) -> None:
        self.examples = examples
        self.char_vocab = char_vocab
        self.bigram_vocab = bigram_vocab
        self.max_len = max_len
        self.pad_id = char_vocab["<PAD>"]
        self.unk_id = char_vocab["<UNK>"]
        self.bigram_pad_id = bigram_vocab["<PAD>"]
        self.bigram_unk_id = bigram_vocab["<UNK>"]

    def __len__(self) -> int:
        return len(self.examples)

    def _encode_chars(self, text: str) -> list[int]:
        # 先截断到最大长度，再映射成字符 id。
        chars = list(text)[: self.max_len]
        return [self.char_vocab.get(ch, self.unk_id) for ch in chars]

    def _encode_bigrams(self, text: str, length: int) -> list[int]:
        # bigram 用来补充中文里相邻两个字的局部搭配信息。
        chars = list(text)[: self.max_len]
        bigrams: list[int] = []
        for i in range(length):
            if i + 1 < length:
                token = chars[i] + chars[i + 1]
                bigrams.append(self.bigram_vocab.get(token, self.bigram_unk_id))
            else:
                bigrams.append(self.bigram_pad_id)
        return bigrams

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        char_ids = self._encode_chars(example.text)
        length = len(char_ids)
        # bigram_ids 与 char_ids 按位置对齐：
        # 第 i 个 bigram 表示 "第 i 个字 + 第 i+1 个字"，
        # 这样模型在看当前位置时，同时能拿到单字信息和局部双字搭配信息。
        bigram_ids = self._encode_bigrams(example.text, length)
        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "bigram_ids": torch.tensor(bigram_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(example.label, dtype=torch.long),
        }

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        # batch 内做动态 padding，并生成 mask，告诉模型哪些位置是真实字符。
        max_len = max(int(item["length"].item()) for item in batch)
        batch_size = len(batch)

        char_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        bigram_ids = torch.full((batch_size, max_len), self.bigram_pad_id, dtype=torch.long)
        mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
        labels = torch.zeros((batch_size,), dtype=torch.long)

        for row, item in enumerate(batch):
            length = int(item["length"].item())
            char_ids[row, :length] = item["char_ids"][:length]
            bigram_ids[row, :length] = item["bigram_ids"][:length]
            mask[row, :length] = 1.0
            labels[row] = item["label"]

        return {
            "char_ids": char_ids,
            "bigram_ids": bigram_ids,
            "mask": mask,
            "labels": labels,
        }


class ChinesePoemQSNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        bigram_vocab_size: int,
        latent_dim: int,
        max_len: int,
        t_unitary: float,
        t_dissipative: float,
        stage2_steps: int,
        dropout: float,
        use_bigrams: bool = True,
    ) -> None:
        super().__init__()
        # total_dim = 潜在 QSNN 神经元数 + 2 个分类输出节点。
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.t_unitary = float(t_unitary)
        self.t_dissipative = float(t_dissipative)
        self.stage2_steps = int(stage2_steps)
        self.use_bigrams = use_bigrams and bigram_vocab_size > 2
        self.total_dim = latent_dim + 2

        self.char_emb = nn.Embedding(vocab_size, latent_dim, padding_idx=0)
        self.bigram_emb = nn.Embedding(bigram_vocab_size, latent_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, latent_dim)
        self.local_conv = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate_proj = nn.Linear(latent_dim, 1)
        self.phase_proj = nn.Linear(latent_dim, latent_dim)

        self.H_raw = nn.Parameter(0.02 * torch.randn(latent_dim, latent_dim))
        self.gamma_raw = nn.Parameter(0.02 * torch.randn(2, latent_dim))

    def _build_hamiltonian(self) -> torch.Tensor:
        # 相干演化只放在潜在层内部，输出节点主要用于最后的耗散读出。
        full = torch.zeros(
            (self.total_dim, self.total_dim),
            device=self.H_raw.device,
            dtype=torch.complex64,
        )
        hu = self.H_raw.to(torch.complex64)
        hu = 0.5 * (hu + hu.mH)
        full[: self.latent_dim, : self.latent_dim] = hu
        return full

    def _encode_sentence_state(
        self,
        char_ids: torch.Tensor,
        bigram_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        这是整个任务里最关键的“编码”步骤。

        输入:
        - char_ids: (B, T)，每个位置是一个中文字符 id
        - bigram_ids: (B, T)，每个位置是“当前位置字符 + 下一个字符”的双字 id
        - mask: (B, T)，1 表示真实字符，0 表示 padding

        输出:
        - psi: (B, total_dim, 1)，每个句子对应一个量子纯态列向量

        可以把它理解成：
        先把一句中文诗句变成一串经典向量，再把这串经典向量压缩成
        一个低维复向量，最后归一化成量子态 |psi>。
        """
        # 编码流程：
        # 1. 字符嵌入 + 位置嵌入 + bigram 嵌入
        # 2. 局部卷积提取近邻模式
        # 3. 每个 token 变成带幅值/相位的复向量
        # 4. 汇聚成一个低维量子纯态 |psi>
        batch_size, seq_len = char_ids.shape
        positions = torch.arange(seq_len, device=char_ids.device).unsqueeze(0)

        # char_emb 给出“这个字本身像什么”，
        # pos_emb 给出“这个字在句子里的位置”，
        # bigram_emb 给出“它和后一个字连在一起像什么”。
        # 三者相加后，当前位置就同时带有内容、顺序和局部搭配信息。
        x = self.char_emb(char_ids) + self.pos_emb(positions)
        if self.use_bigrams:
            x = x + self.bigram_emb(bigram_ids)

        # 用一层 1D 卷积在时间轴上看相邻几个字，
        # 比如“明月”“秋风”“白云”这类局部模式会在这里更容易被提出来。
        conv_out = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x + conv_out)
        x = self.dropout(x)
        # padding 位直接清零，避免补齐位置参与后面的量子态构造。
        x = x * mask.unsqueeze(-1)

        # token_scale 是每个位置经典向量的长度。
        # token_dir = 单位化方向，只保留“方向信息”，不保留长度大小。
        # 这一步的直觉类似：
        # - 方向决定这个 token 指向潜在空间中的哪个区域
        # - 大小不直接拿来当最终振幅，后面交给 gate 单独控制
        token_scale = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-6)
        token_dir = x / token_scale
        # gate 决定每个 token 对整句量子态贡献多大，
        # phase 则把经典特征映射为复振幅中的相位。
        #
        # 更具体一点：
        # - gate: 标量，范围约在 0~1，像“注意力强度/振幅强度”
        # - phase: 向量，每一维都在 [-pi, pi] 左右，决定复平面上的相位角
        #
        # 于是同一个字不只是在“哪个方向”，还会带一个“旋转角度”，
        # 这就是从实向量过渡到复向量、再过渡到量子态编码的关键一步。
        gate = torch.sigmoid(self.gate_proj(x)).squeeze(-1) * mask
        phase = math.pi * torch.tanh(self.phase_proj(x))

        # exp(i * phase) 会得到单位模长的复数相位因子。
        # token_amp 就是每个 token 的复向量表示：
        #   token_amp = 方向 * 相位
        # 之后再用 gate 作为幅值，把各位置贡献累加起来。
        token_amp = token_dir.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64))
        # 对时间维求和，相当于把整句所有 token 的贡献汇总成一个句向量。
        # 这里得到的 psi_latent 还不是最终量子态，只是 latent space 里的复向量。
        psi_latent = (gate.unsqueeze(-1).to(torch.complex64) * token_amp).sum(dim=1)

        # 防御性处理：极端情况下如果整句是空的，不让向量全 0。
        pad_guard = (mask.sum(dim=1, keepdim=True) == 0).to(torch.complex64)
        psi_latent[:, 0] = psi_latent[:, 0] + pad_guard.squeeze(-1)
        # 归一化后，psi_latent 才能被看作合法的量子纯态振幅向量。
        psi_latent = psi_latent / torch.linalg.vector_norm(psi_latent, dim=-1, keepdim=True).clamp_min(1e-6)

        psi = torch.zeros(
            (batch_size, self.total_dim, 1),
            device=char_ids.device,
            dtype=torch.complex64,
        )
        # 前 latent_dim 维存放句子的潜在量子态；
        # 最后两个位置暂时留空，后面给“非诗句/诗句”两个输出节点使用。
        psi[:, : self.latent_dim, 0] = psi_latent
        return psi

    def forward(self, char_ids: torch.Tensor, bigram_ids: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        # 整个 QSNN 分两段：
        # 先把句子编码成 rho0 并做相干演化，再通过结构化 Lindblad 耗散把概率泵到输出节点。
        psi = self._encode_sentence_state(char_ids, bigram_ids, mask)
        # 纯态 |psi> 变成密度矩阵 rho = |psi><psi|，
        # 这样后面就能直接接仓库里基于密度矩阵的 QSW / Lindblad 演化代码。
        rho0 = psi @ psi.mH

        H = self._build_hamiltonian()
        # 第一段只有哈密顿量、没有耗散项，表示潜在节点之间的相干混合。
        rho_u = qsw.evolve_auto(rho0, H, [], self.t_unitary)

        gamma = F.softplus(self.gamma_raw).to(torch.complex64) + 1e-4
        # 这里复用仓库已有的结构化耗散演化器，避免手写完整 Liouvillian。
        # 它会把 latent_dim 个潜在节点上的概率，按可学习的 gamma
        # 向两个输出节点传输，形成最终分类读出。
        rho_out = qsw.evolve_qsnn2d_stage2_structured(
            rho_u,
            H,
            gamma,
            self.t_dissipative,
            self.latent_dim,
            steps=self.stage2_steps,
        )

        # 读出时只看对角线占据概率，不直接用复振幅本身做分类。
        diag = torch.real(torch.diagonal(rho_out, dim1=-2, dim2=-1))
        out_pop = diag[:, self.latent_dim : self.latent_dim + 2].clamp_min(1e-8)
        # 直接读取两个输出节点上的占据概率，作为二分类结果。
        probs = out_pop / out_pop.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        logits = torch.log(probs)
        latent_mass = diag[:, : self.latent_dim].sum(dim=-1)

        return {
            "rho0": rho0,
            "rho_out": rho_out,
            "probs": probs,
            "logits": logits,
            "latent_mass": latent_mass,
        }


def split_dataset(
    examples: list[Example],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Example], list[Example], list[Example]]:
    # 分层切分，尽量避免小样本时验证集/测试集只落到某一类标签。
    rng = random.Random(seed)
    grouped: dict[int, list[Example]] = {}
    for example in examples:
        grouped.setdefault(example.label, []).append(example)

    train: list[Example] = []
    val: list[Example] = []
    test: list[Example] = []

    for label_examples in grouped.values():
        pool = list(label_examples)
        rng.shuffle(pool)
        n = len(pool)
        n_train = max(1, int(n * train_ratio))
        n_val = int(n * val_ratio)
        remaining = n - n_train - n_val

        if n >= 3 and remaining <= 0:
            if n_val > 0:
                n_val -= 1
            else:
                n_train = max(1, n_train - 1)
            remaining = n - n_train - n_val

        if n >= 5 and n_val == 0:
            n_val = 1
            remaining = n - n_train - n_val
            if remaining <= 0:
                n_train = max(1, n_train - 1)

        train.extend(pool[:n_train])
        val.extend(pool[n_train : n_train + n_val])
        test.extend(pool[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    if not train or not val or not test:
        raise ValueError("Dataset split produced an empty partition. Increase data size or adjust ratios.")

    return train, val, test


def build_examples(poem_texts: list[str], non_poem_texts: list[str]) -> list[Example]:
    examples = [Example(text=t, label=1) for t in poem_texts]
    examples.extend(Example(text=t, label=0) for t in non_poem_texts)
    return examples


@torch.no_grad()
def evaluate(
    model: ChinesePoemQSNN,
    loader: DataLoader,
    device: torch.device,
    latent_penalty: float,
) -> dict[str, float]:
    # 验证和测试共用这套评估逻辑，统一统计 loss / accuracy / latent_mass。
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_latent_mass = 0.0

    for batch in loader:
        char_ids = batch["char_ids"].to(device)
        bigram_ids = batch["bigram_ids"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(char_ids, bigram_ids, mask)
        loss = F.nll_loss(out["logits"], labels) + latent_penalty * out["latent_mass"].mean()

        preds = out["probs"].argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total_loss += float(loss.item()) * labels.numel()
        total_latent_mass += float(out["latent_mass"].mean().item()) * labels.numel()
        total_count += labels.numel()

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "latent_mass": total_latent_mass / max(total_count, 1),
    }


@torch.no_grad()
def collect_preview_predictions(
    model: ChinesePoemQSNN,
    examples: list[Example],
    dataset: PoetryBinaryDataset,
    device: torch.device,
    limit: int = 16,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model.eval()
    for idx, example in enumerate(examples[:limit]):
        item = dataset[idx]
        batch = dataset.collate([item])
        out = model(
            batch["char_ids"].to(device),
            batch["bigram_ids"].to(device),
            batch["mask"].to(device),
        )
        probs = out["probs"][0].detach().cpu().tolist()
        rows.append(
            {
                "text": example.text,
                "label": example.label,
                "prob_non_poem": float(probs[0]),
                "prob_poem": float(probs[1]),
                "pred": int(probs[1] >= probs[0]),
            }
        )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Chinese poetry-recognition QSNN with improved character/bigram encoding."
    )
    parser.add_argument(
        "--poem-data",
        type=str,
        default="tasks/corpus_pipline/poet.tang.0.joined.cleaned.jsonl",
        help="Positive poetry corpus path. Supports json/jsonl/txt/file-or-directory.",
    )
    parser.add_argument(
        "--non-poem-data",
        type=str,
        default="",
        help="Negative corpus path. If omitted, synthetic negatives are generated from the poem corpus.",
    )
    parser.add_argument("--poem-limit", type=int, default=0, help="Optional cap for positive samples.")
    parser.add_argument("--non-poem-limit", type=int, default=0, help="Optional cap for negative samples.")
    parser.add_argument("--max-char-vocab", type=int, default=3000)
    parser.add_argument("--min-char-freq", type=int, default=1)
    parser.add_argument("--max-bigram-vocab", type=int, default=2048)
    parser.add_argument("--min-bigram-freq", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--latent-dim", type=int, default=128, help="QSNN latent dimension, recommended 20~512.")
    parser.add_argument("--disable-bigrams", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--t-unitary", type=float, default=1.2)
    parser.add_argument("--t-dissipative", type=float, default=1.8)
    parser.add_argument("--stage2-steps", type=int, default=16)
    parser.add_argument("--latent-penalty", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=str, default="outputs/poem_recognition/chinese_qsnn")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 20 <= args.latent_dim <= 512:
        raise ValueError("--latent-dim should stay within the requested 20~512 range.")
    if args.max_len < 8:
        raise ValueError("--max-len should be at least 8 for Chinese poetry lines.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train-ratio + --val-ratio must be < 1.")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    poem_path = Path(args.poem_data)
    poem_texts = load_corpus_texts(poem_path, limit=args.poem_limit or None)
    if not poem_texts:
        raise RuntimeError(f"No positive samples found in: {poem_path}")

    if args.non_poem_data:
        non_poem_texts = load_corpus_texts(Path(args.non_poem_data), limit=args.non_poem_limit or None)
        if not non_poem_texts:
            raise RuntimeError(f"No negative samples found in: {args.non_poem_data}")
    else:
        # 没有真实负样本时，自动从诗句语料构造一份训练用负样本。
        non_poem_texts = build_synthetic_negative_corpus(poem_texts, seed=args.seed + 17)

    target_negatives = min(len(non_poem_texts), len(poem_texts))
    non_poem_texts = non_poem_texts[:target_negatives]
    poem_texts = poem_texts[:target_negatives]

    examples = build_examples(poem_texts, non_poem_texts)
    train_examples, val_examples, test_examples = split_dataset(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_texts = [ex.text for ex in train_examples]
    char_vocab, char_items = build_vocab(
        train_texts,
        max_size=args.max_char_vocab,
        min_freq=args.min_char_freq,
        reserved_tokens=["<PAD>", "<UNK>"],
        ngram=1,
    )
    bigram_vocab, bigram_items = build_vocab(
        train_texts,
        max_size=args.max_bigram_vocab,
        min_freq=args.min_bigram_freq,
        reserved_tokens=["<PAD>", "<UNK>"],
        ngram=2,
    )

    train_dataset = PoetryBinaryDataset(train_examples, char_vocab, bigram_vocab, args.max_len)
    val_dataset = PoetryBinaryDataset(val_examples, char_vocab, bigram_vocab, args.max_len)
    test_dataset = PoetryBinaryDataset(test_examples, char_vocab, bigram_vocab, args.max_len)

    # DataLoader 负责打乱、分 batch，并调用 dataset.collate 做动态 padding。
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate,
    )

    model = ChinesePoemQSNN(
        vocab_size=len(char_vocab),
        bigram_vocab_size=len(bigram_vocab),
        latent_dim=args.latent_dim,
        max_len=args.max_len,
        t_unitary=args.t_unitary,
        t_dissipative=args.t_dissipative,
        stage2_steps=args.stage2_steps,
        dropout=args.dropout,
        use_bigrams=not args.disable_bigrams,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    print(f"device={device}")
    print(
        "dataset "
        f"train={len(train_examples)} val={len(val_examples)} test={len(test_examples)} "
        f"char_vocab={len(char_vocab)} bigram_vocab={len(bigram_vocab)} latent_dim={args.latent_dim}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_count = 0

        for batch in train_loader:
            char_ids = batch["char_ids"].to(device)
            bigram_ids = batch["bigram_ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(char_ids, bigram_ids, mask)
            # 除了分类损失，还额外约束 latent_mass，
            # 鼓励模型把更多概率读出到两个输出节点。
            loss = F.nll_loss(out["logits"], labels) + args.latent_penalty * out["latent_mass"].mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            preds = out["probs"].argmax(dim=-1)
            epoch_correct += int((preds == labels).sum().item())
            epoch_loss += float(loss.item()) * labels.numel()
            epoch_count += labels.numel()

        train_metrics = {
            "loss": epoch_loss / max(epoch_count, 1),
            "accuracy": epoch_correct / max(epoch_count, 1),
        }
        val_metrics = evaluate(model, val_loader, device=device, latent_penalty=args.latent_penalty)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_latent_mass": val_metrics["latent_mass"],
            }
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_latent_mass={val_metrics['latent_mass']:.4f}"
        )

        if val_metrics["accuracy"] >= best_val_acc:
            # 始终保留验证集表现最好的那组参数。
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device=device, latent_penalty=args.latent_penalty)
    preview = collect_preview_predictions(
        model=model,
        examples=test_examples,
        dataset=test_dataset,
        device=device,
        limit=min(16, len(test_examples)),
    )

    print(
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_latent_mass={test_metrics['latent_mass']:.4f}"
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "char_vocab": char_vocab,
            "bigram_vocab": bigram_vocab,
            "args": vars(args),
            "history": history,
            "test_metrics": test_metrics,
        },
        outdir / "model.pt",
    )

    # 除模型外，也把配置、词表、指标和若干预测样例一起保存，方便复现实验。
    with (outdir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    with (outdir / "vocab.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "char_vocab": char_vocab,
                "bigram_vocab": bigram_vocab,
                "top_chars": char_items[:100],
                "top_bigrams": bigram_items[:100],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    with (outdir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_val_accuracy": best_val_acc,
                "test_metrics": test_metrics,
                "history": history,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    with (outdir / "predictions_preview.jsonl").open("w", encoding="utf-8") as handle:
        for row in preview:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
