import json
import re
from collections import Counter
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*")
TOKEN_RE = re.compile(r"\n|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]")
NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "]", "}", "%", "'s"}
NO_SPACE_AFTER = {"(", "[", "{", '"'}


def pretokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def is_word_token(tok: str) -> bool:
    return WORD_RE.fullmatch(tok) is not None


def detokenize(tokens: list[str]) -> str:
    out: list[str] = []
    prev = ""
    for t in tokens:
        if t == "\n":
            if out and out[-1].endswith(" "):
                out[-1] = out[-1].rstrip(" ")
            out.append("\n")
            prev = "\n"
            continue

        if not out or prev == "\n" or t in NO_SPACE_BEFORE or prev in NO_SPACE_AFTER:
            out.append(t)
        else:
            out.append(" " + t)
        prev = t
    return "".join(out)


class BaseTokenizer:
    def __init__(self) -> None:
        self.token2id: dict[str, int] = {"<UNK>": 0}
        self.id2token: dict[int, str] = {0: "<UNK>"}

    def _build_vocab(self, tokens: list[str]) -> None:
        uniq = sorted(set(tokens))
        # Remove UNK if it's in the list, we'll add it explicitly
        uniq = [t for t in uniq if t != "<UNK>"]
        self.token2id = {"<UNK>": 0}
        for i, t in enumerate(uniq, start=1):
            self.token2id[t] = i
        self.id2token = {i: t for t, i in self.token2id.items()}

    def vocab_size(self) -> int:
        return len(self.token2id)

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        ids = [self.token2id.get(t, 0) for t in tokens]
        # Ensure all IDs are within valid range [0, vocab_size)
        vocab_size = len(self.token2id)
        ids = [min(id, vocab_size - 1) for id in ids]
        return ids

    def decode_ids(self, ids: list[int]) -> str:
        toks = [self.id2token.get(int(i), "<UNK>") for i in ids]
        return self.detokenize_tokens(toks)

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError

    def detokenize_tokens(self, tokens: list[str]) -> str:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        p = Path(path)
        payload = self.to_serializable()
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def to_serializable(self) -> dict:
        return {"type": self.__class__.__name__, "token2id": self.token2id}


class CharTokenizer(BaseTokenizer):
    def fit(self, text: str) -> None:
        self._build_vocab(list(text))

    def tokenize(self, text: str) -> list[str]:
        return list(text)

    def detokenize_tokens(self, tokens: list[str]) -> str:
        return "".join(tokens)


class WordPunctTokenizer(BaseTokenizer):
    def fit(self, text: str) -> None:
        tokens = pretokenize(text)
        self._build_vocab(tokens)

    def tokenize(self, text: str) -> list[str]:
        return pretokenize(text)

    def detokenize_tokens(self, tokens: list[str]) -> str:
        return detokenize(tokens)


class BPETokenizer(BaseTokenizer):
    def __init__(self, num_merges: int = 4000, min_pair_freq: int = 2) -> None:
        super().__init__()
        self.num_merges = num_merges
        self.min_pair_freq = min_pair_freq
        self.merges: list[tuple[str, str]] = []

    @staticmethod
    def _merge_once(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
        out: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                out.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return tuple(out)

    @staticmethod
    def _word_to_symbols(word: str) -> tuple[str, ...]:
        return tuple(list(word) + ["</w>"])

    def _learn_merges(self, words: list[str]) -> None:
        vocab = Counter(words)
        word_symbols: dict[str, tuple[str, ...]] = {w: self._word_to_symbols(w) for w in vocab}
        self.merges = []

        for _ in range(self.num_merges):
            pair_freq: Counter[tuple[str, str]] = Counter()
            for w, freq in vocab.items():
                syms = word_symbols[w]
                for i in range(len(syms) - 1):
                    pair_freq[(syms[i], syms[i + 1])] += freq

            if not pair_freq:
                break

            best_pair, best_freq = pair_freq.most_common(1)[0]
            if best_freq < self.min_pair_freq:
                break

            self.merges.append(best_pair)
            for w in word_symbols:
                word_symbols[w] = self._merge_once(word_symbols[w], best_pair)

        # Build final token inventory from merged word symbols + punctuation/newline tokens.
        final_symbols = set()
        for w in vocab:
            syms = self._apply_merges_to_word(w)
            final_symbols.update(syms)
        self._build_vocab(sorted(final_symbols))

    def _apply_merges_to_word(self, word: str) -> list[str]:
        syms = self._word_to_symbols(word)
        for pair in self.merges:
            syms = self._merge_once(syms, pair)
        return list(syms)

    def fit(self, text: str) -> None:
        base_tokens = pretokenize(text)
        words = [t for t in base_tokens if is_word_token(t)]
        punct = [t for t in base_tokens if not is_word_token(t)]

        self._learn_merges(words)
        # Ensure punctuation/newline tokens are in vocab.
        all_tokens = list(self.token2id.keys()) + punct
        self._build_vocab(all_tokens)

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for tok in pretokenize(text):
            if is_word_token(tok):
                out.extend(self._apply_merges_to_word(tok))
            else:
                out.append(tok)
        return out

    def detokenize_tokens(self, tokens: list[str]) -> str:
        rebuilt: list[str] = []
        cur = ""
        for t in tokens:
            if t == "<UNK>":
                if cur:
                    rebuilt.append(cur)
                    cur = ""
                rebuilt.append(t)
                continue
            if t == "\n" or (len(t) == 1 and not t.isalnum() and t not in {"</w>"}):
                if cur:
                    rebuilt.append(cur)
                    cur = ""
                rebuilt.append(t)
                continue
            if t.endswith("</w>"):
                cur += t[:-4]
                rebuilt.append(cur)
                cur = ""
            else:
                cur += t
        if cur:
            rebuilt.append(cur)
        return detokenize(rebuilt)

    def to_serializable(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "token2id": self.token2id,
            "num_merges": self.num_merges,
            "min_pair_freq": self.min_pair_freq,
            "merges": self.merges,
        }


def create_tokenizer(level: str, bpe_merges: int, bpe_min_pair_freq: int):
    if level == "char":
        return CharTokenizer()
    if level == "word":
        return WordPunctTokenizer()
    if level == "bpe":
        return BPETokenizer(num_merges=bpe_merges, min_pair_freq=bpe_min_pair_freq)
    raise ValueError(f"Unknown tokenizer level: {level}")

