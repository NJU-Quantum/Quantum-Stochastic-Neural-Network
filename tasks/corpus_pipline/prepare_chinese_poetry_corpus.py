from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


URL_RE = re.compile(r"https?://[^\s<>\]]+|www\.[^\s<>\]]+")
WHITESPACE_RE = re.compile(r"\s+")
HAN_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([，。！？；：、）》】」』])")
SPACE_AFTER_OPEN_PUNCT_RE = re.compile(r"([（《【「『])\s+")


@dataclass
class PoetryStats:
    records_seen: int = 0
    records_kept: int = 0
    paragraphs_written: int = 0
    characters_written: int = 0
    dropped_author: int = 0
    dropped_empty: int = 0
    dropped_short: int = 0
    dropped_low_chinese: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a smaller Chinese poetry corpus into paragraph-level JSONL for LLM pretraining."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a local poetry dataset file or directory. Supported: .json, .jsonl, .txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL path. Defaults to <input_stem>.poetry.cleaned.jsonl",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        help="Stats JSON path. Defaults to <input_stem>.poetry.stats.json",
    )
    parser.add_argument(
        "--min-chinese-ratio",
        type=float,
        default=0.80,
        help="Minimum Chinese-character ratio required to keep a sample. Default: 0.80",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=4,
        help="Minimum sample length after cleaning. Default: 4",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of frequent characters to include in stats. Default: 100",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for quick verification.",
    )
    parser.add_argument(
        "--join-paragraphs",
        action="store_true",
        help="If set, join multi-line poem paragraphs into one training sample per poem.",
    )
    parser.add_argument(
        "--authors",
        type=str,
        default=None,
        help="Optional comma-separated author allowlist. Only poems from these authors will be kept.",
    )
    parser.add_argument(
        "--authors-file",
        type=Path,
        default=None,
        help="Optional UTF-8 text file containing one allowed author per line.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.min_chinese_ratio <= 1.0:
        parser.error("--min-chinese-ratio must be between 0 and 1.")
    if args.min_chars < 1:
        parser.error("--min-chars must be >= 1.")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1.")
    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records must be >= 1.")
    return args


def build_converter():
    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'opencc'. Install it with "
            "'pip install -r tasks/corpus_pipline/requirements.txt'."
        ) from exc
    return OpenCC("t2s")


def default_output_paths(input_path: Path) -> tuple[Path, Path]:
    stem = input_path.stem if input_path.is_file() else input_path.name
    return Path(f"{stem}.poetry.cleaned.jsonl"), Path(f"{stem}.poetry.stats.json")


def parse_author_allowlist(args: argparse.Namespace, converter) -> set[str] | None:
    allowed: set[str] = set()
    if args.authors:
        for name in args.authors.split(","):
            normalized = normalize_text(name, converter)
            if normalized:
                allowed.add(normalized)

    if args.authors_file:
        for raw_line in args.authors_file.read_text(encoding="utf-8-sig").splitlines():
            normalized = normalize_text(raw_line, converter)
            if normalized:
                allowed.add(normalized)

    return allowed or None


def iter_input_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return

    for path in sorted(input_path.rglob("*")):
        if path.suffix.lower() in {".json", ".jsonl", ".txt"} and path.is_file():
            yield path


def chinese_ratio(text: str) -> float:
    visible_chars = [char for char in text if not char.isspace()]
    if not visible_chars:
        return 0.0
    han_count = sum(1 for char in visible_chars if HAN_CHAR_RE.match(char))
    return han_count / len(visible_chars)


def normalize_text(text: str, converter) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = URL_RE.sub(" ", text)
    text = converter.convert(text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = SPACE_AFTER_OPEN_PUNCT_RE.sub(r"\1", text)
    return text


def clean_lines(lines: Iterable[str], converter) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        text = normalize_text(line, converter)
        if text:
            cleaned.append(text)
    return cleaned


def extract_poem_text_fields(item: object) -> tuple[str, str, list[str]] | None:
    if isinstance(item, str):
        return "", "", [item]

    if not isinstance(item, dict):
        return None

    title = str(item.get("title") or item.get("rhythmic") or item.get("name") or "").strip()
    author = str(item.get("author") or item.get("writer") or item.get("poet") or "").strip()

    paragraphs = item.get("paragraphs")
    if isinstance(paragraphs, list):
        lines = [str(part).strip() for part in paragraphs if str(part).strip()]
        return title, author, lines

    content = item.get("content") or item.get("text") or item.get("body")
    if isinstance(content, list):
        lines = [str(part).strip() for part in content if str(part).strip()]
        return title, author, lines
    if isinstance(content, str):
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        return title, author, lines

    return None


def iter_poems_from_json(path: Path) -> Iterable[tuple[str, str, list[str], str]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))

    if isinstance(data, list):
        for item in data:
            parsed = extract_poem_text_fields(item)
            if parsed is not None:
                title, author, lines = parsed
                yield title, author, lines, path.name
        return

    if isinstance(data, dict):
        containers = []
        for key in ("poems", "data", "items", "records"):
            value = data.get(key)
            if isinstance(value, list):
                containers.append(value)

        if containers:
            for container in containers:
                for item in container:
                    parsed = extract_poem_text_fields(item)
                    if parsed is not None:
                        title, author, lines = parsed
                        yield title, author, lines, path.name
            return

        parsed = extract_poem_text_fields(data)
        if parsed is not None:
            title, author, lines = parsed
            yield title, author, lines, path.name


def iter_poems_from_jsonl(path: Path) -> Iterable[tuple[str, str, list[str], str]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parsed = extract_poem_text_fields(json.loads(line))
            if parsed is None:
                continue
            title, author, lines = parsed
            yield title, author, lines, path.name


def iter_poems_from_txt(path: Path) -> Iterable[tuple[str, str, list[str], str]]:
    text = path.read_text(encoding="utf-8-sig")
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        title = ""
        author = ""
        poem_lines = lines

        if len(lines) >= 2 and len(lines[0]) <= 20 and len(lines[1]) <= 20:
            title = lines[0]
            author = lines[1]
            poem_lines = lines[2:] or lines[1:]

        yield title, author, poem_lines, path.name


def iter_poems(input_path: Path) -> Iterable[tuple[str, str, list[str], str]]:
    for path in iter_input_files(input_path):
        suffix = path.suffix.lower()
        if suffix == ".json":
            yield from iter_poems_from_json(path)
        elif suffix == ".jsonl":
            yield from iter_poems_from_jsonl(path)
        elif suffix == ".txt":
            yield from iter_poems_from_txt(path)


def build_samples(
    lines: list[str],
    converter,
    join_paragraphs: bool,
    min_chars: int,
    min_chinese_ratio: float,
    stats: PoetryStats,
) -> list[str]:
    cleaned_lines = clean_lines(lines, converter)
    candidates = [" ".join(cleaned_lines)] if join_paragraphs else cleaned_lines

    samples: list[str] = []
    for text in candidates:
        text = text.strip()
        if not text:
            stats.dropped_empty += 1
            continue
        if len(text) < min_chars:
            stats.dropped_short += 1
            continue
        if chinese_ratio(text) < min_chinese_ratio:
            stats.dropped_low_chinese += 1
            continue
        samples.append(text)
    return samples


def write_stats(
    stats_path: Path,
    input_path: Path,
    output_path: Path,
    stats: PoetryStats,
    char_counter: Counter[str],
    author_counter: Counter[str],
    allowed_authors: set[str] | None,
    args: argparse.Namespace,
) -> None:
    avg_length = (
        stats.characters_written / stats.paragraphs_written
        if stats.paragraphs_written
        else 0.0
    )
    payload = {
        "input_path": str(input_path.resolve()),
        "output_file": str(output_path.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_seen": stats.records_seen,
        "records_kept": stats.records_kept,
        "paragraph_count": stats.paragraphs_written,
        "character_count": stats.characters_written,
        "vocab_size": len(char_counter),
        "average_paragraph_length": round(avg_length, 4),
        "dropped_author": stats.dropped_author,
        "dropped_empty": stats.dropped_empty,
        "dropped_short": stats.dropped_short,
        "dropped_low_chinese": stats.dropped_low_chinese,
        "filters": {
            "min_chinese_ratio": args.min_chinese_ratio,
            "min_chars": args.min_chars,
            "join_paragraphs": args.join_paragraphs,
            "author_filter_enabled": allowed_authors is not None,
            "allowed_authors": sorted(allowed_authors) if allowed_authors else [],
        },
        "author_counts": [
            {"author": author, "count": count}
            for author, count in author_counter.most_common()
        ],
        "high_frequency_characters": [
            {"char": char, "count": count}
            for char, count in char_counter.most_common(args.top_k)
        ],
    }
    stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    default_output, default_stats = default_output_paths(input_path)
    output_path = (args.output or default_output).resolve()
    stats_path = (args.stats or default_stats).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    converter = build_converter()
    allowed_authors = parse_author_allowlist(args, converter)
    stats = PoetryStats()
    char_counter: Counter[str] = Counter()
    author_counter: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as writer:
        for record_index, (title, author, lines, source_name) in enumerate(iter_poems(input_path), start=1):
            if args.max_records is not None and record_index > args.max_records:
                break

            stats.records_seen += 1
            samples = build_samples(
                lines=lines,
                converter=converter,
                join_paragraphs=args.join_paragraphs,
                min_chars=args.min_chars,
                min_chinese_ratio=args.min_chinese_ratio,
                stats=stats,
            )
            if not samples:
                continue

            normalized_title = normalize_text(title, converter)
            normalized_author = normalize_text(author, converter)
            if allowed_authors is not None and normalized_author not in allowed_authors:
                stats.dropped_author += 1
                continue

            stats.records_kept += 1
            author_counter.update([normalized_author or "<unknown>"])

            for sample_index, sample in enumerate(samples, start=1):
                record = {
                    "id": f"{source_name}:{record_index}#{sample_index}",
                    "title": normalized_title,
                    "author": normalized_author,
                    "text": sample,
                    "source": source_name,
                    "genre": "chinese_poetry",
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.paragraphs_written += 1
                stats.characters_written += len(sample)
                char_counter.update(char for char in sample if not char.isspace())

    write_stats(stats_path, input_path, output_path, stats, char_counter, author_counter, allowed_authors, args)
    print(f"Cleaned poetry corpus written to: {output_path}")
    print(f"Stats written to: {stats_path}")


if __name__ == "__main__":
    main()
