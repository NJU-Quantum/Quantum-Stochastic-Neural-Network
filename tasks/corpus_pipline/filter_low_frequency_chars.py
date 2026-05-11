from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


HAN_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
DEFAULT_INPUT_PATH = Path("tasks/corpus_pipline/tang_selected.cleaned.jsonl")


@dataclass
class FilterStats:
    records_seen: int = 0
    records_kept: int = 0
    records_dropped: int = 0
    characters_before: int = 0
    characters_after: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Second-pass cleaning for JSONL corpora: drop samples whose text contains "
            "any low-frequency characters."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Input JSONL file from the first-pass cleaner. If omitted, uses "
            f"{DEFAULT_INPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL path. Defaults to <input_stem>.lf<bottom_k>.filtered.jsonl",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        help="Stats JSON path. Defaults to <input_stem>.lf<bottom_k>.stats.json",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="text",
        help="Record field to inspect and filter on. Default: text",
    )
    parser.add_argument(
        "--bottom-k",
        type=int,
        default=4000,
        help="Number of lowest-frequency characters to treat as low-frequency. Default: 1000",
    )
    parser.add_argument(
        "--han-only",
        action="store_true",
        default=True,
        help="Only count/filter CJK Han characters. Default: enabled",
    )
    parser.add_argument(
        "--include-non-han",
        action="store_true",
        help="If set, allow all visible characters into the low-frequency pool instead of Han-only.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of high-frequency characters to keep in output stats. Default: 100",
    )
    args = parser.parse_args()

    if args.bottom_k < 1:
        parser.error("--bottom-k must be >= 1.")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1.")
    if args.include_non_han:
        args.han_only = False
    return args


def default_output_paths(input_path: Path, bottom_k: int) -> tuple[Path, Path]:
    stem = input_path.stem
    return (
        Path(f"{stem}.lf{bottom_k}.filtered.jsonl"),
        Path(f"{stem}.lf{bottom_k}.stats.json"),
    )


def should_count_char(char: str, han_only: bool) -> bool:
    if char.isspace():
        return False
    if han_only:
        return HAN_CHAR_RE.match(char) is not None
    return True


def iter_jsonl_records(input_path: Path):
    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {input_path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_no} is not a JSON object in {input_path}")
            yield record


def collect_char_counts(input_path: Path, field: str, han_only: bool) -> tuple[Counter[str], int]:
    counter: Counter[str] = Counter()
    total_chars = 0

    for record in iter_jsonl_records(input_path):
        text = str(record.get(field, "") or "")
        for char in text:
            if should_count_char(char, han_only):
                counter[char] += 1
                total_chars += 1

    return counter, total_chars


def select_low_frequency_chars(counter: Counter[str], bottom_k: int) -> list[tuple[str, int]]:
    sorted_chars = sorted(counter.items(), key=lambda item: (item[1], item[0]))
    return sorted_chars[: min(bottom_k, len(sorted_chars))]


def record_contains_low_frequency_char(text: str, low_chars: set[str], han_only: bool) -> bool:
    for char in text:
        if should_count_char(char, han_only) and char in low_chars:
            return True
    return False


def write_stats(
    stats_path: Path,
    input_path: Path,
    output_path: Path,
    stats: FilterStats,
    args: argparse.Namespace,
    vocab_before: int,
    vocab_after: int,
    low_frequency_items: list[tuple[str, int]],
    after_counter: Counter[str],
) -> None:
    avg_len_before = stats.characters_before / stats.records_seen if stats.records_seen else 0.0
    avg_len_after = stats.characters_after / stats.records_kept if stats.records_kept else 0.0

    payload = {
        "input_file": str(input_path.resolve()),
        "output_file": str(output_path.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "field": args.field,
        "bottom_k": args.bottom_k,
        "han_only": args.han_only,
        "records_seen": stats.records_seen,
        "records_kept": stats.records_kept,
        "records_dropped": stats.records_dropped,
        "characters_before": stats.characters_before,
        "characters_after": stats.characters_after,
        "vocab_size_before": vocab_before,
        "vocab_size_after": vocab_after,
        "average_length_before": round(avg_len_before, 4),
        "average_length_after": round(avg_len_after, 4),
        "low_frequency_characters": [
            {"char": char, "count": count} for char, count in low_frequency_items
        ],
        "high_frequency_characters_after": [
            {"char": char, "count": count}
            for char, count in after_counter.most_common(args.top_k)
        ],
    }
    stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    default_output, default_stats = default_output_paths(input_path, args.bottom_k)
    output_path = (args.output or default_output).resolve()
    stats_path = (args.stats or default_stats).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    before_counter, total_chars_before = collect_char_counts(input_path, args.field, args.han_only)
    low_frequency_items = select_low_frequency_chars(before_counter, args.bottom_k)
    low_frequency_chars = {char for char, _ in low_frequency_items}

    stats = FilterStats(characters_before=total_chars_before)
    after_counter: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as writer:
        for record in iter_jsonl_records(input_path):
            stats.records_seen += 1
            text = str(record.get(args.field, "") or "")

            if record_contains_low_frequency_char(text, low_frequency_chars, args.han_only):
                stats.records_dropped += 1
                continue

            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats.records_kept += 1
            for char in text:
                if should_count_char(char, args.han_only):
                    after_counter[char] += 1
                    stats.characters_after += 1

    write_stats(
        stats_path=stats_path,
        input_path=input_path,
        output_path=output_path,
        stats=stats,
        args=args,
        vocab_before=len(before_counter),
        vocab_after=len(after_counter),
        low_frequency_items=low_frequency_items,
        after_counter=after_counter,
    )

    print(f"Filtered corpus written to: {output_path}")
    print(f"Stats written to: {stats_path}")


if __name__ == "__main__":
    main()
