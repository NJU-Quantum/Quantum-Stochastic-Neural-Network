import re
from pathlib import Path


INPUT_PATH = Path("tasks/poem_recognition/TS.txt.txt")
OUTPUT_PATH = Path("tasks/poem_recognition/TS_modern_english.txt")


def apply_case(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src.istitle():
        return dst.title()
    if src and src[0].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


def replace_words(text: str, mapping: dict[str, str], counters: dict[str, int]) -> str:
    keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b", re.IGNORECASE)

    def repl(m: re.Match[str]) -> str:
        src = m.group(0)
        dst_base = mapping[m.group(1).lower()]
        dst = apply_case(src, dst_base)
        counters[src.lower()] = counters.get(src.lower(), 0) + 1
        return dst

    return pattern.sub(repl, text)


def replace_contractions(text: str, mapping: dict[str, str], counters: dict[str, int]) -> str:
    # Contractions include apostrophes and should not rely on strict word boundary only.
    keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)", re.IGNORECASE)

    def repl(m: re.Match[str]) -> str:
        src = m.group(0)
        dst_base = mapping[m.group(1).lower()]
        dst = apply_case(src, dst_base)
        counters[src.lower()] = counters.get(src.lower(), 0) + 1
        return dst

    return pattern.sub(repl, text)


def convert_eth_form(word: str) -> str:
    lower = word.lower()
    base = lower[:-3]
    if not base:
        return word
    if base.endswith(("s", "x", "z", "ch", "sh", "o")):
        modern = base + "es"
    else:
        modern = base + "s"
    return apply_case(word, modern)


def replace_eth_verbs(text: str, counters: dict[str, int]) -> str:
    # Only convert likely verb tokens ending in -eth (avoid very short stems).
    pattern = re.compile(r"\b([A-Za-z]{4,}eth)\b")

    def repl(m: re.Match[str]) -> str:
        src = m.group(1)
        # Skip known words that are not archaic verb forms.
        if src.lower() in {"teeth", "beneath"}:
            return src
        dst = convert_eth_form(src)
        if dst != src:
            counters["-eth"] = counters.get("-eth", 0) + 1
        return dst

    return pattern.sub(repl, text)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    text = INPUT_PATH.read_text(encoding="utf-8")
    counters: dict[str, int] = {}

    contraction_mapping = {
        "know't": "know it",
        "is't": "is it",
        "on't": "on it",
        "'tis": "it is",
        "'twas": "it was",
        "'twere": "it were",
        "'twere": "it were",
        "o'er": "over",
        "e'er": "ever",
        "ne'er": "never",
        "'twixt": "between",
        "i'th'": "in the",
        "i'th": "in the",
    }

    word_mapping = {
        "thou": "you",
        "thee": "you",
        "thy": "your",
        "thine": "yours",
        "ye": "you",
        "art": "are",
        "wert": "were",
        "wast": "were",
        "hast": "have",
        "hath": "has",
        "dost": "do",
        "doth": "does",
        "didst": "did",
        "shalt": "will",
        "shouldst": "should",
        "wouldst": "would",
        "canst": "can",
        "couldst": "could",
        "mayst": "may",
        "mightst": "might",
        "whilst": "while",
        "amongst": "among",
        "betwixt": "between",
        "oft": "often",
        "nay": "no",
        "unto": "to",
        "ere": "before",
    }

    modern = replace_contractions(text, contraction_mapping, counters)
    modern = replace_words(modern, word_mapping, counters)
    modern = replace_eth_verbs(modern, counters)

    OUTPUT_PATH.write_text(modern, encoding="utf-8")

    total_replacements = sum(counters.values())
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Total replacements: {total_replacements}")
    print("Top replacements:")
    for k, v in sorted(counters.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {k} -> {v}")


if __name__ == "__main__":
    main()
