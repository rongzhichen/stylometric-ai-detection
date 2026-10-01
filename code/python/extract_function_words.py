#!/usr/bin/env python3
"""Turn a directory of essays into function-word count vectors.

Input layout:  <input>/<topic>/<essay>.txt
Output layout: <output>/<topic>/<essay>.txt --- functionwords.txt
Each output file holds one comma-separated row: counts of the 70 function words in
`data/function_words.txt` (in file order) followed by the count of all other words.
This is the 71-dimensional representation used throughout the paper.

Examples
  # full-length essays
  python3 code/python/extract_function_words.py --input path/to/humanessays --output data/function_words/human
  # first 200 words only (Study 2)
  python3 code/python/extract_function_words.py --input path/to/humanessays --output data/function_words_200/human --max-words 200
"""
from __future__ import annotations

import argparse
import string
from collections import Counter
from pathlib import Path

_PUNCT = str.maketrans("", "", string.punctuation)


def load_function_words(path: Path) -> list[str]:
    return [line.split(",")[0].strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize(text: str) -> list[str]:
    """Lower-case, printable-ASCII, punctuation-free tokens, in document order."""
    text = text.replace("\r", " ").replace("\n", " ")
    text = "".join(c for c in text if 31 < ord(c) < 127)
    text = text.translate(_PUNCT).lower()
    return [w for w in text.split() if w]


def count_vector(words: list[str], function_words: list[str], max_words: int | None) -> list[int]:
    if max_words is not None:
        words = words[:max_words]
    counts = Counter(words)
    vec = [counts[w] for w in function_words]
    vec.append(len(words) - sum(vec))  # non-function words
    return vec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="directory of <topic>/<essay>.txt")
    ap.add_argument("--output", required=True, type=Path, help="destination directory (created if missing)")
    ap.add_argument("--function-words", type=Path, default=Path("data/function_words.txt"))
    ap.add_argument("--max-words", type=int, default=None, help="truncate each essay to its first N words")
    args = ap.parse_args()

    fws = load_function_words(args.function_words)
    n = 0
    for topic_dir in sorted(p for p in args.input.iterdir() if p.is_dir()):
        out_dir = args.output / topic_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for essay in sorted(topic_dir.glob("*.txt")):
            words = tokenize(essay.read_text(encoding="utf-8", errors="replace"))
            vec = count_vector(words, fws, args.max_words)
            (out_dir / f"{essay.name} --- functionwords.txt").write_text(", ".join(map(str, vec)) + "\n")
            n += 1
    print(f"wrote {n} feature files to {args.output}")


if __name__ == "__main__":
    main()
