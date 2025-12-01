import argparse
import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def normalize_line(s: str) -> str:
    # Collapse whitespace and strip simple punctuation for loose matching.
    s = re.sub(r"[\\.\\-–—•·,:;()\\[\\]]", " ", s)
    s = re.sub(r"\\s+", " ", s)
    return s.strip().lower()


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_chunks(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser(description="Check coverage of chunks vs. PDF source.")
    ap.add_argument("--pdf", type=Path, default=Path("data/raw/cap-615.pdf"), help="Path to source PDF")
    ap.add_argument("--chunks", type=Path, default=Path("data/processed/cap615.jsonl"), help="Path to chunks JSONL")
    ap.add_argument("--show_missing", type=int, default=10, help="Show first N missing lines")
    ap.add_argument("--min_line_len", type=int, default=40, help="Minimum line length to consider for missing check")
    args = ap.parse_args()

    raw_text = clean(read_pdf(args.pdf))
    chunks = load_chunks(args.chunks)
    covered_text = "\n".join(c["text"] for c in chunks)
    covered_norm = normalize_line(covered_text)
    covered_tokens = set(covered_norm.split())

    # Stats
    total_chunks = len(chunks)
    total_tokens = sum(c.get("tokens", len(c["text"].split())) for c in chunks)
    sec_counts = Counter((c.get("section"), c.get("heading")) for c in chunks)

    print(f"Chunks: {total_chunks}")
    print(f"Total tokens (approx words): {total_tokens}")
    print(f"Unique section/headings: {len(sec_counts)}")
    for (sec, head), n in sec_counts.most_common():
        print(f"{n} chunk(s): section={sec!r}, heading={head!r}")

    # Missing line heuristic
    missing = []
    for line in raw_text.splitlines():
        line = line.strip()
        if len(line) < args.min_line_len:
            continue
        nline = normalize_line(line)
        tokens = set(nline.split())
        if nline not in covered_norm and not tokens.issubset(covered_tokens):
            missing.append(line)

    print(f"\nPotential missing lines (len>={args.min_line_len}): {len(missing)}")
    for line in missing[: args.show_missing]:
        print("-", line)


if __name__ == "__main__":
    main()
