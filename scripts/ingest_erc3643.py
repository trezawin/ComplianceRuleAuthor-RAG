import argparse
import json
from pathlib import Path
from typing import List

from rich.console import Console

from src.config import paths
from src.ingest import clean

console = Console()


def chunk_paragraphs(text: str, max_tokens: int = 500) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = []
    buf_tokens = 0
    for p in paras:
        tokens = p.split()
        if buf_tokens + len(tokens) <= max_tokens:
            buf.append(p)
            buf_tokens += len(tokens)
        else:
            if buf:
                chunks.append("\n\n".join(buf))
            buf = [p]
            buf_tokens = len(tokens)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def ingest_erc(input_path: Path, out_path: Path, max_tokens: int = 500):
    if not input_path.exists():
        raise FileNotFoundError(f"ERC-3643 source not found: {input_path}")
    raw = input_path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean(raw)
    texts = chunk_paragraphs(cleaned, max_tokens=max_tokens)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, t in enumerate(texts):
            payload = {
                "id": idx,
                "text": t,
                "section": None,
                "heading": None,
                "division": None,
                "subdivision": None,
                "source": "erc_3643_spec",
                "tokens": len(t.split()),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    console.print(f"[bold green]Wrote {len(texts)} ERC-3643 chunks to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Ingest ERC-3643 spec text into chunks.")
    ap.add_argument("--source", type=Path, default=paths.erc_raw, help="Path to ERC-3643 text file")
    ap.add_argument("--out", type=Path, default=paths.erc_chunks, help="Output JSONL path")
    ap.add_argument("--max_tokens", type=int, default=500, help="Max tokens per chunk")
    args = ap.parse_args()
    ingest_erc(args.source, args.out, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
