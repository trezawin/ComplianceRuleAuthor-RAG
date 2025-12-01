import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import argparse
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from rich.console import Console

from .config import paths, settings

console = Console()


def load_chunks(chunks_path: Path) -> List[Dict]:
    with chunks_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    console.print(f"[cyan]Loading embedding model: {model_name} on {settings.device}")
    model = SentenceTransformer(model_name, device=settings.device)
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)


def build_bm25(chunks: List[Dict]) -> Tuple[BM25Okapi, List[List[str]]]:
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def index(chunks_path: Path, index_path: Path, bm25_path: Path, extra_paths: List[Path] = None) -> None:
    console.print(f"[cyan]Loading chunks from {chunks_path}")
    chunks = load_chunks(chunks_path)
    extra_paths = extra_paths or []
    for p in extra_paths:
        if p.exists():
            console.print(f"[cyan]Appending extra chunks from {p}")
            chunks.extend(load_chunks(p))
        else:
            console.print(f"[yellow]Extra chunks file not found: {p}")
    texts = [c["text"] for c in chunks]

    embeddings = build_embeddings(texts, settings.embedding_model)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(index_path, embeddings)
    console.print(f"[bold green]Saved embeddings to {index_path}")

    # Save metadata for retrieval
    meta_path = index_path.with_suffix(".meta.jsonl")
    with meta_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    console.print(f"[bold green]Saved metadata to {meta_path}")

    # BM25
    bm25, tokenized_corpus = build_bm25(chunks)
    with bm25_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "corpus": tokenized_corpus, "chunks": chunks}, f)
    console.print(f"[bold green]Saved BM25 store to {bm25_path}")


def main():
    parser = argparse.ArgumentParser(description="Build embedding + BM25 indexes for Cap. 615 chunks.")
    parser.add_argument("--chunks", type=Path, default=paths.default_chunks, help="JSONL chunks produced by ingest")
    parser.add_argument("--index_out", type=Path, default=paths.default_index, help="Embeddings .npy path")
    parser.add_argument("--bm25_out", type=Path, default=paths.bm25_store, help="BM25 pickle path")
    parser.add_argument("--extra", type=Path, nargs="*", default=[], help="Optional extra chunk files to include (e.g., ERC-3643 spec)")
    args = parser.parse_args()
    index(args.chunks, args.index_out, args.bm25_out, extra_paths=args.extra)


if __name__ == "__main__":
    main()
