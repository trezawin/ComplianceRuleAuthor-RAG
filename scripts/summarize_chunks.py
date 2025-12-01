import json
from collections import Counter
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Summarize chunk counts and sections.")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/cap615.jsonl"), help="Path to chunks JSONL")
    parser.add_argument("--show", type=int, default=0, help="Show first N chunks (JSON)")
    args = parser.parse_args()

    chunks_path = args.chunks
    chunks = [json.loads(line) for line in chunks_path.open()]
    print(f"Total chunks: {len(chunks)}")

    sec_counts = Counter((c.get("section"), c.get("heading")) for c in chunks)
    print(f"Unique sections/headings: {len(sec_counts)}")
    for (sec, head), n in sec_counts.most_common():
        print(f"{n} chunk(s): section={sec!r}, heading={head!r}")

    if args.show > 0:
        print("\nSample chunks:")
        for c in chunks[: args.show]:
            print(json.dumps(c, ensure_ascii=False))


if __name__ == "__main__":
    main()
