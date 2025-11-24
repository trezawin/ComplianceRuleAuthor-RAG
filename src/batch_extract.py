import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

from .config import paths
from .pipeline import call_llm
from .prompts import build_prompt

console = Console()


def load_chunks(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def group_by_section(chunks: List[Dict], max_tokens: int) -> List[Dict]:
    """Combine contiguous chunks from the same section into contexts capped by token count."""
    grouped = []
    buffer: List[Dict] = []
    buffer_tokens = 0

    def flush():
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        section = buffer[0].get("section")
        heading = buffer[0].get("heading")
        texts = [c["text"] for c in buffer]
        context = "\n\n---\n\n".join(texts)
        grouped.append({"section": section, "heading": heading, "context": context})
        buffer = []
        buffer_tokens = 0

    for chunk in chunks:
        tokens = chunk["tokens"]
        if not buffer:
            buffer = [chunk]
            buffer_tokens = tokens
            continue
        same_section = chunk.get("section") == buffer[-1].get("section")
        if same_section and buffer_tokens + tokens <= max_tokens:
            buffer.append(chunk)
            buffer_tokens += tokens
        else:
            flush()
            buffer = [chunk]
            buffer_tokens = tokens
    flush()
    return grouped


def parse_json_response(raw: Optional[str]) -> List[Dict]:
    if not raw:
        return []
    # Strip simple fences if present.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        return [data]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Extract obligations across all sections and aggregate JSON.")
    parser.add_argument("--chunks", type=Path, default=paths.default_chunks, help="Path to JSONL chunks")
    parser.add_argument("--out", type=Path, default=paths.processed_dir / "all_rules.json", help="Output JSON file")
    parser.add_argument("--max_context_tokens", type=int, default=1500, help="Max tokens per grouped context")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick runs")
    parser.add_argument("--dry_run", action="store_true", help="Skip LLM calls; emit prompts only")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    grouped = group_by_section(chunks, max_tokens=args.max_context_tokens)
    if args.limit:
        grouped = grouped[: args.limit]
    console.print(f"[cyan]Processing {len(grouped)} grouped contexts...")

    all_rules: List[Dict] = []
    for idx, item in enumerate(grouped, 1):
        context = item["context"]
        section_label = item.get("section") or f"group-{idx}"
        query = "Extract enforceable obligations in this section."
        prompt = build_prompt(context=context, query=query)
        if args.dry_run:
            console.print(f"[yellow]Dry run for section {section_label}; prompt length={len(prompt)}")
            continue
        resp = call_llm(prompt)
        rules = parse_json_response(resp)
        for rule in rules:
            rule["section"] = item.get("section")
            rule["heading"] = item.get("heading")
        all_rules.extend(rules)
        if idx % 10 == 0:
            console.print(f"[green]Processed {idx}/{len(grouped)} contexts; total rules: {len(all_rules)}")

    if args.dry_run:
        console.print("[yellow]Dry run complete; no output file written.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_rules, indent=2, ensure_ascii=False))
    console.print(f"[bold green]Wrote {len(all_rules)} rules to {args.out}")


if __name__ == "__main__":
    main()
