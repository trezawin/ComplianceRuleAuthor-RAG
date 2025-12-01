import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from pypdf import PdfReader
from rich.console import Console
from striprtf.striprtf import rtf_to_text
import argparse

from .config import paths, settings

console = Console()


@dataclass
class Chunk:
    text: str
    section: Optional[str]
    heading: Optional[str]
    division: Optional[str]
    subdivision: Optional[str]
    source: str

    def to_json(self, idx: int) -> str:
        payload = {
            "id": idx,
            "text": self.text.strip(),
            "section": self.section,
            "heading": self.heading,
            "division": self.division,
            "subdivision": self.subdivision,
            "source": self.source,
            "tokens": len(self.text.split()),
        }
        return json.dumps(payload, ensure_ascii=False)


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n".join(pages)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_rtf(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return rtf_to_text(raw)


def clean(text: str) -> str:
    # Normalize spacing while preserving paragraph breaks; fix common PDF line artifacts.
    text = text.replace("\r", "")
    # Join hyphenated breaks (e.g., “cust-\nomer” -> “customer”).
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # Join linebreak-split words without hyphens (e.g., “cus\nstomer” -> “customer”).
    text = re.sub(r"(?<=\w)\n(?=\w)", " ", text)
    # Collapse extra spaces and blank lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


SECTION_PATTERN = re.compile(
    r"^\s*(?P<num>\d+[A-Z]*)\.\s*(?P<title>[^\n]{1,300})",
    re.MULTILINE,
)
SCHEDULE_PATTERN = re.compile(r"^\s*(Schedule\s+(?P<num>\d+))\s*(?P<title>[^\n]{1,300})", re.MULTILINE)
DIVISION_PATTERN = re.compile(r"^\s*(Division\s+\d+[A-Za-z0-9]*\s+[—\-–]\s*[^\n]+)", re.MULTILINE)
SUBDIVISION_PATTERN = re.compile(r"^\s*(Subdivision\s+\d+[A-Za-z0-9]*\s+[—\-–]\s*[^\n]+)", re.MULTILINE)


def split_sections(text: str) -> List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]]:
    """
    Split text into sections using Cap. 615-style numbering (e.g., 53ZRD.).
    Track Division/Subdivision headers for context.
    Returns tuples of (section_id, heading, body, division, subdivision).
    """
    matches = list(SECTION_PATTERN.finditer(text)) + list(SCHEDULE_PATTERN.finditer(text))
    matches = sorted(matches, key=lambda m: m.start())
    if not matches:
        return [(None, None, text, None, None)]

    divisions = sorted([(m.start(), m.group(1).strip()) for m in DIVISION_PATTERN.finditer(text)], key=lambda x: x[0])
    subdivisions = sorted([(m.start(), m.group(1).strip()) for m in SUBDIVISION_PATTERN.finditer(text)], key=lambda x: x[0])

    def latest_heading(pos: int, items: List[Tuple[int, str]]) -> Optional[str]:
        active = [h for start, h in items if start <= pos]
        return active[-1] if active else None

    slices: List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        header = match.groupdict().get("num") or match.groupdict().get("header") or ""
        heading_raw = match.groupdict().get("title") or ""
        section_id = header.strip().rstrip(".")
        heading = heading_raw.strip(" -–—.")

        div = latest_heading(match.start(), divisions)
        subdiv = latest_heading(match.start(), subdivisions)

        prefix_lines = []
        if div:
            prefix_lines.append(div)
        if subdiv:
            prefix_lines.append(subdiv)
        header_line = " ".join([p for p in [header, heading] if p])
        if header_line:
            prefix_lines.append(header_line)
        content = "\n".join(prefix_lines + [body]).strip()
        slices.append((section_id, heading, content, div, subdiv))
    return slices


def chunk_text(section_id: Optional[str], heading: Optional[str], text: str, division: Optional[str], subdivision: Optional[str]) -> Iterable[Chunk]:
    """Section-aware chunking with soft token sizing."""
    tokens = text.split()
    max_tokens = settings.max_chunk_tokens
    min_tokens = settings.min_chunk_tokens
    overlap = settings.overlap_tokens

    if len(tokens) <= max_tokens:
        yield Chunk(text=text, section=section_id, heading=heading, division=division, subdivision=subdivision, source="amlo_cap615")
        return

    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        slice_tokens = tokens[start:end]
        if len(slice_tokens) < min_tokens and start != 0:
            break
        chunk_text_val = " ".join(slice_tokens)
        yield Chunk(text=chunk_text_val, section=section_id, heading=heading, division=division, subdivision=subdivision, source="amlo_cap615")
        if end == len(tokens):
            break
        start = end - overlap


def ingest(source: Path, out_path: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Cannot find source file: {source}")

    suffix = source.suffix.lower()
    if suffix == ".pdf":
        raw_text = read_pdf(source)
    elif suffix == ".rtf":
        raw_text = read_rtf(source)
    else:
        raw_text = read_text(source)

    cleaned = clean(raw_text)
    sections = split_sections(cleaned)
    chunks: List[Chunk] = []
    for section_id, heading, body, division, subdivision in sections:
        for ch in chunk_text(section_id, heading, body, division, subdivision):
            chunks.append(ch)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            f.write(chunk.to_json(idx) + "\n")

    console.print(f"[bold green]Wrote {len(chunks)} chunks to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Ingest and chunk Cap. 615 (pdf/rtf/txt).")
    parser.add_argument("--source", required=True, type=Path, help="Path to source document")
    parser.add_argument("--out", default=paths.default_chunks, type=Path, help="Output JSONL path")
    parser.add_argument("--min_tokens", type=int, default=settings.min_chunk_tokens, help="Minimum tokens per chunk")
    parser.add_argument("--max_tokens", type=int, default=settings.max_chunk_tokens, help="Maximum tokens per chunk")
    parser.add_argument("--overlap", type=int, default=settings.overlap_tokens, help="Overlap tokens between chunks")
    args = parser.parse_args()

    settings.min_chunk_tokens = args.min_tokens
    settings.max_chunk_tokens = args.max_tokens
    settings.overlap_tokens = args.overlap
    ingest(args.source, args.out)


if __name__ == "__main__":
    main()
