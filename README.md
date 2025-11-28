# AMLO RAG Extractor

RAG pipeline to extract enforceable obligations from the Hong Kong AML/CFT Ordinance (Cap. 615) and map them to ERC‑3643 concepts. Includes an optional prompt rewriter (heuristic or OpenAI) and a FastAPI service.

## Project contents
- `src/ingest.py`: PDF/text ingestion and section-aware chunking.
- `src/index.py`: Embedding + BM25 indexing.
- `src/pipeline.py`: Retrieval orchestration, prompt assembly, optional prompt rewriter.
- `src/prompt_rewriter.py`: Heuristic and OpenAI-based prompt rewriting.
- `src/prompts.py`: System/task prompts and JSON schema.
- `src/server.py`: FastAPI app for `/extract`.
- `src/config.py`: Paths and tunables (device, weights, model, API key override).
- `data/raw`: Place Cap. 615 text here (`cap615.pdf|rtf|txt`).
- `data/processed`: Generated chunks and indexes.

## Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Mac M1/M2: set `device="cpu"` in `src/config.py` if you hit GPU crashes.

Provide text: copy the Cap.615 file into `data/raw/` (e.g., `cap615.rtf`).

API key (if using OpenAI): export `OPENAI_API_KEY=...` or create a `.env` in repo root. You can also set `openai_api_key` in `src/config.py` (avoid hardcoding).

## Core commands
Ingest (chunk):
```bash
python -m src.ingest --source data/raw/cap615.rtf --out data/processed/cap615.jsonl
```
Index (embeddings + BM25):
```bash
python -m src.index --chunks data/processed/cap615.jsonl --index_out data/processed/embeddings.npy --bm25_out data/processed/bm25.pkl
```
Single extraction:
```bash
# baseline
python -m src.pipeline --query "customer due diligence obligations" --top_k 8 --dry_run
# with prompt rewriter (heuristic)
python -m src.pipeline --query "customer due diligence obligations" --top_k 8 --use_rewriter --rewriter_dry_run --dry_run
# with OpenAI rewriter (requires key; omit --dry_run to call the LLM)
python -m src.pipeline --query "customer due diligence obligations" --top_k 8 --use_rewriter
```
Save output: add `--save data/processed/out.json`.

Batch extraction (dry run first):
```bash
python -m src.batch_extract --chunks data/processed/cap615.jsonl --out data/processed/all_rules.json --dry_run
# real calls: drop --dry_run and ensure OPENAI_API_KEY is set
```

## API server
```bash
uvicorn src.server:app --reload --port 8000
```
Example:
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"query": "customer due diligence obligations", "top_k": 8}'
```

## Troubleshooting
- `OPENAI_API_KEY` missing: export it or add to `.env`.
- `Client.__init__() got an unexpected keyword argument 'proxies'`: upgrade deps  
  `pip install --upgrade "openai>=1.44.0" "httpx>=0.25,<0.28" "httpcore>=1.0.0"`
- M1/M2 crashes: set `device="cpu"` in `src/config.py`.

## Notes
- Chunker keeps section headings/citations; tune `--min_tokens/--max_tokens` in `ingest`.
- Hybrid retrieval mixes embeddings + BM25; adjust weights in `src/config.py` or `src/pipeline.py`.
- Outputs follow the JSON structure defined in prompts; add validation in `pipeline.run_extraction` if needed.
