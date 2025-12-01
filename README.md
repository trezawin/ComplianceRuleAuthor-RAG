# AMLO RAG Extractor

RAG pipeline scaffold to extract enforceable obligations from the HKMA Anti‑Money Laundering and Counter‑Financing of Terrorism Ordinance (Cap. 615) and map them to ERC‑3643 concepts.

## What’s here
- `src/ingest.py`: PDF/text ingestion and section‑aware chunking.
- `src/index.py`: Embedding + BM25 indexing (numpy embeddings + metadata).
- `src/pipeline.py`: Retrieval orchestration and prompt assembly.
- `src/prompts.py`: System/task prompts for the auditor role and JSON schema.
- `src/server.py`: FastAPI service for `/extract`.
- `src/config.py`: Paths and model configurables.
- `requirements.txt`: Python deps (install in a virtualenv).
- `data/raw`: Put the AMLO (Cap. 615) PDF or RTF cleaned text here.
- `data/processed`: Generated structured text, chunks, and vector index cache.

## Quick start
1) Create a virtualenv and install deps:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2) Add the AMLO source into `data/raw/`, e.g., `cap615.pdf`, `cap615.rtf`, or `cap615.txt`. Add the ERC-3643 spec/API text into `data/raw/erc-3643.txt`.

3) Ingest + index:
```bash
python -m src.ingest --source data/raw/cap615.pdf --out data/processed/cap615.jsonl
python scripts/ingest_erc3643.py --source data/raw/erc-3643.txt --out data/processed/erc3643.jsonl
python scripts/build_erc_whitelist.py --source data/raw/erc-3643.txt --out data/processed/erc3643_whitelist.json
python -m src.index --chunks data/processed/cap615.jsonl --index_out data/processed/embeddings.npy --bm25_out data/processed/bm25.pkl --extra data/processed/erc3643.jsonl
```
If you’re on macOS with M1/M2 and hit GPU-related crashes, set CPU mode in `src/config.py` (`device="cpu"`).

4) Run an extraction (prints the prompt & optional LLM call):
```bash
python -m src.pipeline --query "customer due diligence obligations" --top_k 8
```
Provide `OPENAI_API_KEY` (or update `LLM_PROVIDER` in `src/pipeline.py`) to actually call a model. Without a key, the script will emit the assembled prompt for manual/other use.

5) Extract all sections to one JSON file:
```bash
python -m src.batch_extract --chunks data/processed/cap615.jsonl --out data/processed/all_rules.json
```
Use `--limit 5 --dry_run` to test quickly; set `OPENAI_API_KEY` before running to perform real calls.

ERC-3643 grounding: prompts include an ERC-3643 whitelist and post-validation forces `erc_3643` to either a whitelisted function/module or `N/A (off-chain)`. Keep `erc3643_whitelist.json` current and include `erc3643.jsonl` in indexing (`--extra`) to keep mappings accurate.

6) Summarize chunks (for inspection):
```bash
python scripts/summarize_chunks.py --chunks data/processed/cap615.jsonl --show 3
```

7) Coverage sanity check (does chunking match the PDF):
```bash
python scripts/coverage_check.py --pdf data/raw/cap-615.pdf --chunks data/processed/cap615.jsonl --show_missing 10
```

5) Serve an API:
```bash
uvicorn src.server:app --reload --port 8000
```
Then POST to `/extract` with `{"query": "...", "top_k": 8}`.

## Common pitfalls
- If you hit `Client.__init__() got an unexpected keyword argument 'proxies'` from `openai`, upgrade the HTTP stack in your venv:  
  `pip install --upgrade "openai>=1.44.0" "httpx>=0.25,<0.28" "httpcore>=1.0.0"`
- Ensure `OPENAI_API_KEY` is set (shell export or `.env`) before running commands without `--dry_run`.
## Notes
- Chunker preserves section headings and citations; tune `--min_tokens/--max_tokens` as needed.
- Hybrid retrieval combines semantic + BM25; adjust weights in `src/pipeline.py`.
- Outputs enforce the JSON shape requested in the role instruction. Add schema validation in `pipeline.run_extraction` if desired.***
