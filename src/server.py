from typing import List, Optional
import json

from fastapi import FastAPI
from pydantic import BaseModel

from .pipeline import Retriever, call_llm, render_context, load_erc_whitelist, enforce_erc_whitelist
from .config import settings, paths
from .prompts import build_prompt, ROLE_PROMPT, TASK_INSTRUCTION

app = FastAPI(title="AMLO RAG Extractor")
retriever = Retriever()
erc_allowed = load_erc_whitelist(paths.erc_whitelist)
erc_whitelist_text = "\n".join(sorted(erc_allowed))


class ExtractRequest(BaseModel):
    query: str
    top_k: int = settings.top_k
    dry_run: bool = False


class RetrievedChunk(BaseModel):
    id: int
    text: str
    section: Optional[str] = None
    heading: Optional[str] = None
    source: Optional[str] = None
    tokens: Optional[int] = None


class ExtractResponse(BaseModel):
    query: str
    prompt: str
    response: Optional[str]
    retrieved: List[RetrievedChunk]


@app.post("/extract", response_model=ExtractResponse)
def extract(body: ExtractRequest) -> ExtractResponse:
    results = retriever.search(body.query, k=body.top_k)
    context = render_context(results)
    prompt = build_prompt(context=context, query=body.query, erc_whitelist=erc_whitelist_text)
    response = None if body.dry_run else call_llm(prompt)
    if response:
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                parsed = [parsed]
            cleaned = [enforce_erc_whitelist(r, erc_allowed) for r in parsed if isinstance(r, dict)]
            response = json.dumps(cleaned, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return ExtractResponse(
        query=body.query,
        prompt=prompt,
        response=response,
        retrieved=[RetrievedChunk(**r.chunk) for r in results],
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.embedding_model}
