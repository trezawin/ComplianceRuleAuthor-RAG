from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .pipeline import Retriever, call_llm, render_context
from .config import settings
from .prompts import build_prompt, ROLE_PROMPT, TASK_INSTRUCTION

app = FastAPI(title="AMLO RAG Extractor")
retriever = Retriever()


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
    prompt = build_prompt(context=context, query=body.query)
    response = None if body.dry_run else call_llm(prompt)
    return ExtractResponse(
        query=body.query,
        prompt=prompt,
        response=response,
        retrieved=[RetrievedChunk(**r.chunk) for r in results],
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.embedding_model}
