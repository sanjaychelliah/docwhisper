"""
server.py — optional FastAPI server for docwhisper.

Start: uvicorn docwhisper.server:app --reload

Endpoints:
  POST /ask          { "question": "..." }  → answer + citations
  POST /ingest       rebuild the index
  GET  /health       liveness check
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .pipeline import DocWhisper
from .telemetry import tracker

app = FastAPI(
    title="docwhisper",
    description="Ask your documents — get cited answers.",
    version="0.1.0",
)

# global pipeline instance — loaded once at startup
_dw: DocWhisper | None = None


@app.on_event("startup")
async def startup():
    global _dw
    _dw = DocWhisper()
    try:
        _dw.load()
    except FileNotFoundError:
        # index doesn't exist yet — that's OK, user should POST /ingest first
        pass


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    ref: str
    source: str
    excerpt: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]
    has_citations: bool
    model: str


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/metrics")
def metrics():
    return tracker.stats()


@app.get("/health")
def health():
    loaded = _dw is not None and _dw._chunks is not None
    return {"status": "ok", "index_loaded": loaded}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if _dw is None or _dw._chunks is None:
        raise HTTPException(status_code=503, detail="Index not loaded. POST /ingest first.")

    answer = _dw.ask(req.question)
    return AskResponse(
        question=answer.question,
        answer=answer.answer,
        citations=[CitationOut(**c) for c in answer.citations],
        has_citations=answer.has_citations,
        model=answer.model,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    global _dw
    if _dw is None:
        _dw = DocWhisper()
    _dw.ingest()
    return IngestResponse(
        message="Ingestion complete",
        chunks_indexed=len(_dw._chunks),
    )
