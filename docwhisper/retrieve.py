"""
retrieve.py — hybrid retrieval (BM25 + vector search) + cross-encoder reranking.

The pipeline:
  1. BM25 recall   — keyword-level, fast, good for exact matches
  2. Vector recall — semantic, catches paraphrase / synonyms
  3. Merge + dedupe
  4. Cross-encoder reranking — slow but precise, runs over top candidates only
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from .telemetry import RequestTrace

import numpy as np

from .config import cfg
from .ingest import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class RetrievedChunk(NamedTuple):
    chunk: Chunk
    score: float           # reranker score (higher = more relevant)
    retrieval_source: str  # "bm25", "vector", or "both"


# ---------------------------------------------------------------------------
# Individual retrievers
# ---------------------------------------------------------------------------


def bm25_search(query: str, bm25, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
    """Return (chunk, score) pairs ranked by BM25."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]


def vector_search(
    query: str,
    embeddings: np.ndarray,
    chunks: list[Chunk],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    """Return (chunk, cosine_similarity) pairs ranked by vector similarity."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")

    model = SentenceTransformer(cfg.embed_model)
    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)[0]

    # cosine similarity — embeddings are already L2-normalised by sentence-transformers
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-9)
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    sims = normed @ q_norm

    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(chunks[i], float(sims[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Hybrid merge
# ---------------------------------------------------------------------------


def hybrid_merge(
    bm25_results: list[tuple[Chunk, float]],
    vector_results: list[tuple[Chunk, float]],
) -> list[tuple[Chunk, str]]:
    """
    Union of both result sets, deduplicated by chunk_id.
    Records retrieval_source for transparency.
    """
    seen: dict[int, tuple[Chunk, str]] = {}

    for chunk, _ in bm25_results:
        seen[chunk.chunk_id] = (chunk, "bm25")

    for chunk, _ in vector_results:
        if chunk.chunk_id in seen:
            seen[chunk.chunk_id] = (chunk, "both")
        else:
            seen[chunk.chunk_id] = (chunk, "vector")

    return list(seen.values())


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------


def rerank(
    query: str,
    candidates: list[tuple[Chunk, str]],
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Score (query, chunk) pairs with a cross-encoder.
    Much slower than bi-encoder but way more accurate.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")

    if not candidates:
        return []

    logger.info("Reranking %d candidates with cross-encoder...", len(candidates))
    model = CrossEncoder(cfg.rerank_model)

    pairs = [(query, chunk.text) for chunk, _ in candidates]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        RetrievedChunk(chunk=chunk, score=float(score), retrieval_source=source)
        for (chunk, source), score in ranked[:top_k]
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    chunks: list[Chunk],
    bm25,
    embeddings: np.ndarray,
    *,
    trace: RequestTrace | None = None,
) -> list[RetrievedChunk]:
    """
    Full hybrid retrieval pipeline.
    Returns top-k reranked chunks.
    """
    from .telemetry import tracker, span_or_null

    logger.info("BM25 search (top %d)...", cfg.bm25_top_k)
    with span_or_null("bm25_search", trace, tracker):
        bm25_results = bm25_search(query, bm25, chunks, top_k=cfg.bm25_top_k)

    logger.info("Vector search (top %d)...", cfg.vector_top_k)
    with span_or_null("vector_search", trace, tracker):
        vector_results = vector_search(query, embeddings, chunks, top_k=cfg.vector_top_k)

    merged = hybrid_merge(bm25_results, vector_results)
    logger.info("Merged %d unique candidates", len(merged))

    with span_or_null("rerank", trace, tracker):
        results = rerank(query, merged, top_k=cfg.rerank_top_k)
    logger.info("Returning %d reranked chunks", len(results))

    if trace is not None and results:
        trace.top_rerank_score = results[0].score
        trace.retrieval_latency_ms = sum(
            s.latency_ms for s in trace.spans
            if s.name in ("bm25_search", "vector_search", "rerank")
            and s.end_time is not None
        )

    return results
