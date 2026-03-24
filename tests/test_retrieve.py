"""
tests/test_retrieve.py — unit tests for hybrid merge logic.
No models loaded, no embeddings needed.
"""

from docwhisper.ingest import Chunk
from docwhisper.retrieve import hybrid_merge


def _make_chunk(chunk_id: int, text: str = "dummy") -> Chunk:
    return Chunk(text=text, source=f"doc_{chunk_id}.txt", chunk_id=chunk_id)


def test_hybrid_merge_deduplication():
    a = _make_chunk(0)
    b = _make_chunk(1)
    c = _make_chunk(2)

    bm25_results = [(a, 0.9), (b, 0.5)]
    vector_results = [(b, 0.8), (c, 0.7)]

    merged = hybrid_merge(bm25_results, vector_results)
    ids = [chunk.chunk_id for chunk, _ in merged]

    assert len(ids) == 3, "Should have 3 unique chunks"
    assert len(set(ids)) == len(ids), "No duplicates"


def test_hybrid_merge_source_tracking():
    a = _make_chunk(0)
    b = _make_chunk(1)

    bm25_results = [(a, 1.0)]
    vector_results = [(a, 0.9), (b, 0.6)]

    merged = hybrid_merge(bm25_results, vector_results)
    source_map = {chunk.chunk_id: source for chunk, source in merged}

    assert source_map[0] == "both", "chunk 0 appeared in both retrievers"
    assert source_map[1] == "vector", "chunk 1 only in vector results"


def test_hybrid_merge_empty_inputs():
    merged = hybrid_merge([], [])
    assert merged == []


def test_hybrid_merge_only_bm25():
    a = _make_chunk(5)
    merged = hybrid_merge([(a, 1.0)], [])
    assert len(merged) == 1
    assert merged[0][1] == "bm25"
