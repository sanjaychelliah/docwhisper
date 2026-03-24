"""
tests/test_answer.py — unit tests for citation parsing.
No LLM calls needed.
"""

from docwhisper.answer import _parse_citations
from docwhisper.ingest import Chunk
from docwhisper.retrieve import RetrievedChunk


def _make_retrieved(chunk_id: int, source: str = "doc.txt") -> RetrievedChunk:
    chunk = Chunk(text=f"This is chunk {chunk_id} content. " * 5, source=source, chunk_id=chunk_id)
    return RetrievedChunk(chunk=chunk, score=0.9, retrieval_source="both")


def test_parse_citations_basic():
    answer = "The sky is blue [1]. Water is wet [2]."
    retrieved = [_make_retrieved(0), _make_retrieved(1)]
    citations = _parse_citations(answer, retrieved)
    assert len(citations) == 2
    refs = [c["ref"] for c in citations]
    assert "[1]" in refs
    assert "[2]" in refs


def test_parse_citations_duplicate_refs():
    # [1] mentioned twice — should deduplicate
    answer = "According to [1], this is true. And again [1] confirms it."
    retrieved = [_make_retrieved(0)]
    citations = _parse_citations(answer, retrieved)
    assert len(citations) == 1


def test_parse_citations_out_of_range():
    # [5] is referenced but only 2 chunks retrieved — should not crash
    answer = "This is true [5]."
    retrieved = [_make_retrieved(0), _make_retrieved(1)]
    citations = _parse_citations(answer, retrieved)
    assert len(citations) == 0  # [5] is out of range


def test_parse_citations_no_refs():
    answer = "I don't know the answer."
    retrieved = [_make_retrieved(0)]
    citations = _parse_citations(answer, retrieved)
    assert citations == []


def test_citation_source_preserved():
    answer = "The policy says [1]."
    retrieved = [_make_retrieved(0, source="policies/refunds.md")]
    citations = _parse_citations(answer, retrieved)
    assert citations[0]["source"] == "policies/refunds.md"
