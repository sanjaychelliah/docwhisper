"""
tests/test_ingest.py — unit tests for chunking and loading.
No network calls, no LLM needed.
"""

import json
import tempfile
from pathlib import Path

import pytest

from docwhisper.ingest import Chunk, _split_text, build_chunks


# ---------------------------------------------------------------------------
# _split_text
# ---------------------------------------------------------------------------


def test_split_text_basic():
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = _split_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    # every chunk should have content
    for c in chunks:
        assert c.strip()


def test_split_text_short_doc():
    text = "hello world"
    chunks = _split_text(text, chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_split_text_overlap():
    words = [f"w{i}" for i in range(30)]
    text = " ".join(words)
    chunks = _split_text(text, chunk_size=10, overlap=5)
    # with overlap the second chunk should share words with the first
    first_words = set(chunks[0].split())
    second_words = set(chunks[1].split())
    assert first_words & second_words, "Expected overlapping words between consecutive chunks"


def test_split_text_empty():
    chunks = _split_text("", chunk_size=512, overlap=64)
    assert chunks == []


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------


def test_build_chunks_assigns_ids():
    docs = [
        ("some text about cats " * 20, Path("a.txt")),
        ("some text about dogs " * 20, Path("b.txt")),
    ]
    chunks = build_chunks(docs)
    ids = [c.chunk_id for c in chunks]
    assert ids == list(range(len(chunks))), "chunk_ids should be sequential"


def test_build_chunks_preserves_source():
    docs = [("hello world " * 10, Path("my_doc.txt"))]
    chunks = build_chunks(docs)
    for c in chunks:
        assert c.source == "my_doc.txt"


# ---------------------------------------------------------------------------
# Chunk serialization round-trip
# ---------------------------------------------------------------------------


def test_chunk_json_roundtrip():
    c = Chunk(text="hello world", source="test.md", chunk_id=0, metadata={"page": 1})
    data = {"text": c.text, "source": c.source, "chunk_id": c.chunk_id, "metadata": c.metadata}
    c2 = Chunk(**data)
    assert c.text == c2.text
    assert c.source == c2.source
    assert c.chunk_id == c2.chunk_id
    assert c.metadata == c2.metadata
