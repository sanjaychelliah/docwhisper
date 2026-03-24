"""
ingest.py — load documents, chunk them, build BM25 + vector indexes.

Supports: .txt, .md, .pdf, .html
Run: python -m docwhisper.ingest --docs-dir ./docs
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Chunk:
    """One piece of a document, ready to be retrieved."""

    def __init__(self, text: str, source: str, chunk_id: int, metadata: dict | None = None):
        self.text = text
        self.source = source          # filename or URL
        self.chunk_id = chunk_id      # global index across all chunks
        self.metadata = metadata or {}

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(id={self.chunk_id}, source={self.source!r}, text={preview!r}...)"


# ---------------------------------------------------------------------------
# Document loaders
# ---------------------------------------------------------------------------


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        logger.warning("pypdf not installed — skipping %s. Run: pip install pypdf", path)
        return ""


def _load_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        return soup.get_text(separator="\n")
    except ImportError:
        logger.warning("beautifulsoup4 not installed — skipping %s. Run: pip install beautifulsoup4", path)
        return ""


LOADERS = {".txt": _load_txt, ".md": _load_md, ".pdf": _load_pdf, ".html": _load_html}


def load_documents(docs_dir: Path) -> list[tuple[str, Path]]:
    """Return [(raw_text, path), ...] for every supported file under docs_dir."""
    docs = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() in LOADERS and path.is_file():
            logger.info("Loading %s", path)
            text = LOADERS[path.suffix.lower()](path)
            if text.strip():
                docs.append((text, path))
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Simple sliding-window word-level chunker.
    Not rocket science — good enough for most use cases.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_chunks(docs: list[tuple[str, Path]]) -> list[Chunk]:
    """Turn raw documents into Chunk objects."""
    all_chunks = []
    chunk_id = 0
    for text, path in docs:
        pieces = _split_text(text, cfg.chunk_size, cfg.chunk_overlap)
        for piece in pieces:
            all_chunks.append(Chunk(text=piece, source=str(path), chunk_id=chunk_id))
            chunk_id += 1
    logger.info("Built %d chunks from %d documents", len(all_chunks), len(docs))
    return all_chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def build_bm25_index(chunks: list[Chunk]):
    """Build a BM25 index from chunks. Returns a rank_bm25.BM25Okapi object."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("Run: pip install rank-bm25")

    tokenized = [chunk.text.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)


def build_vector_index(chunks: list[Chunk]) -> np.ndarray:
    """
    Encode all chunks with sentence-transformers.
    Returns a float32 numpy array of shape (N, D).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")

    logger.info("Encoding %d chunks with model '%s' ...", len(chunks), cfg.embed_model)
    model = SentenceTransformer(cfg.embed_model)
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def save_index(chunks: list[Chunk], bm25, embeddings: np.ndarray, index_dir: Path):
    """Persist everything to disk."""
    index_dir.mkdir(parents=True, exist_ok=True)

    # chunks as JSON (human-readable, easy to inspect)
    chunks_data = [
        {"text": c.text, "source": c.source, "chunk_id": c.chunk_id, "metadata": c.metadata}
        for c in chunks
    ]
    (index_dir / "chunks.json").write_text(json.dumps(chunks_data, indent=2))

    # BM25 index as pickle (rank_bm25 objects aren't JSON-serializable)
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # embeddings as numpy binary
    np.save(index_dir / "embeddings.npy", embeddings)

    logger.info("Index saved to %s", index_dir)


def load_index(index_dir: Path):
    """Load persisted index. Returns (chunks, bm25, embeddings)."""
    chunks_data = json.loads((index_dir / "chunks.json").read_text())
    chunks = [Chunk(**d) for d in chunks_data]

    with open(index_dir / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    embeddings = np.load(index_dir / "embeddings.npy")
    return chunks, bm25, embeddings


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_ingest(docs_dir: Path | None = None, index_dir: Path | None = None):
    docs_dir = docs_dir or cfg.docs_dir
    index_dir = index_dir or cfg.index_dir

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("=== docwhisper ingest ===")
    logger.info("docs_dir : %s", docs_dir)
    logger.info("index_dir: %s", index_dir)

    docs = load_documents(docs_dir)
    if not docs:
        logger.error("No documents found in %s", docs_dir)
        return

    chunks = build_chunks(docs)
    bm25 = build_bm25_index(chunks)
    embeddings = build_vector_index(chunks)
    save_index(chunks, bm25, embeddings, index_dir)
    logger.info("Done! Index ready at %s", index_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index documents for docwhisper")
    parser.add_argument("--docs-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    args = parser.parse_args()

    run_ingest(docs_dir=args.docs_dir, index_dir=args.index_dir)
