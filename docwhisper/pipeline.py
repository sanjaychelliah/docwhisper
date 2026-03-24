"""
pipeline.py — the main entrypoint that glues everything together.

Typical usage:
    from docwhisper.pipeline import DocWhisper

    dw = DocWhisper()
    dw.ingest()
    answer = dw.ask("What is the return policy?")
    print(answer.format())
"""

import logging
from pathlib import Path

from .config import cfg
from .ingest import build_chunks, build_bm25_index, build_vector_index, load_documents, load_index, save_index
from .retrieve import retrieve
from .answer import Answer, generate_answer

logger = logging.getLogger(__name__)


class DocWhisper:
    """
    High-level interface.

    dw = DocWhisper()
    dw.ingest()                   # index your docs once
    answer = dw.ask("question")   # ask anything
    """

    def __init__(self, docs_dir: Path | None = None, index_dir: Path | None = None):
        self.docs_dir = docs_dir or cfg.docs_dir
        self.index_dir = index_dir or cfg.index_dir
        self._chunks = None
        self._bm25 = None
        self._embeddings = None

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self) -> None:
        """Load, chunk, and index all documents. Call once, then ask many."""
        docs = load_documents(self.docs_dir)
        if not docs:
            raise ValueError(
                f"No documents found in {self.docs_dir}.\n"
                "Add .txt, .md, .pdf, or .html files and run ingest() again."
            )
        chunks = build_chunks(docs)
        bm25 = build_bm25_index(chunks)
        embeddings = build_vector_index(chunks)
        save_index(chunks, bm25, embeddings, self.index_dir)

        self._chunks = chunks
        self._bm25 = bm25
        self._embeddings = embeddings
        logger.info("Ingestion complete — %d chunks indexed.", len(chunks))

    # ------------------------------------------------------------------
    # Load existing index
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load an already-built index from disk. Faster than re-ingesting."""
        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"No index found at {self.index_dir}. Run .ingest() first."
            )
        self._chunks, self._bm25, self._embeddings = load_index(self.index_dir)
        logger.info("Loaded %d chunks from index.", len(self._chunks))

    def _ensure_loaded(self):
        if self._chunks is None:
            # try to auto-load from disk if index exists
            if self.index_dir.exists():
                self.load()
            else:
                raise RuntimeError(
                    "Index not loaded. Call .ingest() to build the index, "
                    "or .load() if the index already exists."
                )

    # ------------------------------------------------------------------
    # Ask
    # ------------------------------------------------------------------

    def ask(self, question: str) -> Answer:
        """Ask a question. Returns an Answer object with cited text."""
        self._ensure_loaded()

        retrieved = retrieve(
            query=question,
            chunks=self._chunks,
            bm25=self._bm25,
            embeddings=self._embeddings,
        )

        return generate_answer(question, retrieved)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self):
        n = len(self._chunks) if self._chunks else 0
        return f"DocWhisper(docs_dir={self.docs_dir}, chunks={n})"
