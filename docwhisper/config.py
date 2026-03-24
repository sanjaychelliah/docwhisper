"""
config.py — all settings in one place. Override via env vars or a .env file.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # paths
    docs_dir: Path = field(default_factory=lambda: Path(os.getenv("DOCWHISPER_DOCS_DIR", "docs/")))
    index_dir: Path = field(default_factory=lambda: Path(os.getenv("DOCWHISPER_INDEX_DIR", ".docwhisper_index")))

    # embedding model (sentence-transformers compatible)
    embed_model: str = os.getenv("DOCWHISPER_EMBED_MODEL", "all-MiniLM-L6-v2")

    # reranker model (cross-encoder)
    rerank_model: str = os.getenv("DOCWHISPER_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # retrieval settings
    bm25_top_k: int = int(os.getenv("DOCWHISPER_BM25_TOP_K", "20"))
    vector_top_k: int = int(os.getenv("DOCWHISPER_VECTOR_TOP_K", "20"))
    rerank_top_k: int = int(os.getenv("DOCWHISPER_RERANK_TOP_K", "5"))

    # chunking
    chunk_size: int = int(os.getenv("DOCWHISPER_CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("DOCWHISPER_CHUNK_OVERLAP", "64"))

    # LLM — uses OpenAI-compatible API, swap any provider
    llm_api_base: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    llm_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("DOCWHISPER_LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = float(os.getenv("DOCWHISPER_LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("DOCWHISPER_LLM_MAX_TOKENS", "1024"))

    # answer format
    require_citations: bool = os.getenv("DOCWHISPER_REQUIRE_CITATIONS", "true").lower() == "true"
    min_citation_count: int = int(os.getenv("DOCWHISPER_MIN_CITATIONS", "1"))


# module-level default — import this everywhere
cfg = Config()
