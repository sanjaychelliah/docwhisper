"""
examples/use_ollama.py — run docwhisper fully locally with Ollama.

No API key needed. Just:
    1. Install Ollama: https://ollama.com
    2. Pull a model: ollama pull llama3.2
    3. Run: python examples/use_ollama.py

Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1
so we just point docwhisper at it.
"""

import os
from pathlib import Path

# Tell docwhisper to use Ollama instead of OpenAI
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"   # Ollama doesn't need a real key
os.environ["DOCWHISPER_LLM_MODEL"] = "llama3.2"

from docwhisper.pipeline import DocWhisper

DOCS_DIR = Path(__file__).parent / "sample_docs"

dw = DocWhisper(docs_dir=DOCS_DIR)
dw.ingest()

answer = dw.ask("What is the refund processing time?")
print(answer.format())
