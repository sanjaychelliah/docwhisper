"""
examples/quickstart.py — get docwhisper running in 5 minutes.

Prerequisites:
    pip install -e ".[all]"
    export OPENAI_API_KEY=sk-...

Then run:
    python examples/quickstart.py
"""

import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

DOCS_DIR = Path(__file__).parent / "sample_docs"

if not os.getenv("OPENAI_API_KEY"):
    print("\n⚠  Set OPENAI_API_KEY first:\n   export OPENAI_API_KEY=sk-...\n")
    raise SystemExit(1)

from docwhisper.pipeline import DocWhisper

dw = DocWhisper(docs_dir=DOCS_DIR)

print("\n── Step 1: Ingesting docs ─────────────────────────────────────")
dw.ingest()

questions = [
    "What is the return policy and how long do I have?",
    "How do I reset my password?",
    "How long does a refund take to process?",
    "What API rate limits apply to a Pro tier account?",
]

print("\n── Step 2: Asking questions ───────────────────────────────────")
for q in questions:
    answer = dw.ask(q)
    print(answer.format())
