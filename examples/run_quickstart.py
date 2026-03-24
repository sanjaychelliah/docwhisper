"""
examples/quickstart.py — minimal working example.

Run this after setting your OPENAI_API_KEY:

    export OPENAI_API_KEY=sk-...
    python examples/quickstart.py
"""

import logging
import os
import sys
from pathlib import Path

# make sure we can import docwhisper from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from docwhisper.pipeline import DocWhisper

DOCS_DIR = Path(__file__).parent / "sample_docs"
INDEX_DIR = Path(__file__).parent / ".index"

if not os.getenv("OPENAI_API_KEY"):
    print("⚠  OPENAI_API_KEY not set.")
    print("   Set it and re-run:  export OPENAI_API_KEY=sk-...")
    sys.exit(1)

# ── 1. Build the index (only need to do this once) ────────────────────────────
print("\n── Step 1: Ingest documents ────────────────────────────────────────────")
dw = DocWhisper(docs_dir=DOCS_DIR, index_dir=INDEX_DIR)
dw.ingest()

# ── 2. Ask questions ──────────────────────────────────────────────────────────
print("\n── Step 2: Ask questions ───────────────────────────────────────────────")

questions = [
    "What is the return policy and how many days do I have?",
    "How do I reset my password?",
    "What payment methods are accepted?",
    "Is my data encrypted and where is it stored?",
]

for question in questions:
    answer = dw.ask(question)
    print(answer.format())
