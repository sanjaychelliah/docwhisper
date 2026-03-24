"""
eval.py — simple evaluation pipeline for docwhisper.

Runs a set of question/expected-answer pairs and checks:
  1. Citation presence rate
  2. Answer relevance (keyword overlap as a naive proxy)
  3. Retrieval recall (did the right source appear in top-k?)

This is meant to be run in CI to catch regressions.
Exit code 0 = pass, 1 = fail.

Usage:
    python -m docwhisper.eval --eval-file eval_questions.json
"""

import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model for an eval case
# ---------------------------------------------------------------------------


class EvalCase(NamedTuple):
    question: str
    expected_keywords: list[str]      # at least one should appear in the answer
    expected_source_hint: str | None  # filename substring that should appear in citations


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class EvalResult(NamedTuple):
    question: str
    citation_ok: bool
    relevance_ok: bool
    source_ok: bool
    answer_preview: str


def _relevance_check(answer_text: str, keywords: list[str]) -> bool:
    """Naive: at least one expected keyword present in the answer (case-insensitive)."""
    lowered = answer_text.lower()
    return any(kw.lower() in lowered for kw in keywords)


def _source_check(citations: list[dict], hint: str | None) -> bool:
    if hint is None:
        return True
    return any(hint.lower() in c["source"].lower() for c in citations)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_eval(eval_cases: list[EvalCase], docs_dir: Path | None = None) -> list[EvalResult]:
    from .pipeline import DocWhisper

    dw = DocWhisper(docs_dir=docs_dir)
    dw.load()

    results = []
    for case in eval_cases:
        logger.info("Evaluating: %s", case.question)
        answer = dw.ask(case.question)

        citation_ok = answer.has_citations
        relevance_ok = _relevance_check(answer.answer, case.expected_keywords)
        source_ok = _source_check(answer.citations, case.expected_source_hint)

        results.append(EvalResult(
            question=case.question,
            citation_ok=citation_ok,
            relevance_ok=relevance_ok,
            source_ok=source_ok,
            answer_preview=answer.answer[:80].replace("\n", " "),
        ))

    return results


def print_report(results: list[EvalResult]) -> bool:
    """Print a table and return True if all cases passed."""
    print("\n" + "═" * 70)
    print("  docwhisper eval report")
    print("═" * 70)

    passed = 0
    for r in results:
        status = "✓" if (r.citation_ok and r.relevance_ok and r.source_ok) else "✗"
        if status == "✓":
            passed += 1
        print(f"\n  {status}  Q: {r.question[:55]}")
        print(f"      citations : {'✓' if r.citation_ok else '✗'}")
        print(f"      relevance : {'✓' if r.relevance_ok else '✗'}")
        print(f"      source    : {'✓' if r.source_ok else '✗'}")
        print(f"      preview   : {r.answer_preview!r}")

    print("\n" + "─" * 70)
    print(f"  Result: {passed}/{len(results)} cases passed")
    print("═" * 70 + "\n")

    return passed == len(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_eval_file(path: Path) -> list[EvalCase]:
    """
    JSON format:
    [
      {
        "question": "What is the return policy?",
        "expected_keywords": ["30 days", "refund"],
        "expected_source_hint": "returns_policy"   // optional
      }
    ]
    """
    data = json.loads(path.read_text())
    return [
        EvalCase(
            question=d["question"],
            expected_keywords=d.get("expected_keywords", []),
            expected_source_hint=d.get("expected_source_hint"),
        )
        for d in data
    ]


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run docwhisper eval suite")
    parser.add_argument("--eval-file", type=Path, default=Path("eval_questions.json"))
    parser.add_argument("--docs-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.eval_file.exists():
        print(f"Eval file not found: {args.eval_file}")
        sys.exit(1)

    cases = _load_eval_file(args.eval_file)
    results = run_eval(cases, docs_dir=args.docs_dir)
    all_passed = print_report(results)

    sys.exit(0 if all_passed else 1)
