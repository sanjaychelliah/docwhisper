"""
eval.py — simple evaluation pipeline for docwhisper.

Runs a set of question/expected-answer pairs and checks:
  1. Citation presence rate
  2. Answer relevance (keyword overlap as a naive proxy)
  3. Retrieval recall (did the right source appear in top-k?)

Optionally logs metrics to MLflow and W&B, and supports regression gating
against a saved baseline.

Usage:
    python -m docwhisper.eval --eval-file eval_questions.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

from .config import cfg

logger = logging.getLogger(__name__)

BASELINE_FILE = Path(".docwhisper_baselines.json")

# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────


class EvalCase(NamedTuple):
    question: str
    expected_keywords: list[str]      # at least one should appear in the answer
    expected_source_hint: str | None  # filename substring that should appear in citations


class EvalResult(NamedTuple):
    question: str
    citation_ok: bool
    relevance_ok: bool
    source_ok: bool
    answer_preview: str
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    top_rerank_score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────


def _relevance_check(answer_text: str, keywords: list[str]) -> bool:
    """Naive: at least one expected keyword present in the answer (case-insensitive)."""
    lowered = answer_text.lower()
    return any(kw.lower() in lowered for kw in keywords)


def _source_check(citations: list[dict], hint: str | None) -> bool:
    if hint is None:
        return True
    return any(hint.lower() in c["source"].lower() for c in citations)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline / regression gating
# ─────────────────────────────────────────────────────────────────────────────


def save_baseline(metrics: dict) -> None:
    """Write current eval metrics as the new baseline."""
    BASELINE_FILE.write_text(json.dumps(metrics, indent=2))
    logger.info("Baseline saved to %s", BASELINE_FILE)


def load_baseline() -> dict | None:
    """Load baseline metrics. Returns None if no baseline exists yet."""
    if not BASELINE_FILE.exists():
        return None
    try:
        return json.loads(BASELINE_FILE.read_text())
    except Exception as exc:
        logger.warning("Could not load baseline file: %s", exc)
        return None


def check_regression(
    current: dict,
    baseline: dict,
    thresholds: dict | None = None,
) -> list[str]:
    """
    Compare current metrics against baseline + allowed degradation thresholds.
    Returns a list of regression messages (empty = no regressions).
    """
    if thresholds is None:
        thresholds = {
            "pass_rate": 0.10,
            "citation_rate": 0.10,
            "mean_latency_ms": 500.0,
            "total_cost_usd": 0.05,
        }

    messages: list[str] = []

    for key, max_drop in [("pass_rate", thresholds["pass_rate"]),
                           ("citation_rate", thresholds["citation_rate"])]:
        base_val = baseline.get(key)
        curr_val = current.get(key)
        if base_val is not None and curr_val is not None:
            drop = base_val - curr_val
            if drop > max_drop:
                messages.append(
                    f"REGRESSION: {key} dropped {drop:.3f} (baseline={base_val:.3f}, "
                    f"current={curr_val:.3f}, max_allowed_drop={max_drop})"
                )

    for key, max_increase in [("mean_latency_ms", thresholds["mean_latency_ms"]),
                               ("total_cost_usd", thresholds["total_cost_usd"])]:
        base_val = baseline.get(key)
        curr_val = current.get(key)
        if base_val is not None and curr_val is not None:
            increase = curr_val - base_val
            if increase > max_increase:
                messages.append(
                    f"REGRESSION: {key} increased {increase:.3f} (baseline={base_val:.3f}, "
                    f"current={curr_val:.3f}, max_allowed_increase={max_increase})"
                )

    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


def run_eval(
    eval_cases: list[EvalCase],
    docs_dir: Path | None = None,
    run_name: str = "eval",
    fail_on_regression: bool = False,
    save_baseline_flag: bool = False,
) -> list[EvalResult]:
    from .pipeline import DocWhisper

    dw = DocWhisper(docs_dir=docs_dir)
    dw.load()

    mlflow_run = None
    if cfg.mlflow_tracking_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
            mlflow.set_experiment(cfg.mlflow_experiment)
            mlflow_run = mlflow.start_run(run_name=run_name)
        except Exception as exc:
            logger.warning("MLflow run start failed: %s", exc)
            mlflow_run = None

    wandb_run = None
    import os
    if os.getenv("WANDB_API_KEY"):
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity or None,
                name=run_name,
                job_type="eval",
            )
        except Exception as exc:
            logger.warning("W&B init failed: %s", exc)

    results: list[EvalResult] = []
    for case in eval_cases:
        logger.info("Evaluating: %s", case.question)
        answer = dw.ask(case.question)

        citation_ok = answer.has_citations
        relevance_ok = _relevance_check(answer.answer, case.expected_keywords)
        source_ok = _source_check(answer.citations, case.expected_source_hint)

        latency_ms = 0.0
        cost_usd = 0.0
        top_rerank_score = 0.0
        if answer.trace is not None:
            latency_ms = answer.trace.total_latency_ms
            cost_usd = answer.trace.estimated_cost_usd
            top_rerank_score = answer.trace.top_rerank_score

        results.append(EvalResult(
            question=case.question,
            citation_ok=citation_ok,
            relevance_ok=relevance_ok,
            source_ok=source_ok,
            answer_preview=answer.answer[:80].replace("\n", " "),
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            top_rerank_score=top_rerank_score,
        ))

    # ── aggregate metrics ──────────────────────────────────────────────────
    n = len(results)
    agg_metrics = {
        "eval/citation_rate": sum(r.citation_ok for r in results) / max(n, 1),
        "eval/relevance_rate": sum(r.relevance_ok for r in results) / max(n, 1),
        "eval/source_recall": sum(r.source_ok for r in results) / max(n, 1),
        "eval/pass_rate": sum(r.citation_ok and r.relevance_ok and r.source_ok for r in results) / max(n, 1),
        "eval/mean_latency_ms": sum(r.latency_ms for r in results) / max(n, 1),
        "eval/p95_latency_ms": _p95([r.latency_ms for r in results]),
        "eval/total_cost_usd": sum(r.cost_usd for r in results),
        "eval/mean_rerank_score": sum(r.top_rerank_score for r in results) / max(n, 1),
    }

    # flat keys for baseline comparison
    flat_metrics = {
        "pass_rate": agg_metrics["eval/pass_rate"],
        "citation_rate": agg_metrics["eval/citation_rate"],
        "mean_latency_ms": agg_metrics["eval/mean_latency_ms"],
        "total_cost_usd": agg_metrics["eval/total_cost_usd"],
    }

    # ── MLflow logging ─────────────────────────────────────────────────────
    if mlflow_run is not None:
        try:
            import mlflow, tempfile

            mlflow.log_metrics(agg_metrics)

            # artifact: full eval results JSON
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="eval_results_"
            ) as tmp:
                json.dump(
                    [r._asdict() for r in results],
                    tmp,
                    indent=2,
                )
                tmp_path = tmp.name
            mlflow.log_artifact(tmp_path, artifact_path="eval")
            mlflow.end_run()
        except Exception as exc:
            logger.warning("MLflow metric logging failed: %s", exc)

    # ── W&B logging ────────────────────────────────────────────────────────
    if wandb_run is not None:
        try:
            import wandb
            wandb.summary.update(agg_metrics)
            wandb.finish()
        except Exception as exc:
            logger.warning("W&B summary logging failed: %s", exc)

    # ── regression gate ────────────────────────────────────────────────────
    baseline = load_baseline()
    if baseline is not None:
        regressions = check_regression(flat_metrics, baseline)
        if regressions:
            print("\n⚠  Regressions detected:")
            for msg in regressions:
                print(f"   {msg}")
            if fail_on_regression:
                print("\nExiting with code 1 (--fail-on-regression set).")
                sys.exit(1)
        else:
            print("\n✓  No regressions detected vs baseline.")
    else:
        print("\n(No baseline found — skipping regression check.)")

    if save_baseline_flag:
        save_baseline(flat_metrics)

    return results


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * 0.95), len(s) - 1)
    return s[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────


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
        if r.latency_ms:
            print(f"      latency   : {r.latency_ms:.0f}ms  cost: ${r.cost_usd:.6f}")

    print("\n" + "─" * 70)
    print(f"  Result: {passed}/{len(results)} cases passed")
    print("═" * 70 + "\n")

    return passed == len(results)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


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
    parser.add_argument("--run-name", type=str, default="eval")
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--save-baseline", action="store_true")
    args = parser.parse_args()

    if not args.eval_file.exists():
        print(f"Eval file not found: {args.eval_file}")
        sys.exit(1)

    cases = _load_eval_file(args.eval_file)
    results = run_eval(
        cases,
        docs_dir=args.docs_dir,
        run_name=args.run_name,
        fail_on_regression=args.fail_on_regression,
        save_baseline_flag=args.save_baseline,
    )
    all_passed = print_report(results)

    sys.exit(0 if all_passed else 1)
