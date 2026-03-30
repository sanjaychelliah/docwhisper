"""
telemetry.py — observability layer for docwhisper.

Tracks latency, token usage, cost, and citation metrics per request.
Logs to MLflow and/or W&B when configured; always emits structured JSON
debug logs via the standard logging system.

All external backends are optional: the module loads and functions correctly
even when mlflow/wandb are not installed.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from .config import cfg

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Pricing table — USD per 1M tokens (input, output)
# ─────────────────────────────────────────────────────────────────────────────

PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (5.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "llama3.2": (0.0, 0.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Span:
    """Represents one timed operation within a request."""

    name: str
    start_time: float
    end_time: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0


@dataclass
class RequestTrace:
    """Collects all spans and metrics for one .ask() call."""

    trace_id: str
    question: str
    spans: list[Span]
    started_at: float

    # filled after pipeline completes
    total_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    estimated_cost_usd: float = 0.0
    citation_rate: float = 0.0
    top_rerank_score: float = 0.0
    has_citations: bool = False
    model: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# LatencyTracker
# ─────────────────────────────────────────────────────────────────────────────


class LatencyTracker:
    """Rolling window of latency values for percentile calculations."""

    def __init__(self, window: int = 1000) -> None:
        self._window = window
        self._values: deque[float] = deque(maxlen=window)

    def record(self, latency_ms: float) -> None:
        self._values.append(latency_ms)

    def _sorted(self) -> list[float]:
        return sorted(self._values)

    def _percentile(self, p: float) -> float:
        vals = self._sorted()
        if not vals:
            return 0.0
        idx = int(len(vals) * p / 100)
        idx = min(idx, len(vals) - 1)
        return vals[idx]

    def p50(self) -> float:
        return self._percentile(50)

    def p95(self) -> float:
        return self._percentile(95)

    def p99(self) -> float:
        return self._percentile(99)

    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def count(self) -> int:
        return len(self._values)


# ─────────────────────────────────────────────────────────────────────────────
# CostEstimator
# ─────────────────────────────────────────────────────────────────────────────


class CostEstimator:
    """Estimates LLM cost from token counts and model pricing."""

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Return estimated cost in USD. Returns 0.0 for unknown models."""
        input_price, output_price = PRICING.get(model, (0.0, 0.0))
        cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
        return cost


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────


class Tracker:
    """Main singleton that orchestrates telemetry collection and backend logging."""

    def __init__(self) -> None:
        self._latency = LatencyTracker()
        self._cost_estimator = CostEstimator()
        self._total_cost: float = 0.0
        self._citation_rates: list[float] = []

    @contextmanager
    def span(self, name: str, trace: RequestTrace | None, **metadata) -> Iterator[Span]:
        """Context manager that times a named operation and appends to trace.spans."""
        s = Span(name=name, start_time=time.perf_counter(), metadata=metadata)
        if trace is not None:
            trace.spans.append(s)
        try:
            yield s
        finally:
            s.end_time = time.perf_counter()

    def record(self, trace: RequestTrace) -> None:
        """Flush a completed trace to all configured backends."""
        if not cfg.telemetry_enabled:
            return

        self._latency.record(trace.total_latency_ms)
        self._total_cost += trace.estimated_cost_usd
        self._citation_rates.append(trace.citation_rate)

        # always emit structured JSON at DEBUG level
        payload = {
            "trace_id": trace.trace_id,
            "question": trace.question[:120],
            "model": trace.model,
            "total_latency_ms": round(trace.total_latency_ms, 2),
            "retrieval_latency_ms": round(trace.retrieval_latency_ms, 2),
            "llm_latency_ms": round(trace.llm_latency_ms, 2),
            "tokens_prompt": trace.tokens_prompt,
            "tokens_completion": trace.tokens_completion,
            "estimated_cost_usd": round(trace.estimated_cost_usd, 6),
            "citation_rate": round(trace.citation_rate, 3),
            "top_rerank_score": round(trace.top_rerank_score, 4),
            "has_citations": trace.has_citations,
        }
        logger.debug("telemetry %s", json.dumps(payload))

        # MLflow
        if cfg.mlflow_tracking_uri:
            self._log_mlflow(trace, payload)

        # W&B
        if _wandb_enabled():
            self._log_wandb(trace, payload)

    def _log_mlflow(self, trace: RequestTrace, payload: dict) -> None:
        try:
            import mlflow

            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
            mlflow.set_experiment(cfg.mlflow_experiment)
            with mlflow.start_run(run_name=f"ask-{trace.trace_id[:8]}"):
                mlflow.set_tags({"trace_id": trace.trace_id, "model": trace.model})
                mlflow.log_metrics({
                    "total_latency_ms": trace.total_latency_ms,
                    "retrieval_latency_ms": trace.retrieval_latency_ms,
                    "llm_latency_ms": trace.llm_latency_ms,
                    "tokens_prompt": float(trace.tokens_prompt),
                    "tokens_completion": float(trace.tokens_completion),
                    "estimated_cost_usd": trace.estimated_cost_usd,
                    "citation_rate": trace.citation_rate,
                    "top_rerank_score": trace.top_rerank_score,
                    "has_citations": float(trace.has_citations),
                })
        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)

    def _log_wandb(self, trace: RequestTrace, payload: dict) -> None:
        try:
            import wandb

            wandb.log({
                **payload,
                "trace_id": trace.trace_id,
                "model": trace.model,
            })
        except Exception as exc:
            logger.warning("W&B logging failed: %s", exc)

    def stats(self) -> dict:
        """Return current aggregate stats (for /metrics endpoint)."""
        import os
        mean_citation = (
            sum(self._citation_rates) / len(self._citation_rates)
            if self._citation_rates
            else 0.0
        )
        return {
            "request_count": self._latency.count(),
            "p50_latency_ms": round(self._latency.p50(), 2),
            "p95_latency_ms": round(self._latency.p95(), 2),
            "p99_latency_ms": round(self._latency.p99(), 2),
            "mean_latency_ms": round(self._latency.mean(), 2),
            "citation_rate_mean": round(mean_citation, 4),
            "estimated_cost_usd_total": round(self._total_cost, 6),
            "mlflow_tracking_uri": cfg.mlflow_tracking_uri or None,
            "wandb_enabled": _wandb_enabled(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _wandb_enabled() -> bool:
    import os
    return bool(os.getenv("WANDB_API_KEY"))


def new_trace(question: str) -> RequestTrace:
    """Create a fresh RequestTrace for one .ask() call."""
    return RequestTrace(
        trace_id=uuid.uuid4().hex,
        question=question,
        spans=[],
        started_at=time.time(),
    )


def span_or_null(name: str, trace: RequestTrace | None, tracker: Tracker):
    """Return tracker.span() if trace is provided, else a nullcontext."""
    if trace is not None:
        return tracker.span(name, trace)
    return nullcontext()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

tracker = Tracker()
cost_estimator = CostEstimator()
