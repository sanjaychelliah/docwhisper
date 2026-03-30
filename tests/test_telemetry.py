"""
tests/test_telemetry.py — unit tests for the telemetry module.

No external services required — all tests run without mlflow or wandb installed.
"""

from __future__ import annotations

import time

import pytest

from docwhisper.telemetry import (
    CostEstimator,
    LatencyTracker,
    RequestTrace,
    Span,
    Tracker,
    new_trace,
)
from docwhisper.eval import check_regression


# ─────────────────────────────────────────────────────────────────────────────
# Span
# ─────────────────────────────────────────────────────────────────────────────


class TestSpan:
    def test_latency_ms_basic(self):
        s = Span(name="test", start_time=1.0, end_time=1.5)
        assert abs(s.latency_ms - 500.0) < 0.001

    def test_latency_ms_zero_when_no_end(self):
        s = Span(name="test", start_time=1.0)
        assert s.latency_ms == 0.0

    def test_latency_ms_small(self):
        s = Span(name="test", start_time=0.0, end_time=0.001)
        assert abs(s.latency_ms - 1.0) < 0.0001


# ─────────────────────────────────────────────────────────────────────────────
# LatencyTracker
# ─────────────────────────────────────────────────────────────────────────────


class TestLatencyTracker:
    def test_empty(self):
        t = LatencyTracker()
        assert t.p50() == 0.0
        assert t.p95() == 0.0
        assert t.p99() == 0.0
        assert t.mean() == 0.0
        assert t.count() == 0

    def test_single_value(self):
        t = LatencyTracker()
        t.record(100.0)
        assert t.p50() == 100.0
        assert t.p95() == 100.0
        assert t.count() == 1

    def test_percentiles_known_inputs(self):
        t = LatencyTracker()
        for v in range(1, 101):   # 1..100
            t.record(float(v))
        # p50 → index 50 of sorted [1..100] = value 51 (0-indexed floor)
        assert t.p50() == 51.0
        # p95 → index 95 = value 96
        assert t.p95() == 96.0
        # p99 → index 99 = value 100
        assert t.p99() == 100.0

    def test_mean(self):
        t = LatencyTracker()
        for v in [10.0, 20.0, 30.0]:
            t.record(v)
        assert abs(t.mean() - 20.0) < 0.001

    def test_window_capped(self):
        t = LatencyTracker(window=5)
        for i in range(10):
            t.record(float(i))
        assert t.count() == 5


# ─────────────────────────────────────────────────────────────────────────────
# CostEstimator
# ─────────────────────────────────────────────────────────────────────────────


class TestCostEstimator:
    def test_known_model_gpt4o_mini(self):
        est = CostEstimator()
        # 1M input tokens → $0.15, 1M output → $0.60
        cost = est.estimate("gpt-4o-mini", 1_000_000, 1_000_000)
        assert abs(cost - 0.75) < 0.0001

    def test_known_model_gpt4o(self):
        est = CostEstimator()
        cost = est.estimate("gpt-4o", 1_000_000, 1_000_000)
        assert abs(cost - 20.0) < 0.0001

    def test_local_model_free(self):
        est = CostEstimator()
        cost = est.estimate("llama3.2", 100_000, 50_000)
        assert cost == 0.0

    def test_unknown_model_returns_zero(self):
        est = CostEstimator()
        cost = est.estimate("some-unknown-model-xyz", 500_000, 200_000)
        assert cost == 0.0

    def test_zero_tokens(self):
        est = CostEstimator()
        cost = est.estimate("gpt-4o-mini", 0, 0)
        assert cost == 0.0

    def test_small_token_count(self):
        est = CostEstimator()
        # 1000 prompt tokens with gpt-4o-mini: 1000 * 0.15 / 1e6 = 0.00000015... ≈ 1.5e-4 cents
        cost = est.estimate("gpt-4o-mini", 1000, 0)
        expected = 1000 * 0.15 / 1_000_000
        assert abs(cost - expected) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────


class TestTracker:
    def test_record_no_backends(self, monkeypatch):
        """record() should not raise even when no backends are configured."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
        monkeypatch.delenv("WANDB_API_KEY", raising=False)

        from docwhisper import config as cfg_mod
        cfg_mod.cfg.mlflow_tracking_uri = ""
        cfg_mod.cfg.telemetry_enabled = True

        tracker = Tracker()
        trace = new_trace("test question")
        trace.total_latency_ms = 500.0
        trace.citation_rate = 0.8
        trace.estimated_cost_usd = 0.001
        trace.model = "gpt-4o-mini"
        trace.has_citations = True

        # should not raise
        tracker.record(trace)
        assert tracker._latency.count() == 1

    def test_span_context_manager(self):
        tracker = Tracker()
        trace = new_trace("test")
        with tracker.span("test_span", trace):
            time.sleep(0.01)
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "test_span"
        assert trace.spans[0].latency_ms > 5  # at least 5ms

    def test_span_with_none_trace(self):
        """span() should work with trace=None (returns a no-op span)."""
        tracker = Tracker()
        with tracker.span("noop", None) as s:
            pass
        assert s.name == "noop"

    def test_stats_empty(self):
        tracker = Tracker()
        stats = tracker.stats()
        assert stats["request_count"] == 0
        assert stats["p50_latency_ms"] == 0.0
        assert stats["estimated_cost_usd_total"] == 0.0

    def test_stats_after_records(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        from docwhisper import config as cfg_mod
        cfg_mod.cfg.mlflow_tracking_uri = ""
        cfg_mod.cfg.telemetry_enabled = True

        tracker = Tracker()
        for latency in [100.0, 200.0, 300.0]:
            trace = new_trace("q")
            trace.total_latency_ms = latency
            trace.citation_rate = 1.0
            tracker.record(trace)

        stats = tracker.stats()
        assert stats["request_count"] == 3
        assert stats["mean_latency_ms"] == 200.0


# ─────────────────────────────────────────────────────────────────────────────
# check_regression
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckRegression:
    def test_no_regression(self):
        baseline = {"pass_rate": 0.9, "citation_rate": 0.85, "mean_latency_ms": 1000.0, "total_cost_usd": 0.01}
        current  = {"pass_rate": 0.9, "citation_rate": 0.85, "mean_latency_ms": 1050.0, "total_cost_usd": 0.01}
        msgs = check_regression(current, baseline)
        assert msgs == []

    def test_pass_rate_regression(self):
        baseline = {"pass_rate": 0.9, "citation_rate": 0.85, "mean_latency_ms": 1000.0, "total_cost_usd": 0.01}
        current  = {"pass_rate": 0.7, "citation_rate": 0.85, "mean_latency_ms": 1000.0, "total_cost_usd": 0.01}
        msgs = check_regression(current, baseline)
        assert any("pass_rate" in m for m in msgs)

    def test_latency_regression(self):
        baseline = {"pass_rate": 0.9, "citation_rate": 0.85, "mean_latency_ms": 1000.0, "total_cost_usd": 0.01}
        current  = {"pass_rate": 0.9, "citation_rate": 0.85, "mean_latency_ms": 1600.0, "total_cost_usd": 0.01}
        msgs = check_regression(current, baseline)
        assert any("mean_latency_ms" in m for m in msgs)

    def test_cost_regression(self):
        baseline = {"pass_rate": 0.9, "citation_rate": 0.85, "mean_latency_ms": 1000.0, "total_cost_usd": 0.01}
        current  = {"pass_rate": 0.9, "citation_rate": 0.85, "mean_latency_ms": 1000.0, "total_cost_usd": 0.10}
        msgs = check_regression(current, baseline)
        assert any("total_cost_usd" in m for m in msgs)

    def test_multiple_regressions(self):
        baseline = {"pass_rate": 0.9, "citation_rate": 0.85, "mean_latency_ms": 1000.0, "total_cost_usd": 0.01}
        current  = {"pass_rate": 0.5, "citation_rate": 0.5, "mean_latency_ms": 2000.0, "total_cost_usd": 0.10}
        msgs = check_regression(current, baseline)
        assert len(msgs) >= 3

    def test_missing_baseline_key_skipped(self):
        baseline = {"pass_rate": 0.9}
        current  = {"pass_rate": 0.5, "citation_rate": 0.5}
        msgs = check_regression(current, baseline)
        # only pass_rate should trigger (citation_rate not in baseline)
        assert any("pass_rate" in m for m in msgs)
        assert not any("citation_rate" in m for m in msgs)
