# docwhisper Observability Guide

docwhisper has a built-in telemetry layer that tracks latency, token usage, cost, and citation
quality per request. It supports two optional backends — **MLflow** and **Weights & Biases** —
and always emits structured JSON debug logs via Python's standard `logging` module.

---

## Quick start — MLflow locally

```bash
pip install -e ".[observability]"
mlflow ui --port 5000 &
export MLFLOW_TRACKING_URI=http://localhost:5000
docwhisper ask "What is the return policy?"
# open http://localhost:5000 to see the logged run
```

Every `.ask()` call creates one MLflow run with all metrics listed below.

---

## What gets tracked

| Metric | Description | Unit |
|---|---|---|
| `total_latency_ms` | End-to-end wall time for one `.ask()` call | ms |
| `retrieval_latency_ms` | Combined BM25 + vector search + reranking time | ms |
| `llm_latency_ms` | Time waiting for the LLM API response | ms |
| `tokens_prompt` | Tokens sent to the LLM | count |
| `tokens_completion` | Tokens received from the LLM | count |
| `estimated_cost_usd` | Estimated API cost based on model pricing | USD |
| `citation_rate` | Cited chunks / `rerank_top_k` | 0–1 |
| `top_rerank_score` | Cross-encoder score of the top-ranked chunk | float |
| `has_citations` | Whether the answer has at least one citation | bool |

Span-level timing (BM25, vector, rerank, llm_call) is also recorded inside each trace
and logged at `DEBUG` level via `docwhisper.telemetry` logger.

---

## W&B setup

Set your API key and (optionally) project/entity:

```bash
export WANDB_API_KEY=your_key_here
export WANDB_PROJECT=docwhisper          # default: "docwhisper"
export WANDB_ENTITY=your_team           # optional, defaults to personal account
```

W&B logging activates automatically when `WANDB_API_KEY` is set. It runs alongside
MLflow — both backends can be active at the same time.

---

## Running the eval suite with MLflow

```bash
python -m docwhisper.eval \
  --eval-file examples/eval_questions.json \
  --run-name my-experiment
```

This logs aggregate metrics to an MLflow run and saves a JSON artifact with per-case results.
Metrics logged:

- `eval/citation_rate`
- `eval/relevance_rate`
- `eval/source_recall`
- `eval/pass_rate`
- `eval/mean_latency_ms`
- `eval/p95_latency_ms`
- `eval/total_cost_usd`
- `eval/mean_rerank_score`

---

## Regression gating

### How baselines work

A baseline is a snapshot of aggregate eval metrics saved to `.docwhisper_baselines.json`.
On each subsequent eval run, the current metrics are compared against the baseline.
If any metric degrades beyond its allowed threshold, a regression message is printed.

Default thresholds:

| Metric | Max allowed degradation |
|---|---|
| `pass_rate` | drop of 0.10 (10 percentage points) |
| `citation_rate` | drop of 0.10 |
| `mean_latency_ms` | increase of 500 ms |
| `total_cost_usd` | increase of $0.05 |

### Setting an initial baseline

```bash
python -m docwhisper.eval \
  --eval-file examples/eval_questions.json \
  --save-baseline
```

### Failing CI on regression

```bash
python -m docwhisper.eval \
  --eval-file examples/eval_questions.json \
  --fail-on-regression
```

Exits with code 1 if any regression is detected. The CI `regression-gate` job uses this flag.

---

## The `/metrics` endpoint

When running the REST server, current aggregate stats are available at `GET /metrics`.

```bash
curl http://localhost:8000/metrics
```

Example response:

```json
{
  "request_count": 42,
  "p50_latency_ms": 1240.5,
  "p95_latency_ms": 2890.1,
  "p99_latency_ms": 4100.3,
  "mean_latency_ms": 1380.2,
  "citation_rate_mean": 0.87,
  "estimated_cost_usd_total": 0.034,
  "mlflow_tracking_uri": "http://localhost:5000",
  "wandb_enabled": false
}
```

Stats are kept in memory and reset when the server restarts. The rolling window holds the
last 1000 requests for percentile calculations.

---

## Docker Compose — MLflow as a sidecar

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.13.0
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0

  docwhisper:
    build: .
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on: [mlflow]
    ports: ["8000:8000"]
    command: uvicorn docwhisper.server:app --host 0.0.0.0 --port 8000
```

---

## Disabling telemetry

Set `DOCWHISPER_TELEMETRY=false` to skip all telemetry processing. The pipeline runs
exactly as before — no performance overhead, no external calls.

```bash
export DOCWHISPER_TELEMETRY=false
```

---

## Structured debug logs

Every request emits a JSON line at `DEBUG` level regardless of backend configuration:

```
DEBUG docwhisper.telemetry telemetry {"trace_id": "...", "question": "...", "total_latency_ms": 1234.5, ...}
```

Capture with:

```bash
DOCWHISPER_TELEMETRY=true python -c "
import logging; logging.basicConfig(level=logging.DEBUG)
from docwhisper.pipeline import DocWhisper
dw = DocWhisper(); dw.load()
dw.ask('What is the return policy?')
" 2>&1 | grep telemetry
```
