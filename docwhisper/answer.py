"""
answer.py — LLM answer generation with hard citation enforcement.

The prompt forces the model to cite chunk IDs inline [1], [2] etc.
If the answer has no citations, we flag it rather than silently return garbage.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .config import cfg
from .retrieve import RetrievedChunk

if TYPE_CHECKING:
    from .telemetry import RequestTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer container
# ---------------------------------------------------------------------------


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[dict]         # [{"ref": "[1]", "source": "...", "excerpt": "..."}]
    retrieved_chunks: list[RetrievedChunk]
    has_citations: bool = False
    model: str = ""
    trace: RequestTrace | None = field(default=None, repr=False)

    def format(self) -> str:
        """Pretty-print for CLI output."""
        lines = [
            "",
            "━" * 60,
            f"Q: {self.question}",
            "━" * 60,
            "",
            self.answer,
            "",
        ]
        if self.citations:
            lines.append("── Sources ──────────────────────────────────────────────")
            for c in self.citations:
                lines.append(f"  {c['ref']}  {c['source']}")
                lines.append(f"      \"{c['excerpt']}\"")
                lines.append("")
        else:
            lines.append("⚠  No citations found in this answer.")
        lines.append("━" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are a precise document assistant. Answer questions using ONLY the provided context chunks.

Rules:
- Cite every factual claim using inline references like [1], [2], etc.
- Each reference number corresponds to the chunk number in the context.
- If the context doesn't contain enough information, say so clearly — do not make things up.
- Keep your answer concise and focused.
- Do not repeat the question back.
"""

CONTEXT_TEMPLATE = """Context chunks:

{chunks}

Question: {question}

Answer (with inline citations like [1], [2]):"""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, rc in enumerate(chunks, start=1):
        source_name = rc.chunk.source.split("/")[-1]  # just the filename
        parts.append(f"[{i}] (source: {source_name}, score: {rc.score:.3f})\n{rc.chunk.text}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call — OpenAI-compatible
# ---------------------------------------------------------------------------


def _call_llm(system: str, user: str) -> tuple[str, str, int, int]:
    """Returns (response_text, model_name, prompt_tokens, completion_tokens)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    if not cfg.llm_api_key:
        raise ValueError(
            "No LLM API key found. Set OPENAI_API_KEY environment variable.\n"
            "You can also point OPENAI_API_BASE to any OpenAI-compatible endpoint (Ollama, Groq, etc.)"
        )

    client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_api_base)

    response = client.chat.completions.create(
        model=cfg.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
    )
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return response.choices[0].message.content, response.model, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------------
# Citation parser
# ---------------------------------------------------------------------------


def _parse_citations(answer_text: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """Extract citation refs from answer and map back to source chunks."""
    refs_in_answer = sorted(set(re.findall(r"\[(\d+)\]", answer_text)))
    citations = []
    for ref_num in refs_in_answer:
        idx = int(ref_num) - 1
        if 0 <= idx < len(chunks):
            rc = chunks[idx]
            citations.append({
                "ref": f"[{ref_num}]",
                "source": rc.chunk.source,
                "excerpt": rc.chunk.text[:120].replace("\n", " ") + "...",
                "score": rc.score,
            })
    return citations


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    *,
    trace: RequestTrace | None = None,
) -> Answer:
    """Generate a cited answer from retrieved chunks."""
    if not retrieved:
        return Answer(
            question=question,
            answer="I couldn't find relevant information in the indexed documents to answer this question.",
            citations=[],
            retrieved_chunks=[],
            has_citations=False,
            trace=trace,
        )

    context = _build_context(retrieved)
    user_prompt = CONTEXT_TEMPLATE.format(chunks=context, question=question)

    logger.info("Calling LLM (%s)...", cfg.llm_model)

    if trace is not None:
        from .telemetry import tracker, cost_estimator
        with tracker.span("llm_call", trace):
            answer_text, model_used, prompt_tokens, completion_tokens = _call_llm(SYSTEM_PROMPT, user_prompt)
        trace.llm_latency_ms = next(
            (s.latency_ms for s in reversed(trace.spans) if s.name == "llm_call"), 0.0
        )
        trace.tokens_prompt = prompt_tokens
        trace.tokens_completion = completion_tokens
        trace.model = model_used
        trace.estimated_cost_usd = cost_estimator.estimate(model_used, prompt_tokens, completion_tokens)
    else:
        answer_text, model_used, _, _ = _call_llm(SYSTEM_PROMPT, user_prompt)

    citations = _parse_citations(answer_text, retrieved)
    has_citations = len(citations) >= cfg.min_citation_count

    if cfg.require_citations and not has_citations:
        logger.warning(
            "Answer has %d citations but %d required. "
            "The model may have answered from general knowledge.",
            len(citations), cfg.min_citation_count
        )

    return Answer(
        question=question,
        answer=answer_text,
        citations=citations,
        retrieved_chunks=retrieved,
        has_citations=has_citations,
        model=model_used,
        trace=trace,
    )
