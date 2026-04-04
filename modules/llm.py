"""
LLM inference engine for RetailMind.

Uses the HuggingFace Inference API (serverless, GPU-backed) so responses
arrive in ~1–2 s instead of 15–20 s on CPU.  Falls back to a structured
template if the API is unavailable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

_client: InferenceClient | None = None
MODEL = "Qwen/Qwen2.5-72B-Instruct"   # strong model, free on HF serverless


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        _client = InferenceClient(token=token)
        logger.info("InferenceClient ready (model=%s)", MODEL)
    return _client


def _build_context(retrieved_items: list[dict[str, Any]]) -> str:
    lines = []
    for i, r in enumerate(retrieved_items, 1):
        p = r["product"]
        stars = "★" * int(p.get("rating", 4)) + "☆" * (5 - int(p.get("rating", 4)))
        lines.append(
            f"{i}. {p['title']} — ${p['price']:.2f}\n"
            f"   Category: {p['category']} | Rating: {stars} ({p.get('reviews', 0)} reviews)\n"
            f"   Materials: {p.get('materials', 'N/A')}\n"
            f"   Description: {p['desc']}"
        )
    return "\n\n".join(lines)


def _fallback_response(retrieved_items: list[dict[str, Any]]) -> str:
    """Structured template used when the API is unavailable."""
    if not retrieved_items:
        return "I couldn't find matching products for your query. Try different keywords."
    lines = ["Here are my top picks for you:\n"]
    for r in retrieved_items:
        p = r["product"]
        lines.append(f"• **{p['title']}** — ${p['price']:.2f}\n  {p['desc'][:120]}…")
    return "\n".join(lines)


def generate_response(
    system_prompt: str,
    user_query: str,
    retrieved_items: list[dict[str, Any]],
) -> str:
    context = _build_context(retrieved_items)
    messages = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                f"══════ Available Inventory ══════\n\n"
                f"{context}\n\n"
                f"════════════════════════════════\n"
                f"You are a helpful AI shopping assistant. "
                f"Only recommend products listed above. "
                f"Cite exact names and prices. Be concise (2–4 sentences)."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    try:
        client = _get_client()
        result = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=150,
            temperature=0.3,
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Inference API failed (%s), using fallback template.", e)
        return _fallback_response(retrieved_items)
