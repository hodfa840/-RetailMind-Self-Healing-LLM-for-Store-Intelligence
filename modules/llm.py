"""
Local LLM inference engine for RetailMind.

Uses Qwen2.5-0.5B-Instruct running entirely on CPU — no API keys, no GPU,
no external dependencies.  Prompt engineering is tuned to minimize
hallucination by grounding all answers in the provided product context.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

_generator = None


def _get_pipeline():
    """Lazy-load the text-generation pipeline (singleton)."""
    global _generator
    if _generator is None:
        logger.info("Loading Qwen2.5-0.5B-Instruct on CPU (first call only)…")
        t0 = time.time()
        _generator = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            device="cpu",
            torch_dtype=torch.float32,
        )
        logger.info("Model loaded in %.1fs", time.time() - t0)
    return _generator


def generate_response(
    system_prompt: str,
    user_query: str,
    retrieved_items: list[dict[str, Any]],
) -> str:
    """
    Generate a grounded product recommendation.

    The retrieved items are injected directly into the system prompt so
    the model can only reference real products.
    """
    # Build structured context from retrieved products
    context_lines = []
    for i, r in enumerate(retrieved_items, 1):
        p = r["product"]
        stars = "★" * int(p.get("rating", 4)) + "☆" * (5 - int(p.get("rating", 4)))
        context_lines.append(
            f"{i}. {p['title']} — ${p['price']:.2f}\n"
            f"   Category: {p['category']} | Rating: {stars} ({p.get('reviews', 0)} reviews)\n"
            f"   Materials: {p.get('materials', 'N/A')}\n"
            f"   Description: {p['desc']}"
        )

    context = "\n\n".join(context_lines)

    messages = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                f"══════ Available Inventory ══════\n\n"
                f"{context}\n\n"
                f"══════════════════════════════════\n"
                f"IMPORTANT: Only recommend from the products listed above. "
                f"Cite exact names and prices."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    try:
        gen = _get_pipeline()
        result = gen(
            messages,
            max_new_tokens=250,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            return_full_text=False,
        )
        generated = result[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1]["content"]
        return generated
    except Exception as e:
        logger.exception("LLM inference failed")
        return f"[RetailMind] I encountered an issue generating a response. Error: {e}"
