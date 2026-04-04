"""
Hybrid retrieval engine for RetailMind.

Combines dense semantic search (SentenceTransformers) with structured
metadata filtering (price range, category, tags) so that queries like
"eco-friendly bag under $30" actually return relevant, correctly-priced items.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from modules.shared import get_embedding_model

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Two-stage retriever: metadata pre-filter → semantic re-rank."""

    def __init__(self, catalog: list[dict]) -> None:
        self.catalog = catalog
        self.model = get_embedding_model()

        # Build rich embedding texts that capture all searchable facets
        texts = [
            (
                f"{p['title']}. {p['desc']} "
                f"Category: {p['category']}. "
                f"Materials: {p.get('materials', 'N/A')}. "
                f"Tags: {', '.join(p.get('tags', []))}."
            )
            for p in catalog
        ]
        logger.info("Encoding %d products…", len(catalog))
        self.embeddings = self.model.encode(texts, show_progress_bar=False)
        self._norms = np.linalg.norm(self.embeddings, axis=1)
        logger.info("Catalog indexed successfully.")

    # ── Public API ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 4,
        category_filter: str | None = None,
        query_emb=None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k products for *query*.

        Pipeline:
        1. Extract price ceiling from natural language (e.g. "under $50").
        2. Pre-filter catalog by price / category if applicable.
        3. Rank remaining items by cosine similarity.
        4. Return top-k with scores.

        Pass *query_emb* to skip re-encoding when the caller already has it.
        """
        price_cap = self._extract_price_cap(query)
        cat_hint = category_filter or self._extract_category_hint(query)

        # Stage 1 — metadata pre-filter
        candidate_indices = self._prefilter(price_cap, cat_hint)

        # Stage 2 — semantic ranking over candidates
        if query_emb is None:
            query_emb = self.model.encode([query], show_progress_bar=False)[0]
        query_norm = np.linalg.norm(query_emb)

        if len(candidate_indices) == 0:
            # Fallback: rank entire catalog if filters yield nothing
            candidate_indices = list(range(len(self.catalog)))

        cand_embs = self.embeddings[candidate_indices]
        cand_norms = self._norms[candidate_indices]

        scores = np.dot(cand_embs, query_emb) / (cand_norms * query_norm + 1e-10)
        top_local = np.argsort(scores)[::-1][:top_k]

        results = []
        for li in top_local:
            if float(scores[li]) < 0.20:
                continue
            global_idx = candidate_indices[li]
            results.append({
                "product": self.catalog[global_idx],
                "score": float(scores[li]),
            })

        logger.debug(
            "Query: %r | price_cap=%s | cat=%s | candidates=%d | top=%d",
            query, price_cap, cat_hint, len(candidate_indices), len(results),
        )
        return results

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_price_cap(query: str) -> float | None:
        """Parse 'under $50', 'below 30', 'less than $25', 'budget' etc."""
        patterns = [
            r"under\s*\$?\s*(\d+(?:\.\d+)?)",
            r"below\s*\$?\s*(\d+(?:\.\d+)?)",
            r"less\s+than\s*\$?\s*(\d+(?:\.\d+)?)",
            r"cheaper\s+than\s*\$?\s*(\d+(?:\.\d+)?)",
            r"max(?:imum)?\s*\$?\s*(\d+(?:\.\d+)?)",
            r"\$(\d+(?:\.\d+)?)\s*(?:or\s+less|max|budget)",
            r"only\s+have\s*\$?\s*(\d+)",
            r"(?:spend|budget)\s*(?:of|is)?\s*\$?\s*(\d+)",
        ]
        for pat in patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                return float(m.group(1))

        # Heuristic: very budget-oriented queries
        budget_keywords = {"cheapest", "budget", "affordable", "inexpensive", "bargain"}
        if any(kw in query.lower() for kw in budget_keywords):
            return 50.0  # Reasonable default budget ceiling

        return None

    def _extract_category_hint(self, query: str) -> str | None:
        """Map common query terms to catalog categories."""
        category_keywords: dict[str, list[str]] = {
            "winter": ["winter", "cold", "snow", "warm", "insulated", "thermal"],
            "summer": ["summer", "beach", "hot", "heat", "sun", "warm weather"],
            "eco-friendly": ["eco", "sustainable", "organic", "recycled", "green", "environment", "plant-based"],
            "sports": ["sport", "fitness", "running", "gym", "training", "workout", "athletic"],
            "electronics": ["tech", "electronic", "gadget", "headphone", "speaker", "charger", "smart"],
            "premium": ["luxury", "premium", "high-end", "designer", "artisan"],
            "home": ["home", "kitchen", "desk", "candle", "bath", "decor"],
            "health": ["health", "beauty", "sunscreen", "lipstick", "serum", "balm", "skincare", "makeup"],
        }
        for cat, keywords in category_keywords.items():
            pattern = r'\b(?:' + '|'.join(keywords) + r')\b'
            if re.search(pattern, query, re.IGNORECASE):
                return cat
        return None

    def _prefilter(
        self, price_cap: float | None, category: str | None
    ) -> list[int]:
        """Return indices of products matching hard constraints."""
        indices = []
        for i, p in enumerate(self.catalog):
            if price_cap is not None and p["price"] > price_cap:
                continue
            if category is not None and p["category"] != category:
                continue
            indices.append(i)
        return indices
