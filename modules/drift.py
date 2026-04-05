"""
Semantic drift detector for RetailMind.

Tracks the rolling semantic similarity of incoming user queries against
predefined *concept anchors* (e.g., price-sensitivity, seasonal shift,
eco-trend).  When the exponentially-weighted moving average for any concept
exceeds a configurable threshold the system flags an active drift — which
triggers the self-healing adapter to rewrite the LLM system prompt.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modules.shared import get_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class DriftEvent:
    """Immutable record of a single drift measurement."""

    timestamp: float
    query: str
    scores: dict[str, float]
    dominant: str


@dataclass
class DriftDetector:
    """
    Monitors semantic drift across configurable concept anchors.

    Uses EWMA (exponentially weighted moving average) to smooth noisy
    single-query scores into stable trend signals.
    """

    threshold: float = 0.38
    ewma_alpha: float = 0.35          # smoothing factor (higher = more reactive)
    history: list[DriftEvent] = field(default_factory=list)
    _ewma: dict[str, float] = field(default_factory=dict)
    _concept_embs: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        model = get_embedding_model()
        # Multiple anchor phrases per concept → averaged embedding for robustness
        concept_phrases = {
            "price_sensitive": [
                "cheap budget discount low price clearance sale savings affordable",
                "what is the cheapest option under twenty dollars bargain deal",
                "I only have a limited budget, show me value picks",
            ],
            "summer_shift": [
                "summer heat warm weather sandals shorts sunscreen beach",
                "lightweight breathable sun protection hot climate UV",
                "vacation tropical poolside outdoor warm temperature",
            ],
            "eco_trend": [
                "eco-friendly sustainable organic recycled environment green",
                "plant-based carbon-neutral zero waste biodegradable vegan",
                "responsible sourcing ethical production renewable materials",
            ],
        }
        for concept, phrases in concept_phrases.items():
            embs = model.encode(phrases, show_progress_bar=False)
            self._concept_embs[concept] = np.mean(embs, axis=0)
            self._ewma[concept] = 0.0

        logger.info("DriftDetector initialized with %d concept anchors.", len(concept_phrases))
    # ── Public API ──────────────────────────────────────────────────────────

    def analyze_drift(
        self, query: str, query_emb=None
    ) -> tuple[str, dict[str, float]]:
        """
        Score *query* against all concept anchors and return
        ``(dominant_concept, raw_scores)``.

        Pass *query_emb* to skip re-encoding when the caller already has it.
        """
        if query_emb is None:
            query_emb = get_embedding_model().encode([query], show_progress_bar=False)[0]

        raw_scores: dict[str, float] = {}
        for concept, ref_emb in self._concept_embs.items():
            sim = float(
                np.dot(query_emb, ref_emb)
                / (np.linalg.norm(query_emb) * np.linalg.norm(ref_emb) + 1e-10)
            )
            raw_scores[concept] = sim

            # Update EWMA
            prev = self._ewma[concept]
            self._ewma[concept] = self.ewma_alpha * sim + (1 - self.ewma_alpha) * prev

        # Determine dominant drift from smoothed signal
        detected = "normal"
        max_smoothed = 0.0
        for concept, smoothed in self._ewma.items():
            if smoothed > self.threshold and smoothed > max_smoothed:
                max_smoothed = smoothed
                detected = concept

        event = DriftEvent(
            timestamp=time.time(),
            query=query,
            scores=raw_scores,
            dominant=detected,
        )
        self.history.append(event)
        if len(self.history) > 200:
            self.history = self.history[-200:]

        logger.debug("Drift analysis: %s | scores=%s | ewma=%s", detected, raw_scores, self._ewma)
        return detected, raw_scores

    def get_ewma_scores(self) -> dict[str, float]:
        """Return current EWMA-smoothed scores for dashboard display."""
        return dict(self._ewma)

    def get_recent_stats(self) -> dict[str, float] | None:
        """Return averaged raw scores from last N queries."""
        if not self.history:
            return None
        recent = self.history[-5:]
        concepts = list(self._concept_embs.keys())
        return {
            c: float(np.mean([e.scores[c] for e in recent]))
            for c in concepts
        }

    def get_history_series(self) -> dict[str, list[float]]:
        """Return full EWMA time-series for each concept (for charts).

        Pads with baseline values when fewer than 5 real events exist so the
        chart renders a smooth baseline line on first load.
        """
        series: dict[str, list[float]] = {c: [] for c in self._concept_embs}
        ewma_state = {c: 0.0 for c in self._concept_embs}

        # Pad with neutral baseline so chart always has something to show
        padding = max(0, 5 - len(self.history))
        for _ in range(padding):
            for c in self._concept_embs:
                series[c].append(0.15)

        for event in self.history:
            for c in self._concept_embs:
                ewma_state[c] = self.ewma_alpha * event.scores[c] + (1 - self.ewma_alpha) * ewma_state[c]
                series[c].append(ewma_state[c])
        return series
