"""
Self-healing prompt adapter for RetailMind.

Dynamically rewrites the LLM system prompt based on detected semantic drift.
This is the "self-healing" core — the system adapts its behavior in real time
without human intervention when it detects shifting user intent patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_BASE_PROMPT = (
    "You are RetailMind, a knowledgeable and friendly AI shopping assistant for "
    "an online retail store. You help customers find the perfect products from "
    "our catalog.\n\n"
    "RULES:\n"
    "1. ONLY recommend products that appear in the 'Available Inventory' below.\n"
    "2. Always mention the exact product name and price.\n"
    "3. Keep responses concise (3–5 sentences) but helpful.\n"
    "4. If a product matches the customer's needs, explain WHY it's a good fit.\n"
    "5. Never invent products that aren't in the inventory list."
)


@dataclass
class AdaptationRule:
    """A single self-healing rule triggered by a drift concept."""

    concept: str
    label: str
    prompt_injection: str
    explanation: str


# Pre-defined adaptation rules — each maps a drift signal to a prompt mutation
_RULES: dict[str, AdaptationRule] = {
    "price_sensitive": AdaptationRule(
        concept="price_sensitive",
        label="💰 Price-Sensitive Mode",
        prompt_injection=(
            "\n\n⚠️ ACTIVE ADAPTATION — PRICE SENSITIVITY DETECTED:\n"
            "Customer intent analysis shows strong budget-consciousness. "
            "You MUST:\n"
            "• Lead with the cheapest matching products first.\n"
            "• Explicitly state the price and any savings.\n"
            "• Compare price-to-value across options.\n"
            "• Mention if an item is the lowest-priced in its category."
        ),
        explanation=(
            "🔧 Self-Healing Activated\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Signal: Price-sensitive keyword drift detected (budget, cheap, under $X)\n"
            "Action: Injected price-prioritization directives into system prompt\n"
            "Effect: LLM now ranks by price-to-value instead of general relevance\n"
            "Trigger: EWMA score exceeded threshold (0.38)"
        ),
    ),
    "summer_shift": AdaptationRule(
        concept="summer_shift",
        label="☀️ Summer Season Mode",
        prompt_injection=(
            "\n\n⚠️ ACTIVE ADAPTATION — SEASONAL SHIFT DETECTED:\n"
            "Query patterns indicate a seasonal shift toward summer. "
            "You MUST:\n"
            "• Prioritize lightweight, breathable, and warm-weather products.\n"
            "• Highlight UV protection and heat-management features.\n"
            "• De-prioritize winter and cold-weather items.\n"
            "• Mention materials suited for hot climates (linen, mesh, moisture-wicking)."
        ),
        explanation=(
            "🔧 Self-Healing Activated\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Signal: Seasonal semantic shift detected (summer, beach, UV, lightweight)\n"
            "Action: Injected warm-weather prioritization into system prompt\n"
            "Effect: LLM now filters for breathable materials and summer categories\n"
            "Trigger: EWMA score exceeded threshold (0.38)"
        ),
    ),
    "eco_trend": AdaptationRule(
        concept="eco_trend",
        label="🌿 Eco-Conscious Mode",
        prompt_injection=(
            "\n\n⚠️ ACTIVE ADAPTATION — SUSTAINABILITY TREND DETECTED:\n"
            "User intent strongly favors eco-friendly products. "
            "You MUST:\n"
            "• Lead with recycled, organic, and plant-based items.\n"
            "• Highlight environmental certifications (GOTS, OEKO-TEX).\n"
            "• Explain the sustainability story behind each recommendation.\n"
            "• Mention materials: recycled ocean plastic, organic cotton, bamboo, cork."
        ),
        explanation=(
            "🔧 Self-Healing Activated\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Signal: Eco-conscious trend detected (sustainable, recycled, organic)\n"
            "Action: Injected sustainability-first directives into system prompt\n"
            "Effect: LLM now leads with eco-credentials and material sourcing\n"
            "Trigger: EWMA score exceeded threshold (0.38)"
        ),
    ),
}

_NORMAL_EXPLANATION = (
    "📊 System Status: Normal\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "No significant drift detected in user intent patterns.\n"
    "System prompt: Default balanced recommendation mode.\n"
    "All EWMA concept scores below threshold (0.38)."
)


class Adapter:
    """Stateless prompt adapter — maps drift signals to prompt mutations."""

    def __init__(self) -> None:
        self.base_prompt: str = _BASE_PROMPT
        self._active_rule: AdaptationRule | None = None

    def adapt_prompt(self, drift_state: str) -> str:
        """Return the adapted system prompt for the current drift state."""
        rule = _RULES.get(drift_state)
        self._active_rule = rule

        if rule:
            logger.info("Adaptation triggered: %s", rule.label)
            return self.base_prompt + rule.prompt_injection

        return self.base_prompt + "\n\nProvide balanced recommendations covering a mix of features, prices, and styles."

    def get_explanation(self, drift_state: str) -> str:
        """Human-readable explanation of what the adapter did and why."""
        rule = _RULES.get(drift_state)
        return rule.explanation if rule else _NORMAL_EXPLANATION

    def get_label(self, drift_state: str) -> str:
        """Short UI label for the active state."""
        rule = _RULES.get(drift_state)
        return rule.label if rule else "⚖️ Balanced Mode"
