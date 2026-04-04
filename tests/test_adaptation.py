"""
Unit tests for the self-healing adapter.
"""

import pytest
from modules.adaptation import Adapter


@pytest.fixture
def adapter():
    return Adapter()


class TestAdapter:
    """Tests for the prompt adaptation engine."""

    def test_normal_returns_base_prompt(self, adapter):
        prompt = adapter.adapt_prompt("normal")
        assert "RetailMind" in prompt
        assert "ACTIVE ADAPTATION" not in prompt

    def test_price_sensitive_injects_rules(self, adapter):
        prompt = adapter.adapt_prompt("price_sensitive")
        assert "PRICE SENSITIVITY" in prompt
        assert "cheapest" in prompt.lower()

    def test_summer_shift_injects_rules(self, adapter):
        prompt = adapter.adapt_prompt("summer_shift")
        assert "SEASONAL SHIFT" in prompt
        assert "lightweight" in prompt.lower()

    def test_eco_trend_injects_rules(self, adapter):
        prompt = adapter.adapt_prompt("eco_trend")
        assert "SUSTAINABILITY" in prompt
        assert "recycled" in prompt.lower() or "organic" in prompt.lower()

    def test_explanation_differs_per_state(self, adapter):
        explanations = set()
        for state in ["normal", "price_sensitive", "summer_shift", "eco_trend"]:
            explanations.add(adapter.get_explanation(state))
        assert len(explanations) == 4, "Each state should produce a unique explanation"

    def test_label_differs_per_state(self, adapter):
        labels = set()
        for state in ["normal", "price_sensitive", "summer_shift", "eco_trend"]:
            labels.add(adapter.get_label(state))
        assert len(labels) == 4, "Each state should produce a unique label"

    def test_base_prompt_contains_anti_hallucination(self, adapter):
        prompt = adapter.adapt_prompt("normal")
        assert "ONLY recommend" in prompt or "only recommend" in prompt.lower()
