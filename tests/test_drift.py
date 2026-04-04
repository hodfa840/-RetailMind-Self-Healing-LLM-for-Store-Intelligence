"""
Unit tests for the drift detection module.
"""

import pytest
from modules.drift import DriftDetector


@pytest.fixture
def detector():
    return DriftDetector()


class TestDriftDetector:
    """Tests for semantic drift detection."""

    def test_normal_query_no_drift(self, detector):
        drift, scores = detector.analyze_drift("I need a good water bottle.")
        assert drift == "normal", f"Expected 'normal', got '{drift}'"
        assert all(isinstance(v, float) for v in scores.values())

    def test_price_sensitive_detection(self, detector):
        # Feed multiple budget-oriented queries to build up EWMA
        for q in ["cheapest option", "budget under $20", "show me the cheapest"]:
            drift, _ = detector.analyze_drift(q)
        assert drift == "price_sensitive", f"Expected 'price_sensitive' after budget queries, got '{drift}'"

    def test_eco_trend_detection(self, detector):
        for q in ["sustainable organic products", "eco-friendly recycled", "I want plant-based items"]:
            drift, _ = detector.analyze_drift(q)
        assert drift == "eco_trend", f"Expected 'eco_trend' after eco queries, got '{drift}'"

    def test_summer_shift_detection(self, detector):
        for q in ["summer beach sandals", "hot weather lightweight", "UV protection for sun"]:
            drift, _ = detector.analyze_drift(q)
        assert drift == "summer_shift", f"Expected 'summer_shift' after summer queries, got '{drift}'"

    def test_scores_have_all_concepts(self, detector):
        _, scores = detector.analyze_drift("test query")
        expected = {"price_sensitive", "summer_shift", "eco_trend"}
        assert set(scores.keys()) == expected

    def test_history_accumulates(self, detector):
        for i in range(5):
            detector.analyze_drift(f"query {i}")
        assert len(detector.history) == 5

    def test_ewma_scores_available(self, detector):
        detector.analyze_drift("some query")
        ewma = detector.get_ewma_scores()
        assert isinstance(ewma, dict)
        assert len(ewma) == 3

    def test_history_series_length_matches(self, detector):
        for i in range(10):
            detector.analyze_drift(f"query {i}")
        series = detector.get_history_series()
        for concept, data in series.items():
            assert len(data) == 10, f"{concept} series length {len(data)} != 10"
