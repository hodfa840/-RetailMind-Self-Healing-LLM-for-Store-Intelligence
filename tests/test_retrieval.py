"""
Unit tests for hybrid retrieval engine.
"""

import pytest
from modules.data_simulation import generate_catalog
from modules.retrieval import HybridRetriever


@pytest.fixture(scope="module")
def retriever():
    catalog = generate_catalog()
    return HybridRetriever(catalog)


class TestHybridRetriever:
    """Tests for the hybrid retrieval system."""

    def test_returns_correct_count(self, retriever):
        results = retriever.search("running shoes", top_k=4)
        assert len(results) == 4

    def test_results_have_scores(self, retriever):
        results = retriever.search("water bottle")
        for r in results:
            assert "score" in r
            assert "product" in r
            assert 0.0 <= r["score"] <= 1.0

    def test_price_filtering_under_30(self, retriever):
        results = retriever.search("shoes under $30", top_k=4)
        for r in results:
            assert r["product"]["price"] <= 30.0, (
                f"Product '{r['product']['title']}' costs ${r['product']['price']} "
                f"but should be under $30"
            )

    def test_price_filtering_under_50(self, retriever):
        results = retriever.search("I only have $50 to spend", top_k=4)
        for r in results:
            assert r["product"]["price"] <= 50.0

    def test_eco_category_relevance(self, retriever):
        results = retriever.search("eco-friendly sustainable products", top_k=4)
        eco_count = sum(1 for r in results if r["product"]["category"] == "eco-friendly")
        assert eco_count >= 2, f"Expected ≥2 eco products, got {eco_count}"

    def test_winter_category_relevance(self, retriever):
        results = retriever.search("warm winter jacket for cold weather", top_k=4)
        winter_count = sum(1 for r in results if r["product"]["category"] == "winter")
        assert winter_count >= 2, f"Expected ≥2 winter products, got {winter_count}"

    def test_results_sorted_by_score(self, retriever):
        results = retriever.search("fitness watch with GPS", top_k=4)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"

    def test_empty_query_returns_results(self, retriever):
        results = retriever.search("", top_k=4)
        assert len(results) == 4  # Should gracefully handle empty queries
