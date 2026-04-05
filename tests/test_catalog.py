"""
Unit tests for RetailMind core modules.

Run with: pytest tests/ -v
"""

from modules.data_simulation import generate_catalog, get_scenarios


class TestCatalog:
    """Tests for the product catalog generator."""

    def test_catalog_size(self):
        catalog = generate_catalog()
        assert len(catalog) >= 50, f"Expected at least 50 products, got {len(catalog)}"

    def test_product_has_required_fields(self):
        catalog = generate_catalog()
        required = {"id", "title", "category", "price", "desc", "tags", "rating", "reviews", "materials"}
        for p in catalog[:5]:
            missing = required - set(p.keys())
            assert not missing, f"Product {p['id']} missing fields: {missing}"

    def test_prices_are_positive(self):
        catalog = generate_catalog()
        for p in catalog:
            assert p["price"] > 0, f"Product {p['id']} has non-positive price: {p['price']}"

    def test_ratings_in_range(self):
        catalog = generate_catalog()
        for p in catalog:
            assert 1.0 <= p["rating"] <= 5.0, f"Product {p['id']} has invalid rating: {p['rating']}"

    def test_categories_are_valid(self):
        valid = {"winter", "summer", "eco-friendly", "sports", "electronics", "premium", "home", "casual", "health"}
        catalog = generate_catalog()
        for p in catalog:
            assert p["category"] in valid, f"Invalid category: {p['category']}"

    def test_unique_ids(self):
        catalog = generate_catalog()
        ids = [p["id"] for p in catalog]
        assert len(ids) == len(set(ids)), "Duplicate product IDs found"

    def test_scenarios_not_empty(self):
        scenarios = get_scenarios()
        assert len(scenarios) >= 4, "Expected at least 4 scenario phases"
        for name, queries in scenarios.items():
            assert len(queries) >= 3, f"Scenario '{name}' has too few queries"
