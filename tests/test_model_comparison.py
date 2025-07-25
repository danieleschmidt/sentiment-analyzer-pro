import pytest

from src.model_comparison import compare_models


def test_compare_models_returns_results():
    try:
        results = compare_models("data/sample_reviews.csv")
    except ImportError:
        pytest.skip("Required ML libraries not installed")
    assert isinstance(results, list)
    assert len(results) >= 1
    assert "model" in results[0] and "accuracy" in results[0]
