import pytest

from src.evaluate import cross_validate


def test_cross_validate_returns_mean_accuracy():
    pytest.importorskip("sklearn")
    texts = ["good", "bad", "good", "bad"]
    labels = ["positive", "negative", "positive", "negative"]
    acc = cross_validate(texts, labels, folds=2)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_cross_validate_custom_folds():
    pytest.importorskip("sklearn")
    texts = ["good", "bad", "good", "bad", "good", "bad"]
    labels = ["positive", "negative", "positive", "negative", "positive", "negative"]
    acc = cross_validate(texts, labels, folds=3)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_cross_validate_insufficient_data():
    pytest.importorskip("sklearn")
    texts = ["good", "bad"]
    labels = ["positive", "negative"]
    with pytest.raises(ValueError):
        cross_validate(texts, labels, folds=3)
