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


def test_cross_validate_custom_scorer():
    pytest.importorskip("sklearn")
    from sklearn.metrics import f1_score

    texts = ["good", "bad", "good", "bad"]
    labels = ["positive", "negative", "positive", "negative"]

    score = cross_validate(
        texts,
        labels,
        folds=2,
        scorer=lambda y_true, y_pred: f1_score(y_true, y_pred, pos_label="positive"),
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_cross_validate_custom_model_fn():
    pytest.importorskip("sklearn")
    from src.models import build_nb_model

    texts = ["good", "bad", "good", "bad"]
    labels = ["positive", "negative", "positive", "negative"]

    score = cross_validate(texts, labels, folds=2, model_fn=build_nb_model)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_cross_validate_insufficient_data():
    pytest.importorskip("sklearn")
    texts = ["good", "bad"]
    labels = ["positive", "negative"]
    with pytest.raises(ValueError):
        cross_validate(texts, labels, folds=3)


def test_cross_validate_invalid_folds():
    pytest.importorskip("sklearn")
    texts = ["good", "bad", "good"]
    labels = ["pos", "neg", "pos"]
    with pytest.raises(ValueError):
        cross_validate(texts, labels, folds=1)


def test_cross_validate_non_int_folds():
    pytest.importorskip("sklearn")
    texts = ["good", "bad", "good"]
    labels = ["pos", "neg", "pos"]
    with pytest.raises(TypeError):
        cross_validate(texts, labels, folds=2.5)


def test_cross_validate_length_mismatch():
    pytest.importorskip("sklearn")
    texts = ["good", "bad", "good"]
    labels = ["pos", "neg"]
    with pytest.raises(ValueError):
        cross_validate(texts, labels, folds=2)
