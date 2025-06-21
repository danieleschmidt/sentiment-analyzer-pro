import pytest

from src.models import build_lstm_model, build_model, build_transformer_model


def test_model_fit_predict():
    pytest.importorskip("sklearn")
    texts = ["good", "bad"]
    labels = ["positive", "negative"]
    model = build_model()
    model.fit(texts, labels)
    preds = model.predict(["good", "bad"])
    assert list(preds) == ["positive", "negative"]


def test_build_lstm_model():
    pytest.importorskip("tensorflow")
    model = build_lstm_model()
    assert isinstance(model.layers[-1].activation.__name__, str)


def test_build_transformer_model():
    pytest.importorskip("transformers")
    model = build_transformer_model()
    assert model.config.num_labels == 2
