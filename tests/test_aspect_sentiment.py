import pytest

from src.aspect_sentiment import extract_aspects, predict_aspect_sentiment
from src.models import build_model


def test_extract_aspects_returns_nouns():
    text = "The battery life of this phone is great"
    aspects = extract_aspects(text)
    assert "battery" in aspects and "life" in aspects and "phone" in aspects


def test_predict_aspect_sentiment_uses_model():
    pytest.importorskip("sklearn")
    model = build_model()
    model.fit(["good", "bad"], ["positive", "negative"])
    result = predict_aspect_sentiment("Good camera", model)
    assert result
    for aspect, sentiment in result.items():
        assert sentiment == "positive"
