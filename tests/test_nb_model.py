import pytest
from src.models import build_nb_model, SentimentModel


def test_build_nb_model_returns_sentiment_model():
    pytest.importorskip("sklearn")
    model = build_nb_model()
    assert isinstance(model, SentimentModel)


def test_nb_model_predicts_correct_labels():
    pytest.importorskip("sklearn")
    model = build_nb_model()
    texts = ["good", "bad"]
    labels = ["positive", "negative"]
    model.fit(texts, labels)
    assert list(model.predict(texts)) == labels


def test_build_nb_model_missing_sklearn(monkeypatch):
    monkeypatch.setattr('src.models.MultinomialNB', None)
    monkeypatch.setattr('src.models.Pipeline', None)
    monkeypatch.setattr('src.models.TfidfVectorizer', None)
    with pytest.raises(ImportError):
        build_nb_model()

