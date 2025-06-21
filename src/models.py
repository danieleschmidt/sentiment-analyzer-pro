"""Baseline sentiment classifier using Logistic Regression."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = LogisticRegression = Pipeline = None

try:
    from tensorflow import keras
except Exception:  # pragma: no cover - optional dependency
    keras = None

try:
    from transformers import DistilBertConfig, DistilBertForSequenceClassification
except Exception:  # pragma: no cover - optional dependency
    DistilBertConfig = None
    DistilBertForSequenceClassification = None


@dataclass
class SentimentModel:
    pipeline: Pipeline

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)

    def predict(self, texts):
        return self.pipeline.predict(texts)


def build_model() -> SentimentModel:
    if Pipeline is None or TfidfVectorizer is None or LogisticRegression is None:
        raise ImportError("scikit-learn is required for build_model")
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    return SentimentModel(pipeline=pipeline)


def build_lstm_model(
    vocab_size: int = 10000, embed_dim: int = 128, sequence_length: int = 100
) -> keras.Model:
    """Return a simple LSTM-based sentiment classifier."""
    if keras is None:
        raise ImportError("TensorFlow is required for build_lstm_model")
    model = keras.Sequential(
        [
            keras.layers.Embedding(vocab_size, embed_dim, input_length=sequence_length),
            keras.layers.LSTM(64),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_transformer_model(num_labels: int = 2) -> DistilBertForSequenceClassification:
    """Return a minimal DistilBERT model for sentiment classification."""
    if DistilBertForSequenceClassification is None or DistilBertConfig is None:
        raise ImportError("transformers is required for build_transformer_model")
    config = DistilBertConfig(vocab_size=30522, num_labels=num_labels)
    model = DistilBertForSequenceClassification(config)
    return model
