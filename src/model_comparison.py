"""Utilities for comparing sentiment models."""

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

try:
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover - optional dependency
    accuracy_score = None
    train_test_split = None

try:
    from tensorflow import keras
except Exception:  # pragma: no cover - optional dependency
    keras = None

from .models import build_lstm_model, build_model
from .preprocessing import clean_text


def compare_models(csv_path: str = "data/sample_reviews.csv"):
    """Train baseline and LSTM models and return accuracy results."""
    if keras is None or pd is None or accuracy_score is None or train_test_split is None:
        raise ImportError("Required ML libraries not installed")
    data = pd.read_csv(csv_path)
    texts = data["text"].apply(clean_text)
    labels = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=0
    )

    results = []

    baseline = build_model()
    baseline.fit(X_train, y_train)
    preds = baseline.predict(X_test)
    results.append(
        {"model": "Logistic Regression", "accuracy": accuracy_score(y_test, preds)}
    )

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(X_train), maxlen=100
    )
    X_test_seq = keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(X_test), maxlen=100
    )
    y_train_bin = (y_train == "positive").astype(int)
    y_test_bin = (y_test == "positive").astype(int)

    lstm_model = build_lstm_model()
    lstm_model.fit(X_train_seq, y_train_bin, epochs=1, batch_size=2, verbose=0)
    lstm_preds = (lstm_model.predict(X_test_seq) > 0.5).astype(int).flatten()
    results.append(
        {"model": "LSTM", "accuracy": accuracy_score(y_test_bin, lstm_preds)}
    )

    return results


if __name__ == "__main__":
    for result in compare_models():
        print(f"{result['model']}: {result['accuracy']:.2f}")
