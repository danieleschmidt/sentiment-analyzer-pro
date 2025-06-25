"""Extended evaluation utilities for sentiment models."""

from __future__ import annotations

from typing import Iterable, List

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import KFold
except Exception:  # pragma: no cover - optional dependency
    classification_report = confusion_matrix = accuracy_score = None
    KFold = None

from .models import build_model


def evaluate(true_labels: Iterable[str], predicted_labels: Iterable[str]) -> str:
    """Return a classification report as a string."""
    if classification_report is None:
        raise ImportError("scikit-learn is required for evaluate")
    return classification_report(list(true_labels), list(predicted_labels))


def compute_confusion(true_labels: Iterable[str], predicted_labels: Iterable[str]) -> List[List[int]]:
    """Return a confusion matrix as a nested list."""
    if confusion_matrix is None:
        raise ImportError("scikit-learn is required for compute_confusion")
    matrix = confusion_matrix(list(true_labels), list(predicted_labels))
    return matrix.tolist()


def analyze_errors(
    texts: Iterable[str], true_labels: Iterable[str], predicted_labels: Iterable[str]
) -> pd.DataFrame:
    """Return a DataFrame of misclassified texts with expected and predicted labels."""
    if pd is None:
        raise ImportError("pandas is required for analyze_errors")
    records = [
        {"text": t, "true": y_true, "predicted": y_pred}
        for t, y_true, y_pred in zip(texts, true_labels, predicted_labels)
        if y_true != y_pred
    ]
    return pd.DataFrame(records)


def cross_validate(
    texts: Iterable[str], labels: Iterable[str], folds: int = 5
) -> float:
    """Return mean accuracy from cross-validation."""
    if KFold is None or accuracy_score is None:
        raise ImportError("scikit-learn is required for cross_validate")

    texts = list(texts)
    labels = list(labels)
    if len(texts) < folds:
        raise ValueError("Dataset too small for the number of folds")

    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    accuracies = []
    for train_index, test_index in kf.split(texts):
        model = build_model()
        X_train = [texts[i] for i in train_index]
        y_train = [labels[i] for i in train_index]
        X_test = [texts[i] for i in test_index]
        y_test = [labels[i] for i in test_index]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, preds))
    return sum(accuracies) / len(accuracies)


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate sentiment model")
    parser.add_argument("csv", help="CSV file with 'text' and 'label' columns")
    args = parser.parse_args()

    if pd is None:
        raise SystemExit("pandas is required for CLI usage")
    df = pd.read_csv(args.csv)
    model = build_model()
    model.fit(df["text"], df["label"])
    preds = model.predict(df["text"])
    print(evaluate(df["label"], preds))
    print("Confusion matrix:")
    print(compute_confusion(df["label"], preds))
