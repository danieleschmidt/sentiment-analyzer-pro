"""Simple evaluation metrics for the sentiment model."""

from sklearn.metrics import classification_report


def evaluate(true_labels, predicted_labels) -> str:
    """Return a classification report."""
    return classification_report(true_labels, predicted_labels)
