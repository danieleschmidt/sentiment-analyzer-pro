# Evaluation Guide

This document outlines how to perform in-depth evaluation of sentiment models.

The `src.evaluate` module provides the following utilities:

- `evaluate` – returns a text classification report.
- `compute_confusion` – returns a confusion matrix as a list of lists.
- `analyze_errors` – returns a DataFrame with misclassified examples.

Example usage:

```python
from src import build_model, evaluate, compute_confusion, analyze_errors
import pandas as pd

reviews = pd.read_csv("data/sample_reviews.csv")
model = build_model()
model.fit(reviews["text"], reviews["label"])
preds = model.predict(reviews["text"])
print(evaluate(reviews["label"], preds))
print(compute_confusion(reviews["label"], preds))
print(analyze_errors(reviews["text"], reviews["label"], preds))
```

These utilities help identify where the model is making mistakes so you can
improve data quality and model architecture.
