# Evaluation Guide

This document outlines how to perform in-depth evaluation of sentiment models.

The `src.evaluate` module provides the following utilities:

- `evaluate` – returns a text classification report.
- `compute_confusion` – returns a confusion matrix as a list of lists.
- `analyze_errors` – returns a DataFrame with misclassified examples.
- `cross_validate` – runs stratified k-fold validation and returns the mean score.

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
score = cross_validate(reviews["text"], reviews["label"], folds=5)
print(f"CV accuracy: {score:.2f}")
# StratifiedKFold is used to keep label distribution consistent across folds
```

`cross_validate` accepts an optional `scorer` callable if you want to
evaluate metrics other than accuracy. The `folds` parameter must be an
integer greater than one. ``texts`` and ``labels`` must be the same length.
For example, to compute macro F1 instead:

```python
from sklearn.metrics import f1_score
score = cross_validate(
    reviews["text"],
    reviews["label"],
    folds=5,
    scorer=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
)
```

You can supply a custom model builder to evaluate different architectures:

```python
from src.models import build_nb_model
score = cross_validate(
    reviews["text"],
    reviews["label"],
    folds=5,
    model_fn=build_nb_model,
)
```

You can also run cross-validation from the command line:

```bash
sentiment-cli crossval data/sample_reviews.csv --folds 5
sentiment-cli crossval data/sample_reviews.csv --metric f1
```

These utilities help identify where the model is making mistakes so you can
improve data quality and model architecture.
