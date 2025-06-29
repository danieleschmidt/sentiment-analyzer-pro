"""Simple command-line interface for Sentiment Analyzer Pro."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from importlib import metadata

from . import compute_confusion
from .evaluate import evaluate, analyze_errors, cross_validate
from .preprocessing import (
    clean_series,
    remove_stopwords,
    lemmatize_tokens,
)
from .schemas import validate_columns
from .predict import main as predict_main
from .train import main as train_main

logger = logging.getLogger(__name__)

MODEL_DEFAULT = os.getenv("MODEL_PATH", "model.joblib")


def load_csv(path: str, required: list[str] | None = None):
    import pandas as pd

    df = pd.read_csv(path)
    if required:
        try:
            validate_columns(df.columns, required)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    return df


def cmd_train(args) -> None:
    load_csv(args.csv, ["text", "label"])
    train_main(args.csv, args.model)


def cmd_predict(args) -> None:
    load_csv(args.csv, ["text"])
    predict_main(args.csv, args.model)


def cmd_eval(args) -> None:
    from .models import build_model

    df = load_csv(args.csv, ["text", "label"])
    model = build_model()
    model.fit(df["text"], df["label"])
    preds = model.predict(df["text"])
    logger.info(evaluate(df["label"], preds))
    logger.info(compute_confusion(df["label"], preds))


def cmd_analyze(args) -> None:
    from .models import build_model

    df = load_csv(args.csv, ["text", "label"])
    model = build_model()
    model.fit(df["text"], df["label"])
    preds = model.predict(df["text"])
    errors = analyze_errors(df["text"], df["label"], preds)
    if errors.empty:
        logger.info("No errors found.")
    else:
        logger.info(errors.to_string(index=False))


def cmd_preprocess(args) -> None:
    df = load_csv(args.csv, ["text"])
    df["text"] = clean_series(df["text"])
    if args.lemmatize or args.remove_stopwords:
        df["text"] = df["text"].str.split()
        if args.lemmatize:
            df["text"] = df["text"].apply(lemmatize_tokens)
        if args.remove_stopwords:
            df["text"] = df["text"].apply(remove_stopwords)
        df["text"] = df["text"].str.join(" ")
    df.to_csv(args.out, index=False)
    logger.info(f"Wrote cleaned data to {args.out}")


def cmd_split(args) -> None:
    from sklearn.model_selection import train_test_split

    df = load_csv(args.csv, ["text", "label"])
    train_df, test_df = train_test_split(df, test_size=args.ratio, random_state=0)
    train_df.to_csv(args.train, index=False)
    test_df.to_csv(args.test, index=False)
    logger.info(
        f"Wrote {len(train_df)} rows to {args.train} and {len(test_df)} rows to {args.test}"
    )


def cmd_crossval(args) -> None:
    from .models import build_model, build_nb_model

    df = load_csv(args.csv, ["text", "label"])
    model_fn = build_nb_model if args.nb else build_model
    scorer = None
    if args.metric == "f1":
        from sklearn.metrics import f1_score

        def scorer(y_true, y_pred) -> float:
            return f1_score(y_true, y_pred, average="macro")

    score = cross_validate(
        df["text"],
        df["label"],
        folds=args.folds,
        model_fn=model_fn,
        scorer=scorer,
    )
    logger.info(f"Cross-val score: {score:.2f}")


def cmd_summary(args) -> None:
    from collections import Counter

    df = load_csv(args.csv, ["text"])
    logger.info(f"Rows: {len(df)}")
    avg_len = df["text"].str.split().apply(len).mean()
    logger.info(f"Avg words: {avg_len:.2f}")
    if "label" in df.columns:
        counts = df["label"].value_counts()
        logger.info("Label counts:")
        for label, count in counts.items():
            logger.info(f"{label}: {count}")
    if args.top:
        tokens = df["text"].str.lower().str.findall(r"\b\w+\b").explode()
        common = Counter(tokens).most_common(args.top)
        logger.info(f"Top {args.top} words:")
        for word, count in common:
            logger.info(f"{word}: {count}")


def cmd_serve(args) -> None:
    from . import webapp

    webapp.main([
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ])


def cmd_version(args) -> None:
    try:
        version = metadata.version("sentiment-analyzer-pro")
    except metadata.PackageNotFoundError:
        version = "0.0.0"
    print(version)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sentiment Analyzer CLI")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity",
    )
    try:
        version = metadata.version("sentiment-analyzer-pro")
    except metadata.PackageNotFoundError:
        version = "0.0.0"
    parser.add_argument("--version", action="version", version=version)
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train", help="Train the model")
    train_p.add_argument(
        "--csv", default="data/sample_reviews.csv", help="Training CSV"
    )
    train_p.add_argument("--model", default=MODEL_DEFAULT, help="Path to save model")

    predict_p = sub.add_parser("predict", help="Predict sentiments")
    predict_p.add_argument("csv", help="CSV with a 'text' column")
    predict_p.add_argument("--model", default=MODEL_DEFAULT, help="Trained model path")

    eval_p = sub.add_parser("eval", help="Quick evaluation on a labeled CSV")
    eval_p.add_argument("csv", help="CSV with 'text' and 'label' columns")

    analyze_p = sub.add_parser("analyze", help="Show misclassified texts")
    analyze_p.add_argument("csv", help="CSV with 'text' and 'label' columns")

    preprocess_p = sub.add_parser("preprocess", help="Clean a CSV of text")
    preprocess_p.add_argument("csv", help="Input CSV with a 'text' column")
    preprocess_p.add_argument("--out", default="clean.csv", help="Output CSV")
    preprocess_p.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove common stop words",
    )
    preprocess_p.add_argument(
        "--lemmatize",
        action="store_true",
        help="Apply lemmatization to tokens",
    )

    split_p = sub.add_parser("split", help="Split dataset into train/test files")
    split_p.add_argument("csv", help="CSV with 'text' and 'label' columns")
    split_p.add_argument("--train", default="train.csv", help="Output training CSV")
    split_p.add_argument("--test", default="test.csv", help="Output test CSV")
    split_p.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Proportion to use as the test set",
    )

    cross_p = sub.add_parser(
        "crossval", help="Evaluate model using k-fold cross-validation"
    )
    cross_p.add_argument("csv", help="CSV with 'text' and 'label' columns")
    cross_p.add_argument(
        "--folds", type=int, default=5, help="Number of cross-validation folds"
    )
    cross_p.add_argument(
        "--nb", action="store_true", help="Use Naive Bayes model instead of Logistic Regression"
    )
    cross_p.add_argument(
        "--metric",
        choices=["accuracy", "f1"],
        default="accuracy",
        help="Scoring metric (accuracy or macro F1)",
    )

    summary_p = sub.add_parser("summary", help="Show dataset statistics")
    summary_p.add_argument("csv", help="CSV with a 'text' column")
    summary_p.add_argument(
        "--top",
        type=int,
        default=0,
        help="List the N most common words",
    )

    serve_p = sub.add_parser("serve", help="Run the web prediction server")
    serve_p.add_argument("--model", default=MODEL_DEFAULT, help="Trained model path")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", default=5000, type=int)

    sub.add_parser("version", help="Show package version")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True
    )
    if args.verbose:
        logger.info(f"Verbosity level: {args.verbose}")

    commands = {
        "train": cmd_train,
        "predict": cmd_predict,
        "eval": cmd_eval,
        "analyze": cmd_analyze,
        "preprocess": cmd_preprocess,
        "split": cmd_split,
        "crossval": cmd_crossval,
        "summary": cmd_summary,
        "serve": cmd_serve,
        "version": cmd_version,
    }

    commands[args.cmd](args)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
