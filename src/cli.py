"""Simple command-line interface for Sentiment Analyzer Pro."""

from __future__ import annotations

import argparse

from . import evaluate, compute_confusion
from .predict import main as predict_main
from .train import main as train_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sentiment Analyzer CLI")
    sub = parser.add_subparsers(dest="cmd")

    train_p = sub.add_parser("train", help="Train the model")
    train_p.add_argument("--csv", default="data/sample_reviews.csv", help="Training CSV")
    train_p.add_argument("--model", default="model.joblib", help="Path to save model")

    predict_p = sub.add_parser("predict", help="Predict sentiments")
    predict_p.add_argument("csv", help="CSV with a 'text' column")
    predict_p.add_argument("--model", default="model.joblib", help="Trained model path")

    eval_p = sub.add_parser("eval", help="Quick evaluation on a labeled CSV")
    eval_p.add_argument("csv", help="CSV with 'text' and 'label' columns")

    args = parser.parse_args(argv)

    if args.cmd == "train":
        train_main(args.csv, args.model)
    elif args.cmd == "predict":
        predict_main(args.csv, args.model)
    elif args.cmd == "eval":
        import pandas as pd
        from .models import build_model

        df = pd.read_csv(args.csv)
        model = build_model()
        model.fit(df["text"], df["label"])
        preds = model.predict(df["text"])
        print(evaluate(df["label"], preds))
        print(compute_confusion(df["label"], preds))
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
