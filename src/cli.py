"""Simple command-line interface for Sentiment Analyzer Pro."""

from __future__ import annotations

import argparse
from importlib import metadata

from . import compute_confusion
from .evaluate import evaluate, analyze_errors
from .preprocessing import clean_text, remove_stopwords
from .predict import main as predict_main
from .train import main as train_main


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
    train_p.add_argument("--model", default="model.joblib", help="Path to save model")

    predict_p = sub.add_parser("predict", help="Predict sentiments")
    predict_p.add_argument("csv", help="CSV with a 'text' column")
    predict_p.add_argument("--model", default="model.joblib", help="Trained model path")

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

    summary_p = sub.add_parser("summary", help="Show dataset statistics")
    summary_p.add_argument("csv", help="CSV with a 'text' column")
    summary_p.add_argument(
        "--top",
        type=int,
        default=0,
        help="List the N most common words",
    )

    serve_p = sub.add_parser("serve", help="Run the web prediction server")
    serve_p.add_argument("--model", default="model.joblib", help="Trained model path")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", default=5000, type=int)

    sub.add_parser("version", help="Show package version")

    args = parser.parse_args(argv)
    if args.verbose:
        print(f"Verbosity level: {args.verbose}")

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
    elif args.cmd == "analyze":
        import pandas as pd
        from .models import build_model

        df = pd.read_csv(args.csv)
        model = build_model()
        model.fit(df["text"], df["label"])
        preds = model.predict(df["text"])
        errors = analyze_errors(df["text"], df["label"], preds)
        if errors.empty:
            print("No errors found.")
        else:
            print(errors.to_string(index=False))
    elif args.cmd == "preprocess":
        import pandas as pd

        df = pd.read_csv(args.csv)
        df["text"] = df["text"].apply(clean_text)
        if args.remove_stopwords:
            df["text"] = (
                df["text"].str.split().apply(remove_stopwords).str.join(" ")
            )
        df.to_csv(args.out, index=False)
        print(f"Wrote cleaned data to {args.out}")
    elif args.cmd == "split":
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(args.csv)
        train_df, test_df = train_test_split(
            df, test_size=args.ratio, random_state=0
        )
        train_df.to_csv(args.train, index=False)
        test_df.to_csv(args.test, index=False)
        print(
            f"Wrote {len(train_df)} rows to {args.train} and {len(test_df)} rows to {args.test}"
        )
    elif args.cmd == "summary":
        import pandas as pd

        df = pd.read_csv(args.csv)
        print(f"Rows: {len(df)}")
        avg_len = df["text"].str.split().apply(len).mean()
        print(f"Avg words: {avg_len:.2f}")
        if "label" in df.columns:
            counts = df["label"].value_counts()
            print("Label counts:")
            for label, count in counts.items():
                print(f"{label}: {count}")
        if args.top:
            from collections import Counter

            tokens = df["text"].str.lower().str.split().explode()
            common = Counter(tokens).most_common(args.top)
            print(f"Top {args.top} words:")
            for word, count in common:
                print(f"{word}: {count}")
    elif args.cmd == "serve":
        from . import webapp

        webapp.main(
            [
                "--model",
                args.model,
                "--host",
                args.host,
                "--port",
                str(args.port),
            ]
        )
    elif args.cmd == "version":
        print(version)
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
