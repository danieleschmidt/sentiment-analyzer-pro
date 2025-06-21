"""Prediction script for baseline model."""

from .models import SentimentModel


def main(input_csv: str, model_path: str = "model.joblib"):
    """Load a trained model and print predictions for a CSV file."""
    import joblib
    import pandas as pd

    data = pd.read_csv(input_csv)
    model: SentimentModel = joblib.load(model_path)
    predictions = model.predict(data["text"])
    for text, pred in zip(data["text"], predictions):
        print(f"{text} => {pred}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict sentiment for reviews.")
    parser.add_argument("csv", help="CSV file with a 'text' column")
    parser.add_argument("--model", default="model.joblib", help="Trained model path")
    args = parser.parse_args()
    main(args.csv, args.model)
