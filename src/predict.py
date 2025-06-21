"""Prediction script for baseline model."""

import joblib
import pandas as pd

from .models import SentimentModel


def main(input_csv: str):
    data = pd.read_csv(input_csv)
    model: SentimentModel = joblib.load("model.joblib")
    predictions = model.predict(data["text"])
    for text, pred in zip(data["text"], predictions):
        print(f"{text} => {pred}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict sentiment for reviews.")
    parser.add_argument("csv", help="CSV file with a 'text' column")
    args = parser.parse_args()
    main(args.csv)
