"""Prediction script for baseline model."""

import logging
import os

from .models import SentimentModel


def main(input_csv: str, model_path: str = os.getenv("MODEL_PATH", "model.joblib")):
    """Load a trained model and print predictions for a CSV file."""
    import joblib
    import pandas as pd

    data = pd.read_csv(input_csv)
    model: SentimentModel = joblib.load(model_path)
    predictions = model.predict(data["text"])
    logger = logging.getLogger(__name__)
    for text, pred in zip(data["text"], predictions):
        logger.info("%s => %s", text, pred)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict sentiment for reviews.")
    parser.add_argument("csv", help="CSV file with a 'text' column")
    parser.add_argument("--model", default=os.getenv("MODEL_PATH", "model.joblib"), help="Trained model path")
    args = parser.parse_args()
    logging.basicConfig(format="%(message)s", level=logging.INFO, force=True)
    main(args.csv, args.model)
