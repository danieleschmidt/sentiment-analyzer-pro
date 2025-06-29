"""Training script for baseline model."""

import logging
import os

from .models import build_model


def main(csv_path: str = "data/sample_reviews.csv", model_path: str = os.getenv("MODEL_PATH", "model.joblib")):
    """Train the baseline model on a CSV and save to disk."""
    import pandas as pd

    data = pd.read_csv(csv_path)
    model = build_model()
    model.fit(data["text"], data["label"])

    # Save the trained model
    import joblib

    joblib.dump(model, model_path)
    logging.getLogger(__name__).info("Model saved to %s", model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument("--csv", default="data/sample_reviews.csv", help="Training data CSV")
    parser.add_argument("--model", default=os.getenv("MODEL_PATH", "model.joblib"), help="Where to save the model")
    args = parser.parse_args()
    logging.basicConfig(format="%(message)s", level=logging.INFO, force=True)
    main(args.csv, args.model)
