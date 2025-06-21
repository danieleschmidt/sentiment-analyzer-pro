"""Training script for baseline model."""

from .models import build_model


def main(csv_path: str = "data/sample_reviews.csv", model_path: str = "model.joblib"):
    """Train the baseline model on a CSV and save to disk."""
    import pandas as pd

    data = pd.read_csv(csv_path)
    model = build_model()
    model.fit(data["text"], data["label"])

    # Save the trained model
    import joblib

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument("--csv", default="data/sample_reviews.csv", help="Training data CSV")
    parser.add_argument("--model", default="model.joblib", help="Where to save the model")
    args = parser.parse_args()

    main(args.csv, args.model)
