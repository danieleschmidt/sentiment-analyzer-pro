"""Training script for baseline model."""

import pandas as pd

from .models import build_model


def main():
    data = pd.read_csv("data/sample_reviews.csv")
    model = build_model()
    model.fit(data["text"], data["label"])

    # Save the trained model
    import joblib

    joblib.dump(model, "model.joblib")
    print("Model saved to model.joblib")


if __name__ == "__main__":
    main()
