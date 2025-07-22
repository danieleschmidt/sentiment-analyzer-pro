"""Training script for baseline model."""

import logging
import os

from .models import build_model


def main(csv_path: str = "data/sample_reviews.csv", model_path: str = os.getenv("MODEL_PATH", "model.joblib")):
    """Train the baseline model on a CSV and save to disk."""
    import pandas as pd
    import joblib

    logger = logging.getLogger(__name__)
    
    # Load and validate training data
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"Training CSV file not found: {csv_path}")
        raise SystemExit(f"Training CSV file not found: {csv_path}")
    except pd.errors.EmptyDataError:
        logger.error(f"Training CSV file is empty: {csv_path}")
        raise SystemExit(f"Training CSV file is empty: {csv_path}")
    except pd.errors.ParserError as exc:
        logger.error(f"Invalid CSV format in {csv_path}: {exc}")
        raise SystemExit(f"Invalid CSV format in {csv_path}: {exc}")
    except PermissionError:
        logger.error(f"Permission denied reading {csv_path}")
        raise SystemExit(f"Permission denied reading {csv_path}")
    
    # Validate required columns
    required_columns = ["text", "label"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"Required columns {missing_columns} not found in {csv_path}. Available columns: {list(data.columns)}")
        raise SystemExit(f"Required columns {missing_columns} not found in {csv_path}")
    
    # Validate data quality
    if len(data) == 0:
        logger.error(f"No training data found in {csv_path}")
        raise SystemExit(f"No training data found in {csv_path}")
    
    if data["text"].isna().all():
        logger.error(f"All text values are missing in {csv_path}")
        raise SystemExit(f"All text values are missing in {csv_path}")
    
    if data["label"].isna().all():
        logger.error(f"All label values are missing in {csv_path}")
        raise SystemExit(f"All label values are missing in {csv_path}")
    
    # Filter out rows with missing values
    clean_data = data.dropna(subset=["text", "label"])
    if len(clean_data) == 0:
        logger.error(f"No valid training samples after removing missing values in {csv_path}")
        raise SystemExit(f"No valid training samples after removing missing values in {csv_path}")
    
    if len(clean_data) < len(data):
        logger.warning(f"Removed {len(data) - len(clean_data)} rows with missing text or label values")
    
    # Build and train model
    try:
        model = build_model()
        logger.info(f"Training model on {len(clean_data)} samples...")
        model.fit(clean_data["text"], clean_data["label"])
    except ValueError as exc:
        logger.error(f"Training failed due to invalid data: {exc}")
        raise SystemExit(f"Training failed due to invalid data: {exc}")
    except Exception as exc:
        logger.error(f"Unexpected error during training: {exc}")
        raise SystemExit(f"Unexpected error during training: {exc}")

    # Save the trained model
    try:
        # Ensure the directory exists
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(model, model_path)
        logger.info("Model saved to %s", model_path)
    except PermissionError:
        logger.error(f"Permission denied writing model to {model_path}")
        raise SystemExit(f"Permission denied writing model to {model_path}")
    except OSError as exc:
        logger.error(f"OS error saving model to {model_path}: {exc}")
        raise SystemExit(f"Failed to save model to {model_path}: {exc}")
    except Exception as exc:
        logger.error(f"Unexpected error saving model to {model_path}: {exc}")
        raise SystemExit(f"Unexpected error saving model to {model_path}: {exc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument("--csv", default="data/sample_reviews.csv", help="Training data CSV")
    parser.add_argument("--model", default=os.getenv("MODEL_PATH", "model.joblib"), help="Where to save the model")
    args = parser.parse_args()
    logging.basicConfig(format="%(message)s", level=logging.INFO, force=True)
    try:
        main(args.csv, args.model)
    except SystemExit:
        raise
    except Exception as exc:
        logging.getLogger(__name__).error(f"Unexpected error: {exc}")
        raise SystemExit(f"Unexpected error: {exc}")
