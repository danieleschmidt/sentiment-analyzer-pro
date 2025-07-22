"""Prediction script for baseline model."""

import logging
import os

from .models import SentimentModel


def main(input_csv: str, model_path: str = os.getenv("MODEL_PATH", "model.joblib")):
    """Load a trained model and print predictions for a CSV file."""
    import joblib
    import pandas as pd

    logger = logging.getLogger(__name__)
    
    # Load and validate input CSV
    try:
        data = pd.read_csv(input_csv)
    except FileNotFoundError:
        logger.error(f"Input CSV file not found: {input_csv}")
        raise SystemExit(f"Input CSV file not found: {input_csv}")
    except pd.errors.EmptyDataError:
        logger.error(f"Input CSV file is empty: {input_csv}")
        raise SystemExit(f"Input CSV file is empty: {input_csv}")
    except pd.errors.ParserError as exc:
        logger.error(f"Invalid CSV format in {input_csv}: {exc}")
        raise SystemExit(f"Invalid CSV format in {input_csv}: {exc}")
    except PermissionError:
        logger.error(f"Permission denied reading {input_csv}")
        raise SystemExit(f"Permission denied reading {input_csv}")
    
    # Validate required columns
    if "text" not in data.columns:
        logger.error(f"Required 'text' column not found in {input_csv}. Available columns: {list(data.columns)}")
        raise SystemExit(f"Required 'text' column not found in {input_csv}")
    
    if len(data) == 0 or data["text"].isna().all():
        logger.error(f"No valid text data found in {input_csv}")
        raise SystemExit(f"No valid text data found in {input_csv}")
    
    # Load trained model
    try:
        model: SentimentModel = joblib.load(model_path)
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise SystemExit(f"Model file not found: {model_path}")
    except (EOFError, ValueError, TypeError) as exc:
        logger.error(f"Invalid or corrupted model file {model_path}: {exc}")
        raise SystemExit(f"Invalid or corrupted model file {model_path}")
    except PermissionError:
        logger.error(f"Permission denied reading model file: {model_path}")
        raise SystemExit(f"Permission denied reading model file: {model_path}")
    
    # Make predictions
    try:
        # Filter out null text values for prediction
        valid_data = data[data["text"].notna()]
        if len(valid_data) == 0:
            logger.warning("No valid text data found for prediction")
            return
        
        predictions = model.predict(valid_data["text"])
        for text, pred in zip(valid_data["text"], predictions):
            logger.info("%s => %s", text, pred)
            
        if len(valid_data) < len(data):
            logger.warning(f"Skipped {len(data) - len(valid_data)} rows with missing text values")
            
    except (ValueError, AttributeError) as exc:
        logger.error(f"Model prediction failed: {exc}")
        raise SystemExit(f"Model prediction failed: {exc}")
    except Exception as exc:
        logger.error(f"Unexpected error during prediction: {exc}")
        raise SystemExit(f"Unexpected error during prediction: {exc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict sentiment for reviews.")
    parser.add_argument("csv", help="CSV file with a 'text' column")
    parser.add_argument("--model", default=os.getenv("MODEL_PATH", "model.joblib"), help="Trained model path")
    args = parser.parse_args()
    logging.basicConfig(format="%(message)s", level=logging.INFO, force=True)
    try:
        main(args.csv, args.model)
    except SystemExit:
        raise
    except Exception as exc:
        logging.getLogger(__name__).error(f"Unexpected error: {exc}")
        raise SystemExit(f"Unexpected error: {exc}")
