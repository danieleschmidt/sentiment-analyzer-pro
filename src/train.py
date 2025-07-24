"""Training script for baseline model."""

import logging
import os
import time

from .models import build_model
from .logging_config import (
    setup_logging, get_logger, log_data_processing, 
    log_training_event, log_model_operation, log_system_event
)


def main(csv_path: str = "data/sample_reviews.csv", model_path: str = os.getenv("MODEL_PATH", "model.joblib")):
    """Train the baseline model on a CSV and save to disk."""
    import pandas as pd
    import joblib

    logger = get_logger(__name__)
    
    log_system_event(logger, 'startup', 'training', {
        'csv_path': csv_path,
        'model_path': model_path
    })
    
    # Load and validate training data
    load_start = time.time()
    try:
        data = pd.read_csv(csv_path)
        load_duration = time.time() - load_start
        log_data_processing(logger, 'load', len(data), load_duration, csv_path)
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
    preprocess_start = time.time()
    clean_data = data.dropna(subset=["text", "label"])
    preprocess_duration = time.time() - preprocess_start
    
    if len(clean_data) == 0:
        logger.error(f"No valid training samples after removing missing values in {csv_path}")
        raise SystemExit(f"No valid training samples after removing missing values in {csv_path}")
    
    removed_rows = len(data) - len(clean_data)
    if removed_rows > 0:
        logger.warning(f"Removed {removed_rows} rows with missing text or label values")
    
    log_data_processing(logger, 'preprocess', len(clean_data), preprocess_duration, 
                       details={
                           'original_rows': len(data),
                           'cleaned_rows': len(clean_data),
                           'removed_rows': removed_rows
                       })
    
    # Build and train model
    try:
        model = build_model()
        log_training_event(logger, 'start', details={
            'training_samples': len(clean_data),
            'model_type': 'baseline'
        })
        
        training_start = time.time()
        model.fit(clean_data["text"], clean_data["label"])
        training_duration = time.time() - training_start
        
        log_training_event(logger, 'complete', details={
            'training_samples': len(clean_data),
            'training_duration_seconds': training_duration,
            'model_type': 'baseline'
        })
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
        
        save_start = time.time()
        joblib.dump(model, model_path)
        save_duration = time.time() - save_start
        
        log_model_operation(logger, 'save', model_path, save_duration, {
            'model_type': 'baseline',
            'training_samples': len(clean_data)
        })
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
    setup_logging(level="INFO", structured=True)
    try:
        main(args.csv, args.model)
    except SystemExit:
        raise
    except Exception as exc:
        logging.getLogger(__name__).error(f"Unexpected error: {exc}")
        raise SystemExit(f"Unexpected error: {exc}")
