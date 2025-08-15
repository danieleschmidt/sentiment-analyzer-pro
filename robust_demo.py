#!/usr/bin/env python3
"""Robust demonstration of Generation 2 functionality with error handling, logging, and security."""

import sys
import os
import logging
import hashlib
import time
from pathlib import Path
import json
from typing import Dict, List, Optional, Any
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/robust_demo.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class SecurityValidator:
    """Security validation for input data."""

    @staticmethod
    def validate_text_input(text: str) -> bool:
        """Validate and sanitize text input."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        if len(text) > 10000:  # Prevent memory exhaustion
            raise ValueError("Text too long (max 10,000 characters)")

        # Check for potential malicious patterns
        suspicious_patterns = ["<script", "javascript:", "data:text/html"]
        text_lower = text.lower()
        for pattern in suspicious_patterns:
            if pattern in text_lower:
                logger.warning(f"Suspicious pattern detected: {pattern}")
                return False

        return True

    @staticmethod
    def hash_input(text: str) -> str:
        """Create hash of input for audit trail."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]


class RobustSentimentAnalyzer:
    """Robust sentiment analyzer with comprehensive error handling."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.metadata = {}
        self.performance_metrics = {}
        self.security_validator = SecurityValidator()

        # Ensure directories exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)

        logger.info("RobustSentimentAnalyzer initialized")

    def create_model(self) -> Pipeline:
        """Create robust sentiment analysis pipeline."""
        try:
            pipeline = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=5000,
                            stop_words="english",
                            min_df=2,
                            max_df=0.95,
                            ngram_range=(1, 2),
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            random_state=42, max_iter=1000, class_weight="balanced"
                        ),
                    ),
                ]
            )
            logger.info("Model pipeline created successfully")
            return pipeline
        except Exception as e:
            logger.error(f"Error creating model pipeline: {e}")
            raise

    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean training data."""
        logger.info(f"Validating dataset with {len(data)} samples")

        # Check required columns
        required_columns = ["text", "label"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Remove null values
        initial_size = len(data)
        data = data.dropna(subset=required_columns)
        if len(data) < initial_size:
            logger.warning(f"Removed {initial_size - len(data)} rows with null values")

        # Validate text inputs
        valid_rows = []
        for idx, row in data.iterrows():
            try:
                if self.security_validator.validate_text_input(row["text"]):
                    valid_rows.append(idx)
                else:
                    logger.warning(
                        f"Skipping row {idx} due to security validation failure"
                    )
            except Exception as e:
                logger.warning(f"Skipping row {idx} due to validation error: {e}")

        data = data.loc[valid_rows]
        logger.info(f"Data validation complete: {len(data)} valid samples")

        return data

    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train model with comprehensive error handling and monitoring."""
        start_time = time.time()
        logger.info("Starting model training")

        try:
            # Validate data
            data = self.validate_data(data)

            if len(data) < 10:
                raise ValueError(
                    f"Insufficient training data: {len(data)} samples (minimum 10 required)"
                )

            # Create model
            self.model = self.create_model()

            # Split data for validation
            X = data["text"]
            y = data["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train model
            logger.info(
                f"Training on {len(X_train)} samples, validating on {len(X_test)} samples"
            )
            self.model.fit(X_train, y_train)

            # Evaluate model
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)

            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)

            self.performance_metrics = {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "training_time": time.time() - start_time,
                "model_features": (
                    self.model.named_steps["tfidf"].vocabulary_.__len__()
                    if hasattr(self.model.named_steps["tfidf"], "vocabulary_")
                    else 0
                ),
            }

            logger.info(
                f"Training complete - Train Accuracy: {train_accuracy:.3f}, Test Accuracy: {test_accuracy:.3f}"
            )

            # Save metadata
            self.metadata = {
                "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "data_size": len(data),
                "unique_labels": data["label"].unique().tolist(),
                "performance_metrics": self.performance_metrics,
            }

            return self.performance_metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions with security validation and error handling."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        results = []

        for i, text in enumerate(texts):
            try:
                # Security validation
                if not self.security_validator.validate_text_input(text):
                    results.append(
                        {
                            "index": i,
                            "text": text[:50] + "..." if len(text) > 50 else text,
                            "prediction": "SECURITY_VIOLATION",
                            "confidence": 0.0,
                            "error": "Security validation failed",
                        }
                    )
                    continue

                # Make prediction
                prediction = self.model.predict([text])[0]
                probabilities = self.model.predict_proba([text])[0]
                confidence = max(probabilities)

                # Log for audit trail
                text_hash = self.security_validator.hash_input(text)
                logger.info(
                    f"Prediction made for text_hash:{text_hash} -> {prediction} (confidence: {confidence:.3f})"
                )

                results.append(
                    {
                        "index": i,
                        "text": text[:50] + "..." if len(text) > 50 else text,
                        "prediction": prediction,
                        "confidence": confidence,
                        "text_hash": text_hash,
                    }
                )

            except Exception as e:
                logger.error(f"Prediction failed for text {i}: {e}")
                results.append(
                    {
                        "index": i,
                        "text": text[:50] + "..." if len(text) > 50 else text,
                        "prediction": "ERROR",
                        "confidence": 0.0,
                        "error": str(e),
                    }
                )

        return results

    def save_model(self, path: Optional[str] = None) -> str:
        """Save model with metadata and security validation."""
        if self.model is None:
            raise ValueError("No model to save")

        if path is None:
            path = self.model_path or "models/robust_sentiment_model.joblib"

        try:
            # Save model
            joblib.dump(self.model, path)

            # Save metadata
            metadata_path = path.replace(".joblib", "_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)

            logger.info(f"Model saved to {path}")
            logger.info(f"Metadata saved to {metadata_path}")

            return path

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """Load model with validation."""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            self.model = joblib.load(path)
            self.model_path = path

            # Load metadata if available
            metadata_path = path.replace(".joblib", "_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)

            logger.info(f"Model loaded from {path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


def demo_generation_2():
    """Demonstrate Generation 2: MAKE IT ROBUST functionality."""
    print("🛡️ GENERATION 2: MAKE IT ROBUST - Comprehensive Error Handling & Security")
    print("=" * 80)

    analyzer = RobustSentimentAnalyzer()

    try:
        # Load or create sample data
        try:
            data = pd.read_csv("data/sample_reviews.csv")
            print(f"✅ Loaded {len(data)} samples from data/sample_reviews.csv")
        except FileNotFoundError:
            # Create robust sample data
            data = pd.DataFrame(
                {
                    "text": [
                        "I love this product",
                        "This is terrible",
                        "Amazing quality and great service",
                        "Worst purchase ever, total waste of money",
                        "Great value for money",
                        "Outstanding product with excellent customer support",
                        "Poor quality, very disappointed",
                        "Highly recommended to everyone",
                        "Not worth the price",
                        "Fantastic experience overall",
                    ],
                    "label": [
                        "positive",
                        "negative",
                        "positive",
                        "negative",
                        "positive",
                        "positive",
                        "negative",
                        "positive",
                        "negative",
                        "positive",
                    ],
                }
            )
            print(f"✅ Created robust sample dataset with {len(data)} examples")

        # Train model
        print("\n📚 Training robust model...")
        metrics = analyzer.train(data)
        print(f"✅ Model trained - Test Accuracy: {metrics['test_accuracy']:.3f}")
        print(f"✅ Training time: {metrics['training_time']:.2f} seconds")

        # Save model
        model_path = analyzer.save_model()
        print(f"✅ Model saved with metadata")

        # Test predictions with various inputs including edge cases
        test_texts = [
            "This product is amazing!",
            "I hate this so much",
            "",  # Empty string test
            "A" * 50,  # Long string test
            "It's okay, nothing special",
            "Absolutely wonderful experience",
            "<script>alert('test')</script>",  # Security test
            "Normal text with normal sentiment",
        ]

        print("\n🔍 Making robust predictions...")
        results = analyzer.predict(test_texts)

        print("\n📊 Prediction Results:")
        print("-" * 60)
        for result in results:
            print(f"Text: {result['text']}")
            print(
                f"Prediction: {result['prediction']} (confidence: {result.get('confidence', 0):.3f})"
            )
            if "error" in result:
                print(f"⚠️  Error: {result['error']}")
            if "text_hash" in result:
                print(f"Hash: {result['text_hash']}")
            print()

        # Test model loading
        print("🔄 Testing model persistence...")
        new_analyzer = RobustSentimentAnalyzer()
        new_analyzer.load_model(model_path)
        test_prediction = new_analyzer.predict(["This is a persistence test"])
        print(f"✅ Model loaded and tested: {test_prediction[0]['prediction']}")

        # Display performance metrics
        print("\n📈 Performance Metrics:")
        print("-" * 40)
        for key, value in metrics.items():
            print(f"{key}: {value}")

        print(
            "\n🎉 Generation 2 Complete: Robust error handling, logging, and security implemented!"
        )

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    demo_generation_2()
