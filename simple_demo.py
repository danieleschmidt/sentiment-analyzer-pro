#!/usr/bin/env python3
"""Simple demonstration of Generation 1 functionality."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


def create_simple_model():
    """Create a simple sentiment model without complex caching."""
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english")),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    return pipeline


def demo_generation_1():
    """Demonstrate Generation 1: MAKE IT WORK functionality."""
    print("🚀 GENERATION 1: MAKE IT WORK - Basic Sentiment Analysis")
    print("=" * 60)

    # Load sample data
    try:
        data = pd.read_csv("data/sample_reviews.csv")
        print(f"✅ Loaded {len(data)} samples from data/sample_reviews.csv")
    except FileNotFoundError:
        # Create minimal sample data
        data = pd.DataFrame(
            {
                "text": [
                    "I love this product",
                    "This is terrible",
                    "Amazing quality",
                    "Worst purchase ever",
                    "Great value for money",
                ],
                "label": ["positive", "negative", "positive", "negative", "positive"],
            }
        )
        print(f"✅ Created sample dataset with {len(data)} examples")

    # Create and train model
    model = create_simple_model()
    print("✅ Created simple sentiment analysis pipeline")

    # Train the model
    X = data["text"]
    y = data["label"]
    model.fit(X, y)
    print("✅ Model trained successfully")

    # Make predictions
    test_texts = [
        "This product is amazing!",
        "I hate this so much",
        "It's okay, nothing special",
        "Absolutely wonderful experience",
    ]

    predictions = model.predict(test_texts)
    probabilities = model.predict_proba(test_texts)

    print("\n📊 Prediction Results:")
    print("-" * 40)
    for text, pred, prob in zip(test_texts, predictions, probabilities):
        confidence = max(prob)
        print(f"Text: {text[:30]}...")
        print(f"Prediction: {pred} (confidence: {confidence:.2f})")
        print()

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/simple_sentiment_model.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

    # Test loading
    loaded_model = joblib.load(model_path)
    test_prediction = loaded_model.predict(["This is a test"])
    print(f"✅ Model loaded and tested: {test_prediction[0]}")

    print("\n🎉 Generation 1 Complete: Basic functionality working!")
    return model


if __name__ == "__main__":
    demo_generation_1()
