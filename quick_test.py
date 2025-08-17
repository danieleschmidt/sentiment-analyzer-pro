#!/usr/bin/env python3
"""Quick test of core functionality to verify Generation 1 works."""

import sys
import os
sys.path.insert(0, '/root/repo')

def test_basic_functionality():
    """Test basic sentiment analysis functionality."""
    print("🔍 Testing Generation 1: Basic Functionality")
    
    try:
        # Test basic imports
        from src.models import SentimentModel, build_nb_model
        from src.preprocessing import preprocess_text
        from src.webapp import app
        print("✓ Core imports successful")
        
        # Test preprocessing
        text = "I love this amazing product! It's fantastic."
        processed = preprocess_text(text)
        print(f"✓ Text preprocessing: '{text}' -> '{processed}'")
        
        # Test model building with training data
        sample_texts = ["I love this", "This is bad", "Great product", "Terrible service"]
        sample_labels = ["positive", "negative", "positive", "negative"]
        
        model = build_nb_model()
        model.fit(sample_texts, sample_labels)
        print("✓ Naive Bayes model trained successfully")
        
        # Test prediction pipeline
        prediction = model.predict([processed])[0]
        print(f"✓ Prediction: {prediction}")
        
        # Test web app creation
        with app.test_client() as client:
            response = client.get('/')
            print(f"✓ Web app responds: {response.status_code}")
        
        print("\n🎉 Generation 1 COMPLETE: Basic functionality working!")
        return True
        
    except Exception as e:
        print(f"❌ Error in Generation 1: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)