"""Performance and benchmark tests for the sentiment analyzer."""

import pytest
import time
import pandas as pd
import numpy as np
from typing import List

from src.preprocessing import preprocess_text, prepare_data_for_training
from src.models import build_nb_model
from src.train import train_model
from src.predict import predict_sentiment


@pytest.mark.performance
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_preprocessing_performance(self):
        """Benchmark text preprocessing performance."""
        # Generate large dataset
        texts = ["This is a sample text for preprocessing"] * 1000
        
        start_time = time.time()
        processed_texts = [preprocess_text(text) for text in texts]
        end_time = time.time()
        
        processing_time = end_time - start_time
        texts_per_second = len(texts) / processing_time
        
        # Assert reasonable performance (should process > 100 texts/second)
        assert texts_per_second > 100, f"Preprocessing too slow: {texts_per_second:.2f} texts/sec"
        assert len(processed_texts) == len(texts)
    
    def test_model_training_performance(self, sample_data: pd.DataFrame):
        """Benchmark model training performance."""
        # Create larger dataset
        large_data = pd.concat([sample_data] * 100, ignore_index=True)
        
        X_train, _, y_train, _ = prepare_data_for_training(
            large_data['text'], large_data['label'], test_size=0.2
        )
        
        model = build_nb_model()
        
        start_time = time.time()
        trained_model = train_model(model, X_train, y_train)
        end_time = time.time()
        
        training_time = end_time - start_time
        samples_per_second = len(X_train) / training_time
        
        # Assert reasonable training performance
        assert training_time < 30, f"Training too slow: {training_time:.2f} seconds"
        assert samples_per_second > 10, f"Training throughput too low: {samples_per_second:.2f} samples/sec"
    
    def test_prediction_performance(self, sample_data: pd.DataFrame):
        """Benchmark prediction performance."""
        # Train a model
        X_train, _, y_train, _ = prepare_data_for_training(
            sample_data['text'], sample_data['label'], test_size=0.2
        )
        
        model = build_nb_model()
        trained_model = train_model(model, X_train, y_train)
        
        # Generate test data
        test_texts = ["Sample text for prediction"] * 1000
        
        start_time = time.time()
        predictions = predict_sentiment(trained_model, test_texts)
        end_time = time.time()
        
        prediction_time = end_time - start_time
        predictions_per_second = len(test_texts) / prediction_time
        
        # Assert reasonable prediction performance
        assert predictions_per_second > 1000, f"Prediction too slow: {predictions_per_second:.2f} predictions/sec"
        assert len(predictions) == len(test_texts)
    
    @pytest.mark.parametrize("data_size", [100, 500, 1000])
    def test_scalability(self, sample_data: pd.DataFrame, data_size: int):
        """Test performance scalability with different data sizes."""
        # Create dataset of specified size
        scaled_data = pd.concat([sample_data] * (data_size // len(sample_data) + 1), ignore_index=True)
        scaled_data = scaled_data.iloc[:data_size]
        
        start_time = time.time()
        
        X_train, X_test, y_train, y_test = prepare_data_for_training(
            scaled_data['text'], scaled_data['label'], test_size=0.2
        )
        
        model = build_nb_model()
        trained_model = train_model(model, X_train, y_train)
        predictions = predict_sentiment(trained_model, X_test)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Log performance metrics
        print(f"\nDataset size: {data_size}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Samples per second: {data_size / total_time:.2f}")
        
        # Basic assertions
        assert total_time < 60, f"Processing {data_size} samples took too long: {total_time:.2f}s"
        assert len(predictions) > 0


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage patterns."""
    
    def test_memory_efficient_processing(self, sample_data: pd.DataFrame):
        """Test that processing doesn't use excessive memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create larger dataset
        large_data = pd.concat([sample_data] * 200, ignore_index=True)
        
        # Process data
        X_train, X_test, y_train, y_test = prepare_data_for_training(
            large_data['text'], large_data['label'], test_size=0.2
        )
        
        model = build_nb_model()
        trained_model = train_model(model, X_train, y_train)
        predictions = predict_sentiment(trained_model, X_test)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Assert reasonable memory usage (less than 500MB increase)
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.2f}MB increase"
        assert len(predictions) > 0