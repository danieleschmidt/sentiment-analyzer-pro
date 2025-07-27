"""Performance tests for sentiment analysis components."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.models import build_nb_model, build_logistic_model
from src.predict import predict_sentiment
from src.preprocessing import preprocess_text
from src.train import train_model


@pytest.mark.performance
@pytest.mark.slow
class TestTrainingPerformance:
    """Test training performance benchmarks."""

    def test_naive_bayes_training_speed(self, large_dataset, tmp_path):
        """Benchmark Naive Bayes training speed."""
        csv_file = tmp_path / "perf_data.csv"
        large_dataset.to_csv(csv_file, index=False)
        
        model_path = tmp_path / "perf_model.joblib"
        
        start_time = time.time()
        model = train_model(str(csv_file), str(model_path))
        training_time = time.time() - start_time
        
        assert model is not None
        assert training_time < 30  # Should complete within 30 seconds
        print(f"Training time: {training_time:.2f} seconds")

    def test_model_comparison_performance(self, sample_data, tmp_path):
        """Benchmark model comparison performance."""
        csv_file = tmp_path / "comparison_data.csv"
        sample_data.to_csv(csv_file, index=False)
        
        from src.model_comparison import benchmark_models
        
        start_time = time.time()
        results = benchmark_models(str(csv_file), include_transformer_training=False)
        comparison_time = time.time() - start_time
        
        assert results is not None
        assert comparison_time < 60  # Should complete within 1 minute
        print(f"Model comparison time: {comparison_time:.2f} seconds")


@pytest.mark.performance
class TestPredictionPerformance:
    """Test prediction performance benchmarks."""

    def test_single_prediction_speed(self, sample_csv_file, temp_model_file):
        """Benchmark single prediction speed."""
        # Train model first
        train_model(sample_csv_file, temp_model_file)
        
        test_text = "This is a test text for performance measurement."
        
        # Warm up
        predict_sentiment([test_text], temp_model_file)
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            predict_sentiment([test_text], temp_model_file)
        total_time = time.time() - start_time
        
        avg_time = total_time / 100
        assert avg_time < 0.1  # Should be under 100ms per prediction
        print(f"Average prediction time: {avg_time*1000:.2f} ms")

    def test_batch_prediction_speed(self, sample_csv_file, temp_model_file, performance_test_data):
        """Benchmark batch prediction speed."""
        # Train model first
        train_model(sample_csv_file, temp_model_file)
        
        test_texts = performance_test_data['text'].tolist()
        
        start_time = time.time()
        predictions = predict_sentiment(test_texts, temp_model_file)
        prediction_time = time.time() - start_time
        
        assert len(predictions) == len(test_texts)
        throughput = len(test_texts) / prediction_time
        assert throughput > 10  # Should handle at least 10 predictions per second
        print(f"Batch prediction throughput: {throughput:.2f} predictions/sec")

    def test_concurrent_predictions(self, sample_csv_file, temp_model_file):
        """Test concurrent prediction performance."""
        # Train model first
        train_model(sample_csv_file, temp_model_file)
        
        test_texts = ["Test text for concurrency"] * 10
        
        def predict_batch():
            return predict_sentiment(test_texts, temp_model_file)
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(predict_batch) for _ in range(4)]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        total_predictions = sum(len(result) for result in results)
        
        assert total_predictions == 40  # 4 batches × 10 predictions
        throughput = total_predictions / total_time
        print(f"Concurrent prediction throughput: {throughput:.2f} predictions/sec")


@pytest.mark.performance
class TestPreprocessingPerformance:
    """Test text preprocessing performance."""

    def test_preprocessing_speed(self, large_dataset):
        """Benchmark text preprocessing speed."""
        texts = large_dataset['text'].tolist()
        
        start_time = time.time()
        processed_texts = [preprocess_text(text) for text in texts]
        processing_time = time.time() - start_time
        
        assert len(processed_texts) == len(texts)
        throughput = len(texts) / processing_time
        assert throughput > 100  # Should process at least 100 texts per second
        print(f"Preprocessing throughput: {throughput:.2f} texts/sec")

    def test_preprocessing_batch_vs_individual(self, performance_test_data):
        """Compare batch vs individual preprocessing performance."""
        texts = performance_test_data['text'].tolist()
        
        # Individual processing
        start_time = time.time()
        individual_results = [preprocess_text(text) for text in texts]
        individual_time = time.time() - start_time
        
        # Batch processing (if available)
        start_time = time.time()
        batch_results = [preprocess_text(text) for text in texts]  # Placeholder
        batch_time = time.time() - start_time
        
        assert len(individual_results) == len(batch_results)
        print(f"Individual processing: {individual_time:.3f}s")
        print(f"Batch processing: {batch_time:.3f}s")


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage patterns."""

    def test_model_memory_usage(self, sample_csv_file, temp_model_file):
        """Monitor memory usage during model operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Train model
        model = train_model(sample_csv_file, temp_model_file)
        training_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make predictions
        test_texts = ["Test"] * 100
        predictions = predict_sentiment(test_texts, temp_model_file)
        prediction_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = prediction_memory - baseline_memory
        assert memory_increase < 500  # Should use less than 500MB additional memory
        
        print(f"Baseline memory: {baseline_memory:.1f} MB")
        print(f"Training memory: {training_memory:.1f} MB")
        print(f"Prediction memory: {prediction_memory:.1f} MB")
        print(f"Total increase: {memory_increase:.1f} MB")


@pytest.mark.performance
class TestScalabilityLimits:
    """Test system limits and scalability."""

    def test_large_text_handling(self, sample_csv_file, temp_model_file):
        """Test handling of very large text inputs."""
        # Train model first
        train_model(sample_csv_file, temp_model_file)
        
        # Create very large text
        large_text = "This is a test sentence. " * 1000  # ~25KB text
        
        start_time = time.time()
        prediction = predict_sentiment([large_text], temp_model_file)
        processing_time = time.time() - start_time
        
        assert len(prediction) == 1
        assert processing_time < 5  # Should handle large text within 5 seconds
        print(f"Large text processing time: {processing_time:.2f} seconds")

    @pytest.mark.slow
    def test_maximum_batch_size(self, sample_csv_file, temp_model_file):
        """Test maximum practical batch size."""
        # Train model first
        train_model(sample_csv_file, temp_model_file)
        
        # Test increasing batch sizes
        batch_sizes = [100, 500, 1000, 2000]
        
        for batch_size in batch_sizes:
            test_texts = [f"Test text {i}" for i in range(batch_size)]
            
            start_time = time.time()
            try:
                predictions = predict_sentiment(test_texts, temp_model_file)
                processing_time = time.time() - start_time
                
                assert len(predictions) == batch_size
                throughput = batch_size / processing_time
                print(f"Batch size {batch_size}: {throughput:.1f} predictions/sec")
                
            except Exception as e:
                print(f"Failed at batch size {batch_size}: {e}")
                break