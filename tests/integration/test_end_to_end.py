"""End-to-end integration tests."""

import tempfile
import pandas as pd
import pytest
from pathlib import Path


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete workflows from data input to prediction output."""
    
    def test_complete_training_and_prediction_workflow(self, sample_labeled_data, temp_model_dir):
        """Test complete workflow: data -> training -> prediction."""
        from src.train import train_model
        from src.predict import predict_texts
        
        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_labeled_data.to_csv(f.name, index=False)
            data_path = f.name
        
        model_path = Path(temp_model_dir) / "test_model.joblib"
        
        try:
            # Train model
            train_model(data_path, str(model_path))
            assert model_path.exists()
            
            # Make predictions
            test_texts = ["Great product!", "Terrible experience"]
            predictions = predict_texts(test_texts, str(model_path))
            
            assert len(predictions) == 2
            assert all(pred in ["positive", "negative", "neutral"] for pred in predictions)
            
        finally:
            # Cleanup
            Path(data_path).unlink(missing_ok=True)
    
    def test_cli_training_and_prediction_integration(self, sample_csv_file, temp_model_dir):
        """Test CLI training and prediction commands."""
        import subprocess
        import sys
        
        model_path = Path(temp_model_dir) / "cli_model.joblib"
        
        # Train via CLI
        train_cmd = [
            sys.executable, "-m", "src.cli", "train",
            "--csv", sample_csv_file,
            "--model", str(model_path)
        ]
        
        result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=Path.cwd())
        assert result.returncode == 0
        assert model_path.exists()
        
        # Create test prediction file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame({"text": ["Great!", "Bad!"]}).to_csv(f.name, index=False)
            pred_input = f.name
        
        try:
            # Predict via CLI
            pred_cmd = [
                sys.executable, "-m", "src.cli", "predict",
                pred_input,
                "--model", str(model_path)
            ]
            
            result = subprocess.run(pred_cmd, capture_output=True, text=True, cwd=Path.cwd())
            assert result.returncode == 0
            
        finally:
            Path(pred_input).unlink(missing_ok=True)
    
    def test_web_api_integration(self, flask_client, mock_model):
        """Test web API endpoints integration."""
        # Test health endpoint
        response = flask_client.get("/")
        assert response.status_code == 200
        assert response.json == {"status": "ok"}
        
        # Test prediction endpoint
        with pytest.mock.patch("src.webapp.load_model", return_value=mock_model):
            response = flask_client.post("/predict", json={"text": "Great product!"})
            assert response.status_code == 200
            assert "prediction" in response.json
        
        # Test metrics endpoint
        response = flask_client.get("/metrics")
        assert response.status_code == 200
        assert "requests" in response.json
    
    def test_model_comparison_integration(self, sample_csv_file):
        """Test model comparison framework integration."""
        from src.model_comparison import benchmark_models
        
        # Run quick comparison (no transformer training)
        results = benchmark_models(sample_csv_file, include_transformer_training=False)
        
        assert isinstance(results, dict)
        assert "logistic_regression" in results
        assert "naive_bayes" in results
        
        # Check required metrics
        for model_name, metrics in results.items():
            assert "accuracy" in metrics
            assert "training_time" in metrics
            assert "prediction_time" in metrics


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Test data processing pipeline integration."""
    
    def test_preprocessing_pipeline_integration(self, sample_text_data):
        """Test complete text preprocessing pipeline."""
        from src.preprocessing import preprocess_texts, create_vectorizer
        
        # Preprocess texts
        processed_texts = preprocess_texts(sample_text_data)
        assert len(processed_texts) == len(sample_text_data)
        assert all(isinstance(text, str) for text in processed_texts)
        
        # Vectorize texts
        vectorizer = create_vectorizer(max_features=100)
        vectors = vectorizer.fit_transform(processed_texts)
        
        assert vectors.shape[0] == len(processed_texts)
        assert vectors.shape[1] <= 100
    
    def test_aspect_sentiment_integration(self, sample_text_data):
        """Test aspect-based sentiment analysis integration."""
        try:
            from src.aspect_sentiment import analyze_aspects
            
            results = analyze_aspects(sample_text_data)
            assert isinstance(results, list)
            assert len(results) == len(sample_text_data)
            
        except ImportError:
            pytest.skip("Aspect sentiment analysis not available")


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance-critical integrations."""
    
    def test_large_dataset_processing(self, large_dataset, temp_model_dir, benchmark_runner):
        """Test processing large datasets within acceptable time limits."""
        from src.train import train_model
        
        # Save large dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_dataset.to_csv(f.name, index=False)
            data_path = f.name
        
        model_path = Path(temp_model_dir) / "large_model.joblib"
        
        try:
            # Training should complete within reasonable time
            benchmark_runner.assert_performance(
                train_model, 
                max_time=30.0,  # 30 seconds max
                data_path=data_path,
                model_path=str(model_path)
            )
            
            assert model_path.exists()
            
        finally:
            Path(data_path).unlink(missing_ok=True)
    
    def test_batch_prediction_performance(self, large_dataset, benchmark_runner):
        """Test batch prediction performance."""
        from src.predict import predict_texts
        from src.models import build_logistic_regression_model
        
        # Quick model training
        model = build_logistic_regression_model()
        model.fit(large_dataset["text"][:100], large_dataset["label"][:100])
        
        # Batch prediction should be fast
        test_texts = large_dataset["text"][:500].tolist()
        
        predictions, time_taken = benchmark_runner.time_function(
            predict_texts, test_texts, model
        )
        
        assert len(predictions) == 500
        assert time_taken < 5.0  # Should complete in under 5 seconds


@pytest.mark.integration
class TestSecurityIntegration:
    """Test security-related integrations."""
    
    def test_input_validation_integration(self, flask_client):
        """Test input validation across API endpoints."""
        # Test malformed JSON
        response = flask_client.post("/predict", data="invalid json")
        assert response.status_code == 400
        
        # Test missing required fields
        response = flask_client.post("/predict", json={})
        assert response.status_code == 400
        
        # Test extremely long input
        long_text = "x" * 10000
        response = flask_client.post("/predict", json={"text": long_text})
        # Should handle gracefully (either process or reject cleanly)
        assert response.status_code in [200, 400, 413]
    
    def test_file_handling_security(self, temp_model_dir):
        """Test secure file handling in model operations."""
        from src.train import train_model
        
        # Test with non-existent file
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            train_model("non_existent.csv", str(Path(temp_model_dir) / "model.joblib"))
        
        # Test with invalid file permissions (if applicable)
        # This would be platform-specific testing


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling across system components."""
    
    def test_graceful_degradation_without_ml_dependencies(self, disable_ml_dependencies):
        """Test system behavior when optional ML dependencies are missing."""
        from src.models import build_nb_model, build_logistic_regression_model
        
        # Basic models should still work
        nb_model = build_nb_model()
        lr_model = build_logistic_regression_model()
        
        assert nb_model is not None
        assert lr_model is not None
        
        # Advanced models should handle missing dependencies gracefully
        try:
            from src.transformer_trainer import TransformerTrainer
            pytest.fail("Should have failed due to missing dependencies")
        except ImportError:
            pass  # Expected behavior
    
    def test_corrupted_data_handling(self, temp_model_dir):
        """Test handling of corrupted or malformed data files."""
        from src.train import train_model
        
        # Create corrupted CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,data\nwith,wrong,columns,too,many")
            corrupted_path = f.name
        
        model_path = Path(temp_model_dir) / "model.joblib"
        
        try:
            with pytest.raises((ValueError, KeyError, pd.errors.ParserError)):
                train_model(corrupted_path, str(model_path))
        finally:
            Path(corrupted_path).unlink(missing_ok=True)