"""End-to-end integration tests for the sentiment analysis pipeline."""

import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import requests
from flask import Flask

from src import cli, webapp
from src.models import build_nb_model
from src.predict import predict_sentiment
from src.train import train_model


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test the complete sentiment analysis pipeline."""

    def test_training_to_prediction_pipeline(self, sample_csv_file, temp_model_file):
        """Test complete pipeline from training to prediction."""
        # Step 1: Train model
        model = train_model(sample_csv_file, temp_model_file)
        assert model is not None
        assert Path(temp_model_file).exists()

        # Step 2: Make predictions
        test_texts = ["Great product!", "Terrible experience"]
        predictions = predict_sentiment(test_texts, temp_model_file)
        
        assert len(predictions) == 2
        assert all(pred in ["positive", "negative", "neutral"] for pred in predictions)

    def test_cli_workflow(self, sample_csv_file, tmp_path):
        """Test CLI command workflow."""
        model_path = tmp_path / "cli_model.joblib"
        
        # Train via CLI
        train_result = subprocess.run([
            "python", "-m", "src.cli", "train",
            "--csv", sample_csv_file,
            "--model", str(model_path)
        ], capture_output=True, text=True)
        
        assert train_result.returncode == 0
        assert model_path.exists()
        
        # Create test file for prediction
        test_file = tmp_path / "test_input.csv"
        test_df = pd.DataFrame({
            "text": ["Excellent product", "Poor quality"]
        })
        test_df.to_csv(test_file, index=False)
        
        # Predict via CLI
        predict_result = subprocess.run([
            "python", "-m", "src.cli", "predict",
            str(test_file),
            "--model", str(model_path)
        ], capture_output=True, text=True)
        
        assert predict_result.returncode == 0

    def test_web_api_workflow(self, sample_csv_file, temp_model_file):
        """Test web API functionality."""
        # Train model first
        train_model(sample_csv_file, temp_model_file)
        
        # Create Flask app
        app = webapp.create_app(model_path=temp_model_file)
        app.config['TESTING'] = True
        
        with app.test_client() as client:
            # Test health endpoint
            response = client.get('/')
            assert response.status_code == 200
            assert response.json['status'] == 'ok'
            
            # Test prediction endpoint
            response = client.post('/predict', 
                json={'text': 'This is amazing!'})
            assert response.status_code == 200
            assert 'prediction' in response.json
            
            # Test version endpoint
            response = client.get('/version')
            assert response.status_code == 200
            assert 'version' in response.json

    @pytest.mark.slow
    def test_large_dataset_processing(self, large_dataset, tmp_path):
        """Test processing of larger datasets."""
        large_csv = tmp_path / "large_dataset.csv"
        large_dataset.to_csv(large_csv, index=False)
        
        model_path = tmp_path / "large_model.joblib"
        
        # Train on large dataset
        model = train_model(str(large_csv), str(model_path))
        assert model is not None
        
        # Test batch prediction
        test_texts = large_dataset['text'].head(100).tolist()
        predictions = predict_sentiment(test_texts, str(model_path))
        assert len(predictions) == 100


@pytest.mark.integration
class TestCrossValidationIntegration:
    """Test cross-validation integration."""

    def test_cross_validation_cli(self, sample_csv_file):
        """Test cross-validation via CLI."""
        result = subprocess.run([
            "python", "-m", "src.cli", "crossval",
            sample_csv_file,
            "--folds", "3"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "CV accuracy" in result.stdout


@pytest.mark.integration  
class TestModelComparison:
    """Test model comparison functionality."""

    def test_model_comparison_integration(self, sample_csv_file):
        """Test complete model comparison pipeline."""
        from src.model_comparison import benchmark_models
        
        # Run model comparison (without transformer training for speed)
        results = benchmark_models(sample_csv_file, include_transformer_training=False)
        
        assert 'naive_bayes' in results
        assert 'logistic_regression' in results
        assert 'accuracy' in results['naive_bayes']


@pytest.mark.integration
class TestDataPipeline:
    """Test data processing pipeline integration."""

    def test_preprocessing_pipeline(self, tmp_path):
        """Test complete preprocessing pipeline."""
        # Create test data with various text issues
        messy_data = pd.DataFrame({
            'text': [
                'This is GREAT!!!',
                'terrible... just terrible',
                'OK product I guess',
                'AMAZING QUALITY!!!',
                'not recommended at all'
            ],
            'label': ['positive', 'negative', 'neutral', 'positive', 'negative']
        })
        
        input_file = tmp_path / "messy_data.csv"
        output_file = tmp_path / "clean_data.csv"
        messy_data.to_csv(input_file, index=False)
        
        # Test preprocessing via CLI
        result = subprocess.run([
            "python", "-m", "src.cli", "preprocess",
            str(input_file),
            "--out", str(output_file),
            "--remove-stopwords"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert output_file.exists()
        
        # Verify preprocessing worked
        cleaned_data = pd.read_csv(output_file)
        assert len(cleaned_data) == len(messy_data)
        assert 'text' in cleaned_data.columns


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_invalid_csv_handling(self, tmp_path):
        """Test handling of invalid CSV files."""
        # Create invalid CSV
        invalid_csv = tmp_path / "invalid.csv"
        with open(invalid_csv, 'w') as f:
            f.write("not,proper,csv,format\n")
            f.write("missing,required\n")
        
        # Should handle gracefully
        result = subprocess.run([
            "python", "-m", "src.cli", "train",
            "--csv", str(invalid_csv)
        ], capture_output=True, text=True)
        
        assert result.returncode != 0  # Should fail gracefully
        assert "error" in result.stderr.lower() or "error" in result.stdout.lower()

    def test_missing_model_handling(self, sample_csv_file):
        """Test prediction with missing model file."""
        result = subprocess.run([
            "python", "-m", "src.cli", "predict",
            sample_csv_file,
            "--model", "nonexistent_model.joblib"
        ], capture_output=True, text=True)
        
        assert result.returncode != 0  # Should fail gracefully