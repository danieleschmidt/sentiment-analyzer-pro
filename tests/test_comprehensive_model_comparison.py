"""Tests for comprehensive model comparison framework."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.model_comparison import ComprehensiveModelComparison, ModelResult, benchmark_models


def test_model_result_defaults():
    """Test ModelResult with default values."""
    result = ModelResult(model_name="Test", accuracy=0.85)
    
    assert result.model_name == "Test"
    assert result.accuracy == 0.85
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1_score == 0.0
    assert result.training_time == 0.0
    assert result.prediction_time == 0.0
    assert result.model_size_mb == 0.0
    assert result.additional_metrics == {}


def test_model_result_with_additional_metrics():
    """Test ModelResult with additional metrics."""
    additional = {"custom_metric": 0.9}
    result = ModelResult(
        model_name="Test",
        accuracy=0.85,
        additional_metrics=additional
    )
    
    assert result.additional_metrics == additional


def test_comprehensive_model_comparison_init():
    """Test ComprehensiveModelComparison initialization."""
    comparison = ComprehensiveModelComparison("test.csv")
    
    assert comparison.csv_path == "test.csv"
    assert comparison.data is None
    assert comparison.X_train is None
    assert comparison.X_test is None
    assert comparison.y_train is None
    assert comparison.y_test is None
    assert comparison.results == []


@patch('src.model_comparison.pd', None)
def test_load_data_missing_pandas():
    """Test load_data fails without pandas."""
    comparison = ComprehensiveModelComparison("test.csv")
    
    with pytest.raises(ImportError, match="pandas and scikit-learn are required"):
        comparison.load_data()


def test_load_data_success():
    """Test successful data loading."""
    with patch('src.model_comparison.pd') as mock_pd, \
         patch('src.model_comparison.train_test_split') as mock_split:
        
        # Mock data
        mock_df = Mock()
        mock_texts = Mock()
        mock_texts.apply.return_value = ["text1", "text2", "text3", "text4"]
        mock_df.__getitem__.side_effect = lambda key: {
            "text": mock_texts,
            "label": ["pos", "neg", "pos", "neg"]
        }[key]
        mock_pd.read_csv.return_value = mock_df
        
        mock_split.return_value = (["text1", "text2"], ["text3", "text4"], 
                                 ["pos", "neg"], ["pos", "neg"])
        
        comparison = ComprehensiveModelComparison("test.csv")
        comparison.load_data()
        
        assert comparison.data is not None
        assert comparison.X_train == ["text1", "text2"]
        assert comparison.X_test == ["text3", "text4"]
        assert comparison.y_train == ["pos", "neg"]
        assert comparison.y_test == ["pos", "neg"]


def test_calculate_detailed_metrics():
    """Test detailed metrics calculation."""
    with patch('src.model_comparison.accuracy_score', return_value=0.85), \
         patch('src.model_comparison.precision_recall_fscore_support', 
               return_value=(0.84, 0.86, 0.85, None)):
        
        comparison = ComprehensiveModelComparison("test.csv")
        metrics = comparison._calculate_detailed_metrics(
            ["pos", "neg"], ["pos", "pos"]
        )
        
        assert metrics['accuracy'] == 0.85
        assert metrics['precision'] == 0.84
        assert metrics['recall'] == 0.86
        assert metrics['f1_score'] == 0.85


def test_evaluate_baseline_models():
    """Test baseline model evaluation."""
    with patch('src.model_comparison.build_model') as mock_build, \
         patch('src.model_comparison.build_nb_model') as mock_build_nb:
        
        # Mock models
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = ["pos", "neg"]
        mock_build.return_value = mock_model
        mock_build_nb.return_value = mock_model
        
        comparison = ComprehensiveModelComparison("test.csv")
        comparison.X_train = ["text1", "text2"]
        comparison.X_test = ["text3", "text4"]
        comparison.y_train = ["pos", "neg"]
        comparison.y_test = ["pos", "neg"]
        
        with patch.object(comparison, '_calculate_detailed_metrics', 
                         return_value={'accuracy': 0.8, 'precision': 0.8, 
                                     'recall': 0.8, 'f1_score': 0.8}):
            results = comparison.evaluate_baseline_models()
        
        assert len(results) == 2  # Logistic Regression + Naive Bayes
        assert results[0].model_name == "Logistic Regression"
        assert results[1].model_name == "Naive Bayes"
        assert all(r.accuracy == 0.8 for r in results)


def test_evaluate_baseline_models_nb_failure():
    """Test baseline evaluation when Naive Bayes fails."""
    with patch('src.model_comparison.build_model') as mock_build, \
         patch('src.model_comparison.build_nb_model', side_effect=ImportError("test error")):
        
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = ["pos", "neg"]
        mock_build.return_value = mock_model
        
        comparison = ComprehensiveModelComparison("test.csv")
        comparison.X_train = ["text1", "text2"]
        comparison.X_test = ["text3", "text4"]
        comparison.y_train = ["pos", "neg"]
        comparison.y_test = ["pos", "neg"]
        
        with patch.object(comparison, '_calculate_detailed_metrics', 
                         return_value={'accuracy': 0.8, 'precision': 0.8, 
                                     'recall': 0.8, 'f1_score': 0.8}):
            results = comparison.evaluate_baseline_models()
        
        assert len(results) == 1  # Only Logistic Regression
        assert results[0].model_name == "Logistic Regression"


@patch('src.model_comparison.keras', None)
def test_evaluate_lstm_model_no_keras():
    """Test LSTM evaluation without keras."""
    comparison = ComprehensiveModelComparison("test.csv")
    result = comparison.evaluate_lstm_model()
    
    assert result is None


def test_evaluate_lstm_model_success():
    """Test successful LSTM evaluation."""
    with patch('src.model_comparison.keras') as mock_keras, \
         patch('src.model_comparison.build_lstm_model') as mock_build_lstm:
        
        # Mock keras components
        mock_tokenizer = Mock()
        mock_keras.preprocessing.text.Tokenizer.return_value = mock_tokenizer
        mock_keras.preprocessing.sequence.pad_sequences.return_value = [[1, 2, 3]]
        
        # Mock LSTM model
        mock_lstm = Mock()
        mock_lstm.fit.return_value = None
        mock_lstm.predict.return_value = np.array([[0.8], [0.3]])
        mock_build_lstm.return_value = mock_lstm
        
        comparison = ComprehensiveModelComparison("test.csv")
        comparison.X_train = ["text1", "text2"]
        comparison.X_test = ["text3", "text4"]
        comparison.y_train = ["positive", "negative"]
        comparison.y_test = ["positive", "negative"]
        
        with patch.object(comparison, '_calculate_detailed_metrics', 
                         return_value={'accuracy': 0.85, 'precision': 0.84, 
                                     'recall': 0.86, 'f1_score': 0.85}):
            result = comparison.evaluate_lstm_model()
        
        assert result is not None
        assert result.model_name == "LSTM"
        assert result.accuracy == 0.85


def test_evaluate_lstm_model_failure():
    """Test LSTM evaluation with failure."""
    with patch('src.model_comparison.keras') as mock_keras, \
         patch('src.model_comparison.build_lstm_model', side_effect=Exception("test error")):
        
        comparison = ComprehensiveModelComparison("test.csv")
        comparison.X_train = ["text1", "text2"]
        comparison.X_test = ["text3", "text4"]
        comparison.y_train = ["positive", "negative"]
        comparison.y_test = ["positive", "negative"]
        
        result = comparison.evaluate_lstm_model()
        
        assert result is None


@patch('src.model_comparison.TransformerTrainer', None)
def test_evaluate_transformer_model_no_dependencies():
    """Test transformer evaluation without dependencies."""
    comparison = ComprehensiveModelComparison("test.csv")
    result = comparison.evaluate_transformer_model(use_full_training=False)
    
    assert result is not None
    assert result.model_name == "Transformer (DistilBERT)"
    assert result.accuracy == 0.85  # Placeholder value


def test_evaluate_transformer_model_no_training():
    """Test transformer evaluation without full training."""
    with patch('src.model_comparison.TransformerTrainer', Mock()):
        comparison = ComprehensiveModelComparison("test.csv")
        result = comparison.evaluate_transformer_model(use_full_training=False)
        
        assert result is not None
        assert result.model_name == "Transformer (DistilBERT) - Pre-trained"
        assert result.accuracy == 0.82
        assert result.training_time == 0.0


def test_compare_all_models():
    """Test complete model comparison."""
    comparison = ComprehensiveModelComparison("test.csv")
    
    # Mock all evaluation methods
    baseline_results = [
        ModelResult("Logistic Regression", 0.8),
        ModelResult("Naive Bayes", 0.75)
    ]
    lstm_result = ModelResult("LSTM", 0.85)
    transformer_result = ModelResult("Transformer", 0.88)
    
    with patch.object(comparison, 'load_data'), \
         patch.object(comparison, 'evaluate_baseline_models', return_value=baseline_results), \
         patch.object(comparison, 'evaluate_lstm_model', return_value=lstm_result), \
         patch.object(comparison, 'evaluate_transformer_model', return_value=transformer_result):
        
        results = comparison.compare_all_models(include_transformer_training=True)
        
        assert len(results) == 4
        assert results[0].model_name == "Logistic Regression"
        assert results[1].model_name == "Naive Bayes"
        assert results[2].model_name == "LSTM"
        assert results[3].model_name == "Transformer"


def test_print_comparison_table_no_results():
    """Test printing comparison table with no results."""
    comparison = ComprehensiveModelComparison("test.csv")
    
    # Should not raise exception
    comparison.print_comparison_table()


def test_print_comparison_table_with_results():
    """Test printing comparison table with results."""
    comparison = ComprehensiveModelComparison("test.csv")
    comparison.results = [
        ModelResult("Model A", 0.85, 0.84, 0.86, 0.85, 1.0, 0.1),
        ModelResult("Model B", 0.80, 0.79, 0.81, 0.80, 2.0, 0.2)
    ]
    
    # Should not raise exception - just testing it runs
    comparison.print_comparison_table()


def test_get_results_dataframe_no_pandas():
    """Test get_results_dataframe without pandas."""
    with patch('src.model_comparison.pd', None):
        comparison = ComprehensiveModelComparison("test.csv")
        
        with pytest.raises(ImportError, match="pandas is required"):
            comparison.get_results_dataframe()


def test_get_results_dataframe_empty():
    """Test get_results_dataframe with no results."""
    with patch('src.model_comparison.pd') as mock_pd:
        mock_pd.DataFrame.return_value = "empty_df"
        
        comparison = ComprehensiveModelComparison("test.csv")
        result = comparison.get_results_dataframe()
        
        mock_pd.DataFrame.assert_called_once_with()
        assert result == "empty_df"


def test_get_results_dataframe_with_results():
    """Test get_results_dataframe with results."""
    with patch('src.model_comparison.pd') as mock_pd:
        mock_pd.DataFrame.return_value = "results_df"
        
        comparison = ComprehensiveModelComparison("test.csv")
        comparison.results = [
            ModelResult("Model A", 0.85, 0.84, 0.86, 0.85, 1.0, 0.1, 100.0),
            ModelResult("Model B", 0.80, 0.79, 0.81, 0.80, 2.0, 0.2, 50.0)
        ]
        
        result = comparison.get_results_dataframe()
        
        # Verify DataFrame was called with correct data
        call_args = mock_pd.DataFrame.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]['model'] == "Model A"
        assert call_args[0]['accuracy'] == 0.85
        assert call_args[1]['model'] == "Model B"
        assert call_args[1]['accuracy'] == 0.80


def test_benchmark_models_function():
    """Test benchmark_models convenience function."""
    with patch('src.model_comparison.ComprehensiveModelComparison') as mock_class:
        mock_instance = Mock()
        mock_instance.compare_all_models.return_value = [
            ModelResult("Test Model", 0.85)
        ]
        mock_class.return_value = mock_instance
        
        results = benchmark_models("test.csv", include_transformer_training=True)
        
        mock_class.assert_called_once_with("test.csv")
        mock_instance.compare_all_models.assert_called_once_with(include_transformer_training=True)
        mock_instance.print_comparison_table.assert_called_once()
        assert len(results) == 1
        assert results[0].model_name == "Test Model"


@pytest.mark.integration
def test_full_comparison_integration():
    """Integration test for model comparison without actual ML dependencies."""
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("text,label\n")
        f.write("good movie,positive\n")
        f.write("bad movie,negative\n")
        f.write("okay movie,neutral\n")
        f.write("great film,positive\n")
        temp_csv = f.name
    
    try:
        with patch('src.model_comparison.pd') as mock_pd, \
             patch('src.model_comparison.train_test_split') as mock_split:
            
            # Mock pandas DataFrame
            mock_df = Mock()
            mock_texts = Mock()
            mock_texts.apply.return_value = ["good", "bad", "okay", "great"]
            mock_df.__getitem__.side_effect = lambda key: {
                "text": mock_texts,
                "label": ["positive", "negative", "neutral", "positive"]
            }[key]
            mock_pd.read_csv.return_value = mock_df
            
            mock_split.return_value = (
                ["good", "bad"], ["okay", "great"],
                ["positive", "negative"], ["neutral", "positive"]
            )
            
            comparison = ComprehensiveModelComparison(temp_csv)
            
            # Mock all model evaluations
            with patch.object(comparison, 'evaluate_baseline_models', 
                            return_value=[ModelResult("Baseline", 0.8)]), \
                 patch.object(comparison, 'evaluate_lstm_model', 
                            return_value=ModelResult("LSTM", 0.85)), \
                 patch.object(comparison, 'evaluate_transformer_model', 
                            return_value=ModelResult("Transformer", 0.88)):
                
                results = comparison.compare_all_models()
                
                assert len(results) == 3
                assert any(r.model_name == "Baseline" for r in results)
                assert any(r.model_name == "LSTM" for r in results)
                assert any(r.model_name == "Transformer" for r in results)
    
    finally:
        os.unlink(temp_csv)


if __name__ == "__main__":
    pytest.main([__file__])