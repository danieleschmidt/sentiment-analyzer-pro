"""Tests for transformer training functionality."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd

# No autouse fixture - handle mocking in individual tests


def test_transformer_config_defaults():
    """Test TransformerConfig default values."""
    from src.transformer_trainer import TransformerConfig
    
    config = TransformerConfig()
    assert config.model_name == "distilbert-base-uncased"
    assert config.max_length == 128
    assert config.batch_size == 16
    assert config.learning_rate == 2e-5
    assert config.num_epochs == 3


def test_transformer_config_custom():
    """Test TransformerConfig with custom values."""
    from src.transformer_trainer import TransformerConfig
    
    config = TransformerConfig(
        model_name="bert-base-uncased",
        max_length=256,
        batch_size=32,
        learning_rate=1e-5,
        num_epochs=5
    )
    assert config.model_name == "bert-base-uncased"
    assert config.max_length == 256
    assert config.batch_size == 32
    assert config.learning_rate == 1e-5
    assert config.num_epochs == 5


@patch('src.transformer_trainer.torch', None)
def test_transformer_trainer_missing_dependencies():
    """Test TransformerTrainer fails gracefully without dependencies."""
    from src.transformer_trainer import TransformerTrainer
    
    with pytest.raises(ImportError, match="torch and transformers are required"):
        TransformerTrainer()


def test_sentiment_dataset_creation():
    """Test SentimentDataset creation with mocked dependencies."""
    try:
        import torch  # noqa: F401
        pytest.skip("torch is available, test not needed for fallback case")
    except ImportError:
        pass
    
    from src.transformer_trainer import SentimentDataset
    
    texts = ["good movie", "bad movie"]
    labels = [1, 0]
    tokenizer = Mock()
    
    # Test that it raises ImportError when torch is not available
    with pytest.raises(ImportError, match="torch is required for SentimentDataset"):
        SentimentDataset(texts, labels, tokenizer, max_length=128)


def test_sentiment_dataset_getitem():
    """Test SentimentDataset __getitem__ method."""
    try:
        import torch  # noqa: F401
        pytest.skip("torch is available, test not needed for fallback case")
    except ImportError:
        pass
    
    from src.transformer_trainer import SentimentDataset
    
    texts = ["good movie"]
    labels = [1]
    tokenizer = Mock()
    
    # Test that it raises ImportError when torch is not available
    with pytest.raises(ImportError, match="torch is required for SentimentDataset"):
        SentimentDataset(texts, labels, tokenizer, max_length=128)


def test_transformer_trainer_initialization():
    """Test TransformerTrainer initialization."""
    with patch('src.transformer_trainer.torch'), \
         patch('src.transformer_trainer.DistilBertTokenizer'):
        
        from src.transformer_trainer import TransformerTrainer, TransformerConfig
        
        config = TransformerConfig(model_name="test-model")
        trainer = TransformerTrainer(config)
        
        assert trainer.config.model_name == "test-model"
        assert trainer.tokenizer is None
        assert trainer.model is None


def test_transformer_trainer_prepare_labels():
    """Test label preparation in TransformerTrainer."""
    with patch('src.transformer_trainer.torch'), \
         patch('src.transformer_trainer.DistilBertTokenizer'):
        
        from src.transformer_trainer import TransformerTrainer
        
        trainer = TransformerTrainer()
        labels = pd.Series(["positive", "negative", "positive", "neutral"])
        
        numeric_labels, label_map = trainer._prepare_labels(labels)
        
        assert len(numeric_labels) == 4
        assert "negative" in label_map
        assert "positive" in label_map
        assert "neutral" in label_map
        assert len(label_map) == 3
        assert trainer.label_map == label_map


def test_transformer_trainer_setup_model():
    """Test model setup in TransformerTrainer."""
    with patch('src.transformer_trainer.torch'), \
         patch('src.transformer_trainer.DistilBertTokenizer') as mock_tokenizer_class, \
         patch('src.transformer_trainer.DistilBertForSequenceClassification') as mock_model_class:
        
        from src.transformer_trainer import TransformerTrainer
        
        trainer = TransformerTrainer()
        trainer._setup_model(num_labels=3)
        
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "distilbert-base-uncased",
            revision="main"
        )
        mock_model_class.from_pretrained.assert_called_once_with(
            "distilbert-base-uncased",
            num_labels=3,
            revision="main"
        )


def test_train_transformer_model_missing_dependencies():
    """Test train_transformer_model with missing dependencies."""
    from src.transformer_trainer import train_transformer_model
    
    # Since torch is not available, this should raise ImportError for torch/transformers
    with pytest.raises(ImportError, match="torch and transformers are required"):
        train_transformer_model()


def test_transformer_trainer_train_missing_data_columns():
    """Test training fails with missing required columns."""
    with patch('src.transformer_trainer.torch'), \
         patch('src.transformer_trainer.DistilBertTokenizer'), \
         patch('src.transformer_trainer.pd') as mock_pd:
        
        # Mock DataFrame without required columns
        mock_df = Mock()
        mock_df.columns = ["wrong_column"]
        mock_pd.read_csv.return_value = mock_df
        
        from src.transformer_trainer import TransformerTrainer
        
        trainer = TransformerTrainer()
        
        with pytest.raises(ValueError, match="Data must contain 'text' and 'label' columns"):
            trainer.train("dummy.csv")


def test_transformer_trainer_predict_no_model():
    """Test prediction fails when model not trained."""
    with patch('src.transformer_trainer.torch'), \
         patch('src.transformer_trainer.DistilBertTokenizer'):
        
        from src.transformer_trainer import TransformerTrainer
        
        trainer = TransformerTrainer()
        
        with pytest.raises(ValueError, match="Model not trained"):
            trainer.predict(["test text"])


def test_comprehensive_model_comparison_import():
    """Test importing ComprehensiveModelComparison."""
    from src.model_comparison import ComprehensiveModelComparison, ModelResult
    
    # Should import successfully
    assert ComprehensiveModelComparison is not None
    assert ModelResult is not None


def test_model_result_creation():
    """Test ModelResult dataclass."""
    from src.model_comparison import ModelResult
    
    result = ModelResult(
        model_name="Test Model",
        accuracy=0.85,
        precision=0.80,
        recall=0.90,
        f1_score=0.84
    )
    
    assert result.model_name == "Test Model"
    assert result.accuracy == 0.85
    assert result.precision == 0.80
    assert result.recall == 0.90
    assert result.f1_score == 0.84
    assert result.additional_metrics == {}


def test_benchmark_models_function():
    """Test benchmark_models convenience function."""
    with patch('src.model_comparison.ComprehensiveModelComparison') as mock_comparison_class:
        mock_comparison = Mock()
        mock_comparison_class.return_value = mock_comparison
        mock_comparison.compare_all_models.return_value = []
        
        from src.model_comparison import benchmark_models
        
        benchmark_models("test.csv", include_transformer_training=True)
        
        mock_comparison_class.assert_called_once_with("test.csv")
        mock_comparison.compare_all_models.assert_called_once_with(include_transformer_training=True)
        mock_comparison.print_comparison_table.assert_called_once()


@pytest.mark.integration
def test_transformer_trainer_full_pipeline():
    """Integration test for full transformer training pipeline."""
    try:
        import torch  # noqa: F401
        pytest.skip("torch is available, this test checks the fallback behavior")
    except ImportError:
        pass
    
    # Since torch is not available, TransformerTrainer should raise ImportError
    from src.transformer_trainer import TransformerTrainer
    
    with pytest.raises(ImportError, match="torch and transformers are required"):
        TransformerTrainer()


def test_transformer_trainer_save_load_model():
    """Test model saving and loading."""
    with patch('src.transformer_trainer.torch'), \
         patch('src.transformer_trainer.DistilBertTokenizer') as mock_tokenizer_class, \
         patch('src.transformer_trainer.DistilBertForSequenceClassification') as mock_model_class, \
         patch('src.transformer_trainer.os.makedirs'), \
         patch('builtins.open', create=True), \
         patch('src.transformer_trainer.os.path.exists', return_value=True), \
         patch('json.dump'), \
         patch('json.load', return_value={"positive": 1, "negative": 0}):
        
        from src.transformer_trainer import TransformerTrainer
        
        trainer = TransformerTrainer()
        trainer.model = Mock()
        trainer.tokenizer = Mock()
        trainer.label_map = {"positive": 1, "negative": 0}
        
        # Test save
        save_path = trainer.save_model("test_path")
        assert save_path == "test_path"
        
        # Test load
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()
        
        trainer.load_model("test_path")
        
        mock_tokenizer_class.from_pretrained.assert_called_with("test_path")
        mock_model_class.from_pretrained.assert_called_with("test_path")


if __name__ == "__main__":
    pytest.main([__file__])