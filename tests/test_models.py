import pytest
import unittest.mock as mock

from src.models import (
    build_lstm_model, 
    build_model, 
    build_nb_model,
    build_transformer_model, 
    SentimentModel
)


def test_model_fit_predict():
    pytest.importorskip("sklearn")
    texts = ["good", "bad"]
    labels = ["positive", "negative"]
    model = build_model()
    model.fit(texts, labels)
    preds = model.predict(["good", "bad"])
    assert list(preds) == ["positive", "negative"]


def test_build_lstm_model():
    """Test basic LSTM model creation."""
    pytest.importorskip("tensorflow")
    model = build_lstm_model()
    assert isinstance(model.layers[-1].activation.__name__, str)
    
    # Test default architecture
    assert len(model.layers) == 3
    assert model.layers[0].input_dim == 10000  # default vocab_size
    assert model.layers[0].output_dim == 128   # default embed_dim
    assert model.layers[1].units == 64         # LSTM units
    assert model.layers[2].units == 1          # output units


def test_build_transformer_model():
    pytest.importorskip("transformers")
    model = build_transformer_model()
    assert model.config.num_labels == 2


def test_build_nb_model():
    """Test Naive Bayes model creation and basic functionality."""
    pytest.importorskip("sklearn")
    model = build_nb_model()
    assert isinstance(model, SentimentModel)
    assert model.pipeline is not None
    
    # Test the pipeline has expected components
    pipeline_steps = dict(model.pipeline.steps)
    assert "tfidf" in pipeline_steps
    assert "clf" in pipeline_steps
    
    # Test fit and predict functionality
    texts = ["great movie", "terrible film", "amazing story"]
    labels = ["positive", "negative", "positive"]
    model.fit(texts, labels)
    predictions = model.predict(["good movie", "bad film"])
    assert len(predictions) == 2
    assert all(pred in ["positive", "negative"] for pred in predictions)


def test_build_model_missing_sklearn():
    """Test error handling when scikit-learn is not available."""
    with mock.patch("src.models.Pipeline", None):
        with pytest.raises(ImportError, match="scikit-learn is required for build_model"):
            build_model()


def test_build_nb_model_missing_sklearn():
    """Test error handling when scikit-learn is not available for NB model."""
    with mock.patch("src.models.MultinomialNB", None):
        with pytest.raises(ImportError, match="scikit-learn is required for build_nb_model"):
            build_nb_model()


def test_build_lstm_model_missing_tensorflow():
    """Test error handling when TensorFlow is not available."""
    with mock.patch("src.models.keras", None):
        with pytest.raises(ImportError, match="TensorFlow is required for build_lstm_model"):
            build_lstm_model()


def test_build_transformer_model_missing_transformers():
    """Test error handling when transformers library is not available."""
    with mock.patch("src.models.DistilBertForSequenceClassification", None):
        with pytest.raises(ImportError, match="transformers is required for build_transformer_model"):
            build_transformer_model()


def test_build_lstm_model_custom_parameters():
    """Test LSTM model with custom configuration parameters."""
    tf = pytest.importorskip("tensorflow")
    
    # Test with custom parameters
    vocab_size = 5000
    embed_dim = 64
    sequence_length = 50
    
    model = build_lstm_model(
        vocab_size=vocab_size, 
        embed_dim=embed_dim, 
        sequence_length=sequence_length
    )
    
    # Verify model architecture
    assert len(model.layers) == 3
    embedding_layer = model.layers[0]
    lstm_layer = model.layers[1]
    dense_layer = model.layers[2]
    
    # Check embedding layer configuration
    assert embedding_layer.input_dim == vocab_size
    assert embedding_layer.output_dim == embed_dim
    assert embedding_layer.input_length == sequence_length
    
    # Check LSTM layer
    assert lstm_layer.units == 64
    
    # Check output layer
    assert dense_layer.units == 1
    assert dense_layer.activation.__name__ == "sigmoid"
    
    # Check compilation
    assert model.optimizer is not None
    assert "binary_crossentropy" in str(model.loss)


def test_build_transformer_model_custom_labels():
    """Test transformer model with custom number of labels."""
    pytest.importorskip("transformers")
    
    # Test with custom number of labels
    num_labels = 5
    model = build_transformer_model(num_labels=num_labels)
    
    assert model.config.num_labels == num_labels
    assert model.config.vocab_size == 30522  # DistilBERT default


def test_sentiment_model_dataclass():
    """Test SentimentModel dataclass functionality."""
    pytest.importorskip("sklearn")
    
    # Create a model to test the dataclass
    model = build_model()
    assert isinstance(model, SentimentModel)
    assert hasattr(model, "pipeline")
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    
    # Test that it's a proper dataclass
    import dataclasses
    assert dataclasses.is_dataclass(SentimentModel)
    
    # Test fit and predict methods
    texts = ["happy", "sad"]
    labels = ["positive", "negative"]
    
    model.fit(texts, labels)
    predictions = model.predict(["joyful"])
    assert len(predictions) == 1


def test_model_pipeline_components():
    """Test that model pipelines have correct components."""
    pytest.importorskip("sklearn")
    
    # Test logistic regression model
    lr_model = build_model()
    lr_steps = dict(lr_model.pipeline.steps)
    assert "tfidf" in lr_steps
    assert "clf" in lr_steps
    assert lr_steps["clf"].__class__.__name__ == "LogisticRegression"
    assert lr_steps["clf"].max_iter == 1000
    
    # Test Naive Bayes model
    nb_model = build_nb_model()
    nb_steps = dict(nb_model.pipeline.steps)
    assert "tfidf" in nb_steps
    assert "clf" in nb_steps
    assert nb_steps["clf"].__class__.__name__ == "MultinomialNB"
