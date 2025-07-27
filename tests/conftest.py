"""Test configuration and fixtures for pytest."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_data():
    """Sample sentiment data for testing."""
    data = {
        "text": [
            "This product is amazing! I love it.",
            "Terrible quality, would not recommend.",
            "It's okay, nothing special.",
            "Excellent customer service and fast delivery.",
            "Worst purchase ever, complete waste of money.",
            "Good value for the price.",
            "Not what I expected, disappointed.",
            "Outstanding quality and craftsmanship.",
            "Average product, does the job.",
            "Absolutely fantastic, exceeded expectations!",
        ],
        "label": [
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
            "positive",
            "negative",
            "positive",
            "neutral",
            "positive",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(sample_data, tmp_path):
    """Create a temporary CSV file with sample data."""
    csv_file = tmp_path / "sample_reviews.csv"
    sample_data.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def temp_model_file(tmp_path):
    """Create a temporary model file path."""
    return str(tmp_path / "test_model.joblib")


@pytest.fixture
def test_config():
    """Test configuration dictionary."""
    return {
        "model_path": "test_model.joblib",
        "data_dir": "test_data/",
        "log_level": "WARNING",
        "max_features": 100,
        "test_size": 0.2,
        "random_state": 42,
    }


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_transformer_model():
    """Mock transformer model for testing without dependencies."""

    class MockTransformerModel:
        def __init__(self, *args, **kwargs):
            self.predictions = ["positive", "negative", "positive"]

        def predict(self, texts):
            return self.predictions[: len(texts)]

        def train(self, data):
            return {"accuracy": 0.95, "loss": 0.05}

        def save(self, path):
            pass

        def load(self, path):
            pass

    return MockTransformerModel


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")


@pytest.fixture
def large_dataset():
    """Generate a larger dataset for performance testing."""
    import random

    texts = []
    labels = []
    
    positive_templates = [
        "This is excellent {}",
        "Amazing {} quality",
        "Love this {}",
        "Outstanding {} performance",
        "Fantastic {} experience",
    ]
    
    negative_templates = [
        "Terrible {} quality",
        "Worst {} ever",
        "Disappointing {}",
        "Poor {} performance",
        "Awful {} experience",
    ]
    
    neutral_templates = [
        "Average {} quality",
        "Okay {} performance",
        "Standard {}",
        "Regular {} experience",
        "Normal {} quality",
    ]
    
    products = ["product", "service", "item", "device", "tool", "software"]
    
    for _ in range(1000):
        product = random.choice(products)
        template_choice = random.choice([0, 1, 2])
        
        if template_choice == 0:  # positive
            text = random.choice(positive_templates).format(product)
            label = "positive"
        elif template_choice == 1:  # negative
            text = random.choice(negative_templates).format(product)
            label = "negative"
        else:  # neutral
            text = random.choice(neutral_templates).format(product)
            label = "neutral"
            
        texts.append(text)
        labels.append(label)
    
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture
def performance_test_data():
    """Small dataset for performance benchmarking."""
    data = {
        "text": ["Test text"] * 100,
        "label": ["positive"] * 100,
    }
    return pd.DataFrame(data)


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_ml: Tests requiring ML libraries")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Add integration marker for tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker for tests in performance directory
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Add unit marker for tests in root tests directory
        if str(item.fspath).count("/") == str(item.fspath).count("tests") + 1:
            item.add_marker(pytest.mark.unit)