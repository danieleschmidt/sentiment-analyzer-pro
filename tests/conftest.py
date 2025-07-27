"""Shared pytest fixtures for the test suite."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pandas as pd
import pytest
from unittest.mock import Mock, patch

# Test data fixtures
@pytest.fixture
def sample_text_data() -> list[str]:
    """Sample text data for testing."""
    return [
        "I love this product! It's amazing.",
        "This is terrible. I hate it.",
        "It's okay, nothing special.",
        "Absolutely fantastic! Highly recommended.",
        "Worst purchase ever. Complete waste of money.",
    ]


@pytest.fixture
def sample_labeled_data() -> pd.DataFrame:
    """Sample labeled dataset for testing."""
    return pd.DataFrame({
        "text": [
            "I love this product! It's amazing.",
            "This is terrible. I hate it.",
            "It's okay, nothing special.",
            "Absolutely fantastic! Highly recommended.",
            "Worst purchase ever. Complete waste of money.",
            "Good quality and fast delivery.",
            "Not worth the price.",
            "Excellent customer service.",
        ],
        "label": ["positive", "negative", "neutral", "positive", "negative", "positive", "negative", "positive"]
    })


@pytest.fixture
def sample_csv_file(sample_labeled_data: pd.DataFrame) -> Generator[str, None, None]:
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_labeled_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_model_dir() -> Generator[str, None, None]:
    """Create a temporary directory for model artifacts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.predict.return_value = ["positive", "negative"]
    model.predict_proba.return_value = [[0.8, 0.2], [0.3, 0.7]]
    return model


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Mock configuration for testing."""
    return {
        "model_type": "logistic_regression",
        "max_features": 1000,
        "ngram_range": (1, 2),
        "random_seed": 42,
        "test_size": 0.2,
    }


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    env_vars = {
        "MODEL_PATH": "test_model.joblib",
        "LOG_LEVEL": "DEBUG",
        "FLASK_ENV": "testing",
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def large_dataset() -> pd.DataFrame:
    """Generate a larger dataset for performance testing."""
    import random
    
    positive_texts = [
        "Great product!", "Love it!", "Excellent quality!",
        "Highly recommend!", "Amazing experience!", "Perfect!",
        "Outstanding service!", "Fantastic results!", "Brilliant!",
        "Superb quality!", "Wonderful!", "Incredible value!"
    ]
    
    negative_texts = [
        "Terrible quality!", "Hate it!", "Worst ever!",
        "Complete waste!", "Awful experience!", "Horrible!",
        "Disappointing results!", "Poor service!", "Useless!",
        "Bad quality!", "Regret buying!", "Not recommended!"
    ]
    
    neutral_texts = [
        "It's okay.", "Average product.", "Nothing special.",
        "Standard quality.", "As expected.", "Normal service.",
        "Acceptable.", "Fair enough.", "Typical.", "Regular quality."
    ]
    
    texts = []
    labels = []
    
    for _ in range(1000):
        category = random.choice(["positive", "negative", "neutral"])
        if category == "positive":
            text = random.choice(positive_texts)
        elif category == "negative":
            text = random.choice(negative_texts)
        else:
            text = random.choice(neutral_texts)
        
        texts.append(text)
        labels.append(category)
    
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture
def flask_client():
    """Flask test client."""
    from src.webapp import create_app
    
    app = create_app(testing=True)
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        with app.app_context():
            yield client


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    import logging
    logging.basicConfig(level=logging.DEBUG)
    yield
    # Cleanup logging handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


@pytest.fixture
def disable_ml_dependencies():
    """Mock missing ML dependencies for testing graceful degradation."""
    with patch('src.models.torch', None), \
         patch('src.models.transformers', None), \
         patch('src.models.tensorflow', None):
        yield


# Marks for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "ml: Machine learning tests")
    config.addinivalue_line("markers", "api: API tests")
    config.addinivalue_line("markers", "cli: CLI tests")


# Pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "test_webapp" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        elif "test_cli" in str(item.fspath):
            item.add_marker(pytest.mark.cli)
        elif any(name in str(item.fspath) for name in ["test_models", "test_train", "test_transformer"]):
            item.add_marker(pytest.mark.ml)
        else:
            item.add_marker(pytest.mark.unit)


# Performance testing utilities
@pytest.fixture
def benchmark_runner():
    """Utility for running benchmarks in tests."""
    import time
    
    class BenchmarkRunner:
        def __init__(self):
            self.results = {}
        
        def time_function(self, func, *args, **kwargs):
            """Time a function execution."""
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            return result, execution_time
        
        def assert_performance(self, func, max_time: float, *args, **kwargs):
            """Assert that a function executes within a time limit."""
            _, execution_time = self.time_function(func, *args, **kwargs)
            assert execution_time <= max_time, f"Function took {execution_time:.3f}s, expected <= {max_time}s"
    
    return BenchmarkRunner()