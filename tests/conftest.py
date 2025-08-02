"""
Pytest configuration and fixtures for the sentiment analyzer test suite.

This module provides shared fixtures and configuration for all tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Fixture providing the test data directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Fixture providing sample sentiment data for testing."""
    return pd.DataFrame({
        'text': [
            'I love this product!',
            'This is terrible.',
            'Okay, could be better.',
            'Amazing quality and service!',
            'Not satisfied with this purchase.',
            'Neutral opinion about this item.',
        ],
        'label': ['positive', 'negative', 'neutral', 'positive', 'negative', 'neutral']
    })


@pytest.fixture
def sample_texts() -> list:
    """Fixture providing sample text data."""
    return [
        "I love this product!",
        "This is terrible.",
        "Okay, could be better.",
        "Amazing quality and service!",
        "Not satisfied with this purchase."
    ]


@pytest.fixture
def sample_labels() -> list:
    """Fixture providing sample labels."""
    return ["positive", "negative", "neutral", "positive", "negative"]


@pytest.fixture
def temp_csv_file(sample_data: pd.DataFrame) -> Generator[str, None, None]:
    """Fixture providing a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_model_file() -> Generator[str, None, None]:
    """Fixture providing a temporary file path for model saving."""
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def mock_environment_vars() -> Generator[Dict[str, str], None, None]:
    """Fixture providing mock environment variables."""
    original_env = dict(os.environ)
    test_env = {
        'MODEL_PATH': 'test_model.joblib',
        'LOG_LEVEL': 'DEBUG',
        'FLASK_ENV': 'testing'
    }
    
    # Set test environment variables
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Fixture providing test configuration."""
    return {
        'model_path': 'test_model.joblib',
        'max_features': 1000,
        'test_size': 0.2,
        'random_state': 42,
        'n_jobs': 1  # Use single thread for consistent testing
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Fixture that automatically sets up the test environment."""
    # Ensure we're in test mode
    os.environ['TESTING'] = 'true'
    
    # Set up any global test configuration
    np.random.seed(42)
    
    yield
    
    # Cleanup after tests
    if 'TESTING' in os.environ:
        del os.environ['TESTING']


# Marks for test categorization
pytest_markers = [
    "unit: marks tests as unit tests",
    "integration: marks tests as integration tests", 
    "e2e: marks tests as end-to-end tests",
    "slow: marks tests as slow (skip with -m 'not slow')",
    "performance: marks tests as performance tests",
    "gpu: marks tests requiring GPU resources",
    "network: marks tests requiring network access"
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Add markers based on test file path
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        else:
            item.add_marker(pytest.mark.unit)