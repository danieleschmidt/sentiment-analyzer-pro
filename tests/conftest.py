"""Pytest configuration and fixtures for sentiment analyzer tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data():
    """Sample sentiment data for testing."""
    return pd.DataFrame({
        'text': [
            'I love this product!',
            'This is terrible.',
            'Great quality and fast shipping.',
            'Not worth the money.',
            'Amazing experience!'
        ],
        'label': ['positive', 'negative', 'positive', 'negative', 'positive']
    })


@pytest.fixture
def sample_texts():
    """Sample text data for testing."""
    return [
        'I love this product!',
        'This is terrible.',
        'Great quality and fast shipping.',
        'Not worth the money.',
        'Amazing experience!'
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return ['positive', 'negative', 'positive', 'negative', 'positive']


@pytest.fixture
def temp_csv_file(tmp_path, sample_data):
    """Create a temporary CSV file with sample data."""
    csv_file = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_file, index=False)
    return str(csv_file)
