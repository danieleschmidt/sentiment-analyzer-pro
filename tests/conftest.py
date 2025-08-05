"""Test configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data():
    """Sample sentiment data for testing."""
    return pd.DataFrame({
        'text': [
            'I love this product',
            'This is terrible',
            'Amazing quality',
            'Worst experience ever',
            'Great value for money'
        ],
        'label': ['positive', 'negative', 'positive', 'negative', 'positive']
    })


@pytest.fixture
def sample_texts():
    """Sample text data for testing."""
    return [
        'I love this product',
        'This is terrible',
        'Amazing quality',
        'Worst experience ever',
        'Great value for money'
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return ['positive', 'negative', 'positive', 'negative', 'positive']
