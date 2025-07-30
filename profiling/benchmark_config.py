"""
Performance benchmarking configuration for sentiment analysis components.

This module provides benchmark configurations for pytest-benchmark
to ensure consistent performance testing across the codebase.
"""

import os
from pathlib import Path

# Benchmark configuration
BENCHMARK_CONFIG = {
    'disable_gc': True,
    'min_rounds': 5,
    'max_time': 30,
    'min_time': 0.1,
    'warmup': True,
    'warmup_iterations': 3,
}

# Performance thresholds (in seconds)
PERFORMANCE_THRESHOLDS = {
    'model_loading': {
        'nb_model': 0.5,
        'lstm_model': 2.0,
        'transformer_model': 5.0,
    },
    'single_prediction': {
        'nb_model': 0.01,
        'lstm_model': 0.05,
        'transformer_model': 0.1,
    },
    'batch_prediction_100': {
        'nb_model': 0.5,
        'lstm_model': 2.0,
        'transformer_model': 5.0,
    },
    'preprocessing': {
        'tokenization': 0.001,
        'cleaning': 0.002,
        'vectorization': 0.01,
    }
}

# Memory usage thresholds (in MB)
MEMORY_THRESHOLDS = {
    'model_loading': {
        'nb_model': 50,
        'lstm_model': 200,
        'transformer_model': 500,
    },
    'prediction': {
        'single': 10,
        'batch_100': 100,
    }
}

# Test data configuration
TEST_DATA_CONFIG = {
    'sample_texts': [
        "This product is amazing!",
        "Terrible experience, would not recommend.",
        "Average quality, nothing special.",
        "Outstanding service and fast delivery.",
        "Complete waste of money."
    ],
    'batch_sizes': [1, 10, 50, 100, 500],
    'text_lengths': [10, 50, 100, 200, 500],  # characters
}

def get_benchmark_data_path() -> Path:
    """Get path to benchmark test data."""
    return Path(__file__).parent / "data"

def ensure_benchmark_data():
    """Ensure benchmark test data exists."""
    data_path = get_benchmark_data_path()
    data_path.mkdir(exist_ok=True)
    
    # Create sample benchmark data if it doesn't exist
    sample_file = data_path / "benchmark_reviews.csv"
    if not sample_file.exists():
        import pandas as pd
        
        # Generate synthetic test data for benchmarking
        texts = TEST_DATA_CONFIG['sample_texts'] * 200  # 1000 samples
        labels = ['positive', 'negative', 'neutral'] * 334  # Roughly balanced
        
        df = pd.DataFrame({
            'text': texts[:1000],
            'label': labels[:1000]
        })
        df.to_csv(sample_file, index=False)
    
    return sample_file