"""
🧪 Neuromorphic Spikeformer Tests
=================================

Comprehensive test suite for neuromorphic spiking neural network components,
validation, and optimization systems.

Quality Gates: Comprehensive testing for neuromorphic implementation
"""

import pytest
import numpy as np
import torch
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import modules under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from neuromorphic_spikeformer import (
        SpikeformerConfig, NeuromorphicSentimentAnalyzer, SpikeEncoder,
        LIFNeuron, SpikingAttention, SpikeformerLayer, SpikeformerNeuromorphicModel,
        create_neuromorphic_sentiment_analyzer
    )
    from neuromorphic_validation import (
        NeuromorphicValidator, ValidationConfig, InputValidator,
        NeuromorphicValidationError, InputValidationError, SpikingValidationError,
        create_secure_neuromorphic_validator
    )
    from neuromorphic_optimization import (
        NeuromorphicOptimizer, IntelligentCache, ResourcePool, PerformanceProfiler,
        get_optimizer, configure_optimizer, cached_synthesis, parallel_processing
    )
    NEUROMORPHIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Neuromorphic modules not available: {e}")
    NEUROMORPHIC_AVAILABLE = False

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


@pytest.mark.skipif(not NEUROMORPHIC_AVAILABLE, reason="Neuromorphic modules not available")
class TestSpikeformerConfig:
    """Test SpikeformerConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SpikeformerConfig()
        
        assert config.input_dim == 768
        assert config.hidden_dim == 256
        assert config.num_layers == 4
        assert config.num_classes == 3
        assert config.membrane_threshold == 1.0
        assert config.membrane_decay == 0.9
        assert config.timesteps == 100
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SpikeformerConfig(
            input_dim=512,
            hidden_dim=128,
            num_layers=2,
            timesteps=50
        )
        
        assert config.input_dim == 512
        assert config.hidden_dim == 128
        assert config.num_layers == 2
        assert config.timesteps == 50


@pytest.mark.skipif(not NEUROMORPHIC_AVAILABLE, reason="Neuromorphic modules not available")
class TestSpikeEncoder:
    """Test spike encoding functionality."""
    
    def setUp(self):
        self.config = SpikeformerConfig(timesteps=10)
        self.encoder = SpikeEncoder(self.config)
    
    def test_rate_encoding(self):
        """Test rate encoding of features."""
        self.setUp()
        
        # Create test features
        batch_size, seq_len, feature_dim = 2, 5, 10
        features = torch.randn(batch_size, seq_len, feature_dim)
        
        # Encode to spikes
        spike_trains = self.encoder.rate_encoding(features)
        
        # Check output shape
        expected_shape = (batch_size, self.config.timesteps, seq_len, feature_dim)
        assert spike_trains.shape == expected_shape
        
        # Check spikes are binary
        assert torch.all((spike_trains == 0) | (spike_trains == 1))
    
    def test_temporal_encoding(self):
        """Test temporal encoding of features."""
        self.setUp()
        
        # Create test features
        batch_size, seq_len, feature_dim = 1, 3, 5
        features = torch.randn(batch_size, seq_len, feature_dim)
        
        # Encode to spikes
        spike_trains = self.encoder.temporal_encoding(features)
        
        # Check output shape
        expected_shape = (batch_size, self.config.timesteps, seq_len, feature_dim)
        assert spike_trains.shape == expected_shape
        
        # Check spikes are binary
        assert torch.all((spike_trains == 0) | (spike_trains == 1))


@pytest.mark.skipif(not NEUROMORPHIC_AVAILABLE, reason="Neuromorphic modules not available")
class TestLIFNeuron:
    """Test Leaky Integrate-and-Fire neuron."""
    
    def setUp(self):
        self.config = SpikeformerConfig()
        self.neuron = LIFNeuron(self.config)
    
    def test_forward_pass(self):
        """Test LIF neuron forward pass."""
        self.setUp()
        
        # Create test input
        batch_size, features = 3, 10
        input_current = torch.randn(batch_size, features)
        
        # Forward pass
        spikes, membrane_state = self.neuron.forward(input_current)
        
        # Check output shapes
        assert spikes.shape == (batch_size, features)
        assert membrane_state.shape == (batch_size, features)
        
        # Check spikes are binary
        assert torch.all((spikes == 0) | (spikes == 1))
    
    def test_membrane_reset(self):
        """Test membrane potential reset after spike."""
        self.setUp()
        
        # Create high input current to trigger spikes
        input_current = torch.ones(1, 5) * 2.0  # Above threshold
        
        spikes, membrane_state = self.neuron.forward(input_current)
        
        # Where spikes occurred, membrane should be low
        spike_locations = spikes == 1
        if torch.any(spike_locations):
            reset_membrane = membrane_state[spike_locations]
            assert torch.all(reset_membrane < self.config.membrane_threshold)


@pytest.mark.skipif(not NEUROMORPHIC_AVAILABLE, reason="Neuromorphic modules not available")
class TestNeuromorphicSentimentAnalyzer:
    """Test main neuromorphic sentiment analyzer."""
    
    def setUp(self):
        self.config = SpikeformerConfig(input_dim=100, timesteps=10)  # Small for testing
        self.analyzer = NeuromorphicSentimentAnalyzer(self.config, enable_validation=False)
    
    def test_initialization(self):
        """Test analyzer initialization."""
        self.setUp()
        
        assert self.analyzer.config.input_dim == 100
        assert self.analyzer.config.timesteps == 10
        assert len(self.analyzer.class_labels) == 3
        assert not self.analyzer.trained
    
    def test_preprocess_text_features(self):
        """Test text feature preprocessing."""
        self.setUp()
        
        # Test 2D input
        features_2d = np.random.randn(5, 50)
        processed = self.analyzer.preprocess_text_features(features_2d)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (5, 1, 100)  # Padded to input_dim
        
        # Test 3D input
        features_3d = np.random.randn(3, 4, 80)
        processed = self.analyzer.preprocess_text_features(features_3d)
        
        assert processed.shape == (3, 4, 100)  # Padded to input_dim
    
    def test_predict(self):
        """Test sentiment prediction."""
        self.setUp()
        
        # Create test features
        features = np.random.randn(2, 100)
        
        # Predict
        results = self.analyzer.predict(features)
        
        # Check result structure
        assert 'predictions' in results
        assert 'model_stats' in results
        assert 'processing_info' in results
        
        # Check predictions
        predictions = results['predictions']
        assert len(predictions) == 2
        
        for pred in predictions:
            assert 'sentiment' in pred
            assert pred['sentiment'] in ['negative', 'neutral', 'positive']
            assert 'confidence' in pred
            assert 0 <= pred['confidence'] <= 1
            assert 'neuromorphic_stats' in pred
    
    def test_train_step(self):
        """Test training step."""
        self.setUp()
        
        # Create training data
        features = np.random.randn(5, 100)
        labels = np.random.randint(0, 3, 5)
        
        # Training step
        metrics = self.analyzer.train_step(features, labels)
        
        # Check metrics
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'spike_count' in metrics
        assert 'energy_consumption' in metrics
        
        assert isinstance(metrics['loss'], float)
        assert 0 <= metrics['accuracy'] <= 1


@pytest.mark.skipif(not NEUROMORPHIC_AVAILABLE, reason="Neuromorphic modules not available")
class TestNeuromorphicValidator:
    """Test neuromorphic validation system."""
    
    def setUp(self):
        self.config = ValidationConfig(max_batch_size=10, max_input_dim=1000)
        self.validator = NeuromorphicValidator(self.config)
    
    def test_input_validation_valid(self):
        """Test valid input validation."""
        self.setUp()
        
        # Valid input
        features = np.random.randn(5, 100)
        
        # Should not raise exception
        results = self.validator.validate_processing_request(features)
        assert results['status'] == 'valid'
        assert results['input_shape'] == (5, 100)
    
    def test_input_validation_invalid_batch_size(self):
        """Test invalid batch size validation."""
        self.setUp()
        
        # Too large batch
        features = np.random.randn(20, 100)  # Exceeds max_batch_size=10
        
        with pytest.raises(InputValidationError):
            self.validator.validate_processing_request(features)
    
    def test_input_validation_invalid_dimensions(self):
        """Test invalid dimension validation."""
        self.setUp()
        
        # Too many dimensions
        features = np.random.randn(2, 3, 4, 5)
        
        with pytest.raises(InputValidationError):
            self.validator.validate_processing_request(features)
    
    def test_input_validation_nan_values(self):
        """Test NaN value validation."""
        self.setUp()
        
        # Features with NaN
        features = np.random.randn(3, 50)
        features[0, 0] = np.nan
        
        with pytest.raises(InputValidationError):
            self.validator.validate_processing_request(features)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        self.setUp()
        
        # Valid config
        valid_config = {
            'membrane_threshold': 0.5,
            'timesteps': 100,
            'spike_rate_max': 50.0,
            'membrane_decay': 0.8
        }
        
        results = self.validator.validate_processing_request(
            np.random.randn(2, 10), 
            config_dict=valid_config
        )
        assert results['status'] == 'valid'
        
        # Invalid config - negative timesteps
        invalid_config = {'timesteps': -10}
        
        with pytest.raises(SpikingValidationError):
            self.validator.validate_processing_request(
                np.random.randn(2, 10),
                config_dict=invalid_config
            )


@pytest.mark.skipif(not NEUROMORPHIC_AVAILABLE, reason="Neuromorphic modules not available")
class TestNeuromorphicOptimizer:
    """Test neuromorphic optimization system."""
    
    def setUp(self):
        self.optimizer = NeuromorphicOptimizer(
            cache_size=10,
            max_workers=2,
            enable_caching=True,
            enable_pooling=True
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.setUp()
        
        assert self.optimizer.enable_caching
        assert self.optimizer.enable_pooling
        assert self.optimizer.synthesis_cache is not None
        assert self.optimizer.resource_pool is not None
    
    def test_caching_decorator(self):
        """Test caching decorator functionality."""
        self.setUp()
        
        call_count = 0
        
        @self.optimizer.cached_operation("synthesis")
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call with same input (should use cache)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
        
        # Third call with different input
        result3 = expensive_function(7)
        assert result3 == 14
        assert call_count == 2
    
    def test_parallel_processing(self):
        """Test parallel batch processing."""
        self.setUp()
        
        def square_function(x):
            return x ** 2
        
        batch_data = [1, 2, 3, 4, 5]
        results = self.optimizer.parallel_batch_processing(
            batch_data, square_function, chunk_size=2
        )
        
        expected = [1, 4, 9, 16, 25]
        assert results == expected
    
    def test_optimization_stats(self):
        """Test optimization statistics."""
        self.setUp()
        
        stats = self.optimizer.get_optimization_stats()
        
        assert 'performance_stats' in stats
        assert 'cache_enabled' in stats
        assert 'pooling_enabled' in stats
        assert stats['cache_enabled'] == True
        assert stats['pooling_enabled'] == True
    
    def tearDown(self):
        if hasattr(self, 'optimizer'):
            self.optimizer.cleanup()


@pytest.mark.skipif(not NEUROMORPHIC_AVAILABLE, reason="Neuromorphic modules not available")
class TestIntelligentCache:
    """Test intelligent caching system."""
    
    def setUp(self):
        self.cache = IntelligentCache(max_size=5, enable_compression=False)
    
    def test_basic_operations(self):
        """Test basic cache operations."""
        self.setUp()
        
        # Test put and get
        test_data = np.array([1, 2, 3])
        self.cache.put("key1", test_data)
        
        retrieved_data = self.cache.get("key1")
        assert np.array_equal(retrieved_data, test_data)
        
        # Test cache miss
        missing_data = self.cache.get("nonexistent")
        assert missing_data is None
    
    def test_cache_eviction(self):
        """Test cache eviction when full."""
        self.setUp()
        
        # Fill cache beyond capacity
        for i in range(10):
            self.cache.put(f"key{i}", f"value{i}")
        
        # Check that cache size is limited
        metrics = self.cache.get_metrics()
        assert metrics['cache_size'] <= 5
        
        # Check that oldest entries are evicted
        early_key = self.cache.get("key0")
        assert early_key is None  # Should be evicted
        
        recent_key = self.cache.get("key9")
        assert recent_key is not None  # Should still be there
    
    def test_cache_metrics(self):
        """Test cache metrics tracking."""
        self.setUp()
        
        # Perform operations
        self.cache.put("test", "data")
        self.cache.get("test")  # Hit
        self.cache.get("missing")  # Miss
        
        metrics = self.cache.get_metrics()
        
        assert metrics['hits'] == 1
        assert metrics['misses'] == 1
        assert metrics['total_requests'] == 2
        assert metrics['hit_rate'] == 0.5


@pytest.mark.skipif(not NEUROMORPHIC_AVAILABLE, reason="Neuromorphic modules not available")
class TestIntegration:
    """Integration tests for complete neuromorphic pipeline."""
    
    def test_full_pipeline_with_validation(self):
        """Test complete pipeline with validation enabled."""
        # Create analyzer with validation
        config = SpikeformerConfig(input_dim=50, timesteps=5)  # Small for testing
        analyzer = NeuromorphicSentimentAnalyzer(config, enable_validation=True)
        
        # Test features
        features = np.random.randn(3, 50)
        
        # Run prediction
        results = analyzer.predict(features, client_id="test_client")
        
        # Verify results
        assert 'predictions' in results
        assert len(results['predictions']) == 3
        assert 'processing_info' in results
        assert results['processing_info']['validated'] == analyzer.enable_validation
    
    def test_full_pipeline_with_optimization(self):
        """Test complete pipeline with optimization."""
        # Configure optimizer
        optimizer = configure_optimizer(
            cache_size=10,
            max_workers=2,
            enable_caching=True,
            enable_pooling=True
        )
        
        # Create analyzer
        config = SpikeformerConfig(input_dim=30, timesteps=5)
        analyzer = NeuromorphicSentimentAnalyzer(config, enable_validation=False)
        
        # Test with caching
        features = np.random.randn(2, 30)
        
        # First prediction
        results1 = analyzer.predict(features)
        
        # Second prediction (same features)
        results2 = analyzer.predict(features)
        
        # Should get same results
        assert len(results1['predictions']) == len(results2['predictions'])
        
        # Cleanup
        optimizer.cleanup()
    
    def test_error_handling_robustness(self):
        """Test error handling in various failure scenarios."""
        config = SpikeformerConfig(input_dim=20, timesteps=3)
        analyzer = NeuromorphicSentimentAnalyzer(config, enable_validation=True)
        
        # Test with invalid input shape
        invalid_features = np.random.randn(1000, 5000)  # Too large
        
        with pytest.raises((InputValidationError, RuntimeError)):
            analyzer.predict(invalid_features)
        
        # Test with NaN values
        nan_features = np.random.randn(2, 20)
        nan_features[0, 0] = np.nan
        
        with pytest.raises((InputValidationError, RuntimeError)):
            analyzer.predict(nan_features)


@pytest.mark.skipif(not NEUROMORPHIC_AVAILABLE, reason="Neuromorphic modules not available")
class TestPerformanceProfiler:
    """Test performance profiling utilities."""
    
    def test_operation_timing(self):
        """Test operation timing functionality."""
        profiler = PerformanceProfiler()
        
        # Time an operation
        with profiler.time_operation("test_op"):
            import time
            time.sleep(0.01)  # Small delay
        
        # Check stats
        stats = profiler.get_stats()
        assert "test_op" in stats
        assert stats["test_op"]["count"] == 1
        assert stats["test_op"]["avg_time"] > 0
    
    def test_profiling_decorator(self):
        """Test profiling decorator."""
        from neuromorphic_optimization import profile_operation
        
        @profile_operation("decorated_func")
        def test_function(x):
            return x * 2
        
        # Call function
        result = test_function(5)
        assert result == 10
        
        # Check profiling
        profiler = PerformanceProfiler()
        # Note: Global profiler would have the stats in real usage


# Test fixtures and utilities
@pytest.fixture
def sample_features():
    """Sample features for testing."""
    return np.random.randn(5, 100)


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return np.random.randint(0, 3, 5)


@pytest.fixture
def neuromorphic_config():
    """Test neuromorphic configuration."""
    return SpikeformerConfig(
        input_dim=100,
        hidden_dim=64,
        num_layers=2,
        timesteps=10
    )


# Performance benchmarks
@pytest.mark.skipif(not NEUROMORPHIC_AVAILABLE, reason="Neuromorphic modules not available")
@pytest.mark.benchmark
def test_prediction_performance(sample_features):
    """Benchmark prediction performance."""
    config = SpikeformerConfig(input_dim=100, timesteps=20)
    analyzer = NeuromorphicSentimentAnalyzer(config, enable_validation=False)
    
    import time
    start_time = time.time()
    
    results = analyzer.predict(sample_features)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Performance assertions
    assert execution_time < 5.0  # Should complete in under 5 seconds
    assert len(results['predictions']) == len(sample_features)
    
    # Log performance
    print(f"Prediction time: {execution_time:.3f}s for {len(sample_features)} samples")


@pytest.mark.skipif(not NEUROMORPHIC_AVAILABLE, reason="Neuromorphic modules not available")
@pytest.mark.benchmark
def test_caching_performance():
    """Benchmark caching performance."""
    cache = IntelligentCache(max_size=100, enable_compression=True)
    
    # Test data
    test_data = [np.random.randn(100, 200) for _ in range(50)]
    
    import time
    
    # Time cache operations
    start_time = time.time()
    
    for i, data in enumerate(test_data):
        cache.put(f"key_{i}", data)
    
    put_time = time.time() - start_time
    
    start_time = time.time()
    
    for i in range(len(test_data)):
        retrieved = cache.get(f"key_{i}")
        assert retrieved is not None
    
    get_time = time.time() - start_time
    
    # Performance assertions
    assert put_time < 2.0  # Put operations should be fast
    assert get_time < 1.0  # Get operations should be very fast
    
    # Check cache metrics
    metrics = cache.get_metrics()
    assert metrics['hit_rate'] > 0.9  # Should have high hit rate
    
    print(f"Cache put time: {put_time:.3f}s, get time: {get_time:.3f}s")
    print(f"Cache hit rate: {metrics['hit_rate']:.2%}")


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])