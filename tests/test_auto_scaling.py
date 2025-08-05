"""Tests for auto-scaling functionality."""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock

from src.auto_scaling import AutoScaler, ScalingConfig, ScalingMetrics, get_auto_scaler


@pytest.fixture
def scaling_config():
    """Create a test scaling configuration with lower thresholds."""
    return ScalingConfig(
        max_cpu_threshold=50.0,
        max_memory_threshold=60.0,
        max_response_time_ms=100.0,
        min_cpu_threshold=20.0,
        min_memory_threshold=30.0,
        scale_up_cooldown=1,  # Short cooldown for testing
        scale_down_cooldown=1,
        check_interval=0.1,  # Fast checking for tests
        min_instances=1,
        max_instances=3
    )


@pytest.fixture
def auto_scaler(scaling_config):
    """Create an AutoScaler instance for testing."""
    return AutoScaler(scaling_config)


def test_scaling_metrics_creation():
    """Test ScalingMetrics dataclass creation."""
    metrics = ScalingMetrics(
        cpu_usage=50.0,
        memory_usage=70.0,
        request_rate=25.0,
        response_time_ms=150.0,
        queue_depth=10,
        error_rate=2.5
    )
    
    assert metrics.cpu_usage == 50.0
    assert metrics.memory_usage == 70.0
    assert metrics.request_rate == 25.0
    assert metrics.response_time_ms == 150.0
    assert metrics.queue_depth == 10
    assert metrics.error_rate == 2.5


def test_auto_scaler_initialization(auto_scaler, scaling_config):
    """Test AutoScaler initialization."""
    assert auto_scaler.config == scaling_config
    assert auto_scaler.current_instances == scaling_config.min_instances
    assert not auto_scaler.is_monitoring
    assert len(auto_scaler.metrics_history) == 0


def test_record_metrics(auto_scaler):
    """Test metrics recording."""
    metrics = ScalingMetrics(cpu_usage=45.0, memory_usage=55.0)
    auto_scaler.record_metrics(metrics)
    
    assert len(auto_scaler.metrics_history) == 1
    recorded_metrics = auto_scaler.get_current_metrics()
    assert recorded_metrics.cpu_usage == 45.0
    assert recorded_metrics.memory_usage == 55.0
    assert hasattr(recorded_metrics, 'timestamp')


def test_get_average_metrics(auto_scaler):
    """Test average metrics calculation."""
    # Record multiple metrics
    auto_scaler.record_metrics(ScalingMetrics(cpu_usage=40.0, memory_usage=50.0))
    time.sleep(0.01)  # Small delay to ensure different timestamps
    auto_scaler.record_metrics(ScalingMetrics(cpu_usage=60.0, memory_usage=70.0))
    
    avg_metrics = auto_scaler.get_average_metrics(window_seconds=1)
    assert avg_metrics is not None
    assert avg_metrics.cpu_usage == 50.0  # Average of 40 and 60
    assert avg_metrics.memory_usage == 60.0  # Average of 50 and 70


def test_should_scale_up(auto_scaler):
    """Test scale up decision logic."""
    # Metrics below threshold - should not scale up
    low_metrics = ScalingMetrics(cpu_usage=30.0, memory_usage=40.0)
    assert not auto_scaler.should_scale_up(low_metrics)
    
    # High CPU - should scale up
    high_cpu_metrics = ScalingMetrics(cpu_usage=60.0, memory_usage=40.0)
    assert auto_scaler.should_scale_up(high_cpu_metrics)
    
    # High memory - should scale up
    high_memory_metrics = ScalingMetrics(cpu_usage=30.0, memory_usage=70.0)
    assert auto_scaler.should_scale_up(high_memory_metrics)
    
    # High response time - should scale up
    high_response_metrics = ScalingMetrics(response_time_ms=150.0)
    assert auto_scaler.should_scale_up(high_response_metrics)


def test_should_scale_down(auto_scaler):
    """Test scale down decision logic."""
    # Start with more than min instances
    auto_scaler.current_instances = 2
    
    # High metrics - should not scale down
    high_metrics = ScalingMetrics(cpu_usage=40.0, memory_usage=50.0)
    assert not auto_scaler.should_scale_down(high_metrics)
    
    # All low metrics - should scale down
    low_metrics = ScalingMetrics(
        cpu_usage=15.0, 
        memory_usage=25.0,
        response_time_ms=30.0,
        request_rate=10.0,
        queue_depth=2
    )
    assert auto_scaler.should_scale_down(low_metrics)


def test_scale_up_cooldown(auto_scaler):
    """Test scale up cooldown period."""
    high_metrics = ScalingMetrics(cpu_usage=60.0)
    
    # First scale up should work
    assert auto_scaler.should_scale_up(high_metrics)
    auto_scaler.scale_up()
    
    # Immediate second scale up should be blocked by cooldown
    assert not auto_scaler.should_scale_up(high_metrics)
    
    # After cooldown, should work again
    auto_scaler.last_scale_up = time.time() - auto_scaler.config.scale_up_cooldown - 1
    assert auto_scaler.should_scale_up(high_metrics)


def test_scale_down_cooldown(auto_scaler):
    """Test scale down cooldown period."""
    auto_scaler.current_instances = 2
    low_metrics = ScalingMetrics(
        cpu_usage=15.0, memory_usage=25.0, response_time_ms=30.0,
        request_rate=10.0, queue_depth=2
    )
    
    # First scale down should work
    assert auto_scaler.should_scale_down(low_metrics)
    auto_scaler.scale_down()
    
    # Immediate second scale down should be blocked by cooldown
    assert not auto_scaler.should_scale_down(low_metrics)


def test_scale_up_with_callback(auto_scaler):
    """Test scale up with callback."""
    callback_called = False
    new_instance_count = None
    
    def scale_up_callback(instances):
        nonlocal callback_called, new_instance_count
        callback_called = True
        new_instance_count = instances
    
    auto_scaler.set_scale_callbacks(scale_up_callback, None)
    auto_scaler.scale_up()
    
    assert callback_called
    assert new_instance_count == 2
    assert auto_scaler.current_instances == 2


def test_scale_down_with_callback(auto_scaler):
    """Test scale down with callback."""
    auto_scaler.current_instances = 2
    callback_called = False
    new_instance_count = None
    
    def scale_down_callback(instances):
        nonlocal callback_called, new_instance_count
        callback_called = True
        new_instance_count = instances
    
    auto_scaler.set_scale_callbacks(None, scale_down_callback)
    auto_scaler.scale_down()
    
    assert callback_called
    assert new_instance_count == 1
    assert auto_scaler.current_instances == 1


def test_scale_up_limits(auto_scaler):
    """Test scale up respects max instances limit."""
    auto_scaler.current_instances = auto_scaler.config.max_instances
    initial_count = auto_scaler.current_instances
    
    auto_scaler.scale_up()
    
    # Should not scale beyond max
    assert auto_scaler.current_instances == initial_count


def test_scale_down_limits(auto_scaler):
    """Test scale down respects min instances limit."""
    auto_scaler.current_instances = auto_scaler.config.min_instances
    initial_count = auto_scaler.current_instances
    
    auto_scaler.scale_down()
    
    # Should not scale below min
    assert auto_scaler.current_instances == initial_count


def test_evaluate_scaling(auto_scaler):
    """Test scaling evaluation logic."""
    # Record high metrics and evaluate - should scale up
    high_metrics = ScalingMetrics(cpu_usage=60.0, memory_usage=70.0)
    auto_scaler.record_metrics(high_metrics)
    
    initial_instances = auto_scaler.current_instances
    auto_scaler.evaluate_scaling()
    
    assert auto_scaler.current_instances > initial_instances


def test_get_status(auto_scaler):
    """Test status retrieval."""
    # Record some metrics
    auto_scaler.record_metrics(ScalingMetrics(cpu_usage=45.0))
    
    status = auto_scaler.get_status()
    
    assert "current_instances" in status
    assert "min_instances" in status
    assert "max_instances" in status
    assert "is_monitoring" in status
    assert "current_metrics" in status
    assert "metrics_history_size" in status
    
    assert status["current_instances"] == auto_scaler.current_instances
    assert status["min_instances"] == auto_scaler.config.min_instances
    assert status["max_instances"] == auto_scaler.config.max_instances
    assert status["metrics_history_size"] == 1


def test_monitoring_lifecycle(auto_scaler):
    """Test monitoring start/stop lifecycle."""
    assert not auto_scaler.is_monitoring
    
    auto_scaler.start_monitoring()
    assert auto_scaler.is_monitoring
    assert auto_scaler.monitor_thread is not None
    
    # Give monitoring thread a moment to start
    time.sleep(0.1)
    
    auto_scaler.stop_monitoring()
    assert not auto_scaler.is_monitoring


def test_callback_error_handling(auto_scaler):
    """Test that callback errors don't crash the scaler."""
    def failing_callback(instances):
        raise Exception("Callback failed!")
    
    auto_scaler.set_scale_callbacks(failing_callback, None)
    
    # Should not raise exception despite callback failure
    auto_scaler.scale_up()
    assert auto_scaler.current_instances == 2


def test_global_auto_scaler():
    """Test global auto-scaler getter."""
    scaler1 = get_auto_scaler()
    scaler2 = get_auto_scaler()
    
    # Should return the same instance
    assert scaler1 is scaler2
    assert isinstance(scaler1, AutoScaler)


@pytest.mark.integration
def test_auto_scaling_integration_with_webapp():
    """Integration test with webapp metrics collection."""
    pytest.importorskip('flask')
    pytest.importorskip('psutil')
    
    from src import webapp
    from src.models import build_model
    import joblib
    import os
    
    # Create test model
    model = build_model()
    model.fit(["good", "bad"], ["positive", "negative"])
    model_file = "/tmp/test_scaling_model.joblib"
    joblib.dump(model, model_file)
    
    webapp.load_model.cache_clear()
    webapp.load_model._test_model_path = model_file
    webapp._reset_counters()
    
    # Get auto-scaler and record initial status
    auto_scaler = get_auto_scaler()
    initial_metrics_count = len(auto_scaler.metrics_history)
    
    try:
        with webapp.app.test_client() as client:
            # Make some requests to generate metrics
            for _ in range(3):
                client.post('/predict', json={'text': 'test text'})
            
            # Check that metrics were recorded
            assert len(auto_scaler.metrics_history) > initial_metrics_count
            
            # Get current metrics
            current_metrics = auto_scaler.get_current_metrics()
            assert current_metrics is not None
            assert hasattr(current_metrics, 'cpu_usage')
            assert hasattr(current_metrics, 'memory_usage')
            assert hasattr(current_metrics, 'request_rate')
    
    finally:
        # Clean up
        if os.path.exists(model_file):
            os.unlink(model_file)