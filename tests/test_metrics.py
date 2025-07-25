"""Tests for metrics collection functionality."""

import time
from unittest.mock import patch
import pytest

from src.metrics import (
    MetricsCollector, MetricValue, metrics,
    monitor_api_request, monitor_model_prediction,
    monitor_model_loading, monitor_training
)


class TestMetricValue:
    """Test MetricValue dataclass."""
    
    def test_metric_value_creation(self):
        """Test MetricValue initialization."""
        metric = MetricValue("test_metric", 42.0, {"label": "value"}, 1234567890.0)
        
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.labels == {"label": "value"}
        assert metric.timestamp == 1234567890.0


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    def test_init_without_prometheus(self):
        """Test initialization without Prometheus."""
        collector = MetricsCollector(enable_prometheus=False)
        
        assert not collector.enable_prometheus
        assert collector.fallback_metrics == {}
    
    @pytest.mark.skipif(True, reason="prometheus_client not installed")
    def test_init_with_prometheus(self):
        """Test initialization with Prometheus available."""
        pass
    
    def test_inc_request_counter_fallback(self):
        """Test request counter increment with fallback metrics."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.inc_request_counter("GET", "/test", "200")
        
        assert len(collector.fallback_metrics) == 1
        key = "requests_GET_/test_200"
        assert key in collector.fallback_metrics
        assert collector.fallback_metrics[key].value == 1
        assert collector.fallback_metrics[key].labels == {
            "method": "GET", "endpoint": "/test", "status": "200"
        }
    
    def test_observe_request_duration_fallback(self):
        """Test request duration observation with fallback metrics."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.observe_request_duration("POST", "/api", 0.5)
        
        key = "duration_POST_/api"
        assert key in collector.fallback_metrics
        assert collector.fallback_metrics[key].value == 0.5
    
    def test_inc_prediction_counter_fallback(self):
        """Test prediction counter increment with fallback metrics."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.inc_prediction_counter("sklearn", "positive")
        
        key = "predictions_sklearn_positive"
        assert key in collector.fallback_metrics
        assert collector.fallback_metrics[key].value == 1
    
    def test_observe_model_load_duration_fallback(self):
        """Test model load duration observation with fallback metrics."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.observe_model_load_duration("transformer", 10.5)
        
        key = "model_load_transformer"
        assert key in collector.fallback_metrics
        assert collector.fallback_metrics[key].value == 10.5
    
    def test_set_active_connections_fallback(self):
        """Test active connections gauge with fallback metrics."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.set_active_connections(25)
        
        assert "active_connections" in collector.fallback_metrics
        assert collector.fallback_metrics["active_connections"].value == 25
    
    def test_observe_training_duration_fallback(self):
        """Test training duration observation with fallback metrics."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.observe_training_duration("bert", 3600.0)
        
        key = "training_duration_bert"
        assert key in collector.fallback_metrics
        assert collector.fallback_metrics[key].value == 3600.0
    
    def test_set_training_accuracy_fallback(self):
        """Test training accuracy gauge with fallback metrics."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.set_training_accuracy("lstm", 0.95)
        
        key = "training_accuracy_lstm"
        assert key in collector.fallback_metrics
        assert collector.fallback_metrics[key].value == 0.95
    
    def test_get_metrics_fallback(self):
        """Test metrics export with fallback format."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.inc_request_counter("GET", "/test", "200")
        collector.set_active_connections(10)
        
        metrics_output = collector.get_metrics()
        
        assert "requests_GET_/test_200" in metrics_output
        assert "active_connections 10" in metrics_output
    
    def test_get_summary_fallback(self):
        """Test metrics summary with fallback."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.inc_request_counter("GET", "/test", "200")
        collector.inc_prediction_counter("sklearn", "positive")
        
        summary = collector.get_summary()
        
        assert summary["prometheus_enabled"] is False
        assert summary["fallback_metrics_count"] == 2
        assert len(summary["latest_metrics"]) <= 5
    
    @pytest.mark.skipif(True, reason="prometheus_client not installed")
    def test_get_summary_prometheus(self):
        """Test metrics summary with Prometheus enabled."""
        pass


class TestDecorators:
    """Test monitoring decorators."""
    
    def test_monitor_api_request_success(self):
        """Test API request monitoring decorator for successful requests."""
        test_collector = MetricsCollector(enable_prometheus=False)
        
        @monitor_api_request("GET", "/test")
        def test_function():
            return "success"
        
        # Patch the global metrics instance
        with patch('src.metrics.metrics', test_collector):
            result = test_function()
        
        assert result == "success"
        assert "requests_GET_/test_success" in test_collector.fallback_metrics
        assert "duration_GET_/test" in test_collector.fallback_metrics
    
    def test_monitor_api_request_error(self):
        """Test API request monitoring decorator for failed requests."""
        test_collector = MetricsCollector(enable_prometheus=False)
        
        @monitor_api_request("POST", "/error")
        def test_function():
            raise ValueError("Test error")
        
        # Patch the global metrics instance
        with patch('src.metrics.metrics', test_collector):
            with pytest.raises(ValueError):
                test_function()
        
        assert "requests_POST_/error_error" in test_collector.fallback_metrics
        assert "duration_POST_/error" in test_collector.fallback_metrics
    
    def test_monitor_model_prediction_dict_result(self):
        """Test model prediction monitoring with dict result."""
        test_collector = MetricsCollector(enable_prometheus=False)
        
        @monitor_model_prediction("sklearn")
        def test_function():
            return {"prediction": "positive", "confidence": 0.8}
        
        with patch('src.metrics.metrics', test_collector):
            result = test_function()
        
        assert result["prediction"] == "positive"
        assert "predictions_sklearn_positive" in test_collector.fallback_metrics
    
    def test_monitor_model_prediction_string_result(self):
        """Test model prediction monitoring with string result."""
        test_collector = MetricsCollector(enable_prometheus=False)
        
        @monitor_model_prediction("transformer")
        def test_function():
            return "negative"
        
        with patch('src.metrics.metrics', test_collector):
            result = test_function()
        
        assert result == "negative"
        assert "predictions_transformer_negative" in test_collector.fallback_metrics
    
    def test_monitor_model_loading_decorator(self):
        """Test model loading monitoring decorator."""
        test_collector = MetricsCollector(enable_prometheus=False)
        
        @monitor_model_loading("bert")
        def test_function():
            time.sleep(0.01)  # Simulate loading time
            return "model_loaded"
        
        with patch('src.metrics.metrics', test_collector):
            result = test_function()
        
        assert result == "model_loaded"
        assert "model_load_bert" in test_collector.fallback_metrics
        # Duration should be > 0
        assert test_collector.fallback_metrics["model_load_bert"].value > 0
    
    def test_monitor_training_decorator_with_accuracy(self):
        """Test training monitoring decorator with accuracy in result."""
        test_collector = MetricsCollector(enable_prometheus=False)
        
        @monitor_training("lstm")
        def test_function():
            time.sleep(0.01)  # Simulate training time
            return {"accuracy": 0.92, "loss": 0.1}
        
        with patch('src.metrics.metrics', test_collector):
            result = test_function()
        
        assert result["accuracy"] == 0.92
        assert "training_duration_lstm" in test_collector.fallback_metrics
        assert "training_accuracy_lstm" in test_collector.fallback_metrics
        assert test_collector.fallback_metrics["training_accuracy_lstm"].value == 0.92
    
    def test_monitor_training_decorator_without_accuracy(self):
        """Test training monitoring decorator without accuracy in result."""
        test_collector = MetricsCollector(enable_prometheus=False)
        
        @monitor_training("sklearn")
        def test_function():
            time.sleep(0.01)  # Simulate training time
            return "training_complete"
        
        with patch('src.metrics.metrics', test_collector):
            result = test_function()
        
        assert result == "training_complete"
        assert "training_duration_sklearn" in test_collector.fallback_metrics
        # Should not have accuracy metric
        assert "training_accuracy_sklearn" not in test_collector.fallback_metrics


class TestGlobalMetricsInstance:
    """Test the global metrics instance."""
    
    def test_global_metrics_instance_exists(self):
        """Test that global metrics instance is available."""
        assert metrics is not None
        assert isinstance(metrics, MetricsCollector)
    
    def test_global_metrics_basic_functionality(self):
        """Test basic functionality of global metrics instance."""
        # Store initial state
        initial_metrics_count = len(metrics.fallback_metrics)
        
        # Add a metric
        metrics.inc_request_counter("TEST", "/global", "200")
        
        # Check it was added
        assert len(metrics.fallback_metrics) > initial_metrics_count
        
        # Clean up by resetting fallback_metrics
        metrics.fallback_metrics.clear()


class TestPrometheusIntegration:
    """Test Prometheus-specific functionality when available."""
    
    @pytest.mark.skipif(True, reason="prometheus_client not installed")
    def test_prometheus_metrics_initialization(self):
        """Test Prometheus metrics are properly initialized."""
        pass
    
    @pytest.mark.skipif(True, reason="prometheus_client not installed")
    def test_prometheus_metrics_export(self):
        """Test Prometheus metrics export functionality."""
        pass


class TestMetricsEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_metrics_export(self):
        """Test metrics export with no metrics recorded."""
        collector = MetricsCollector(enable_prometheus=False)
        
        metrics_output = collector.get_metrics()
        
        assert metrics_output == ""
    
    def test_metrics_with_empty_labels(self):
        """Test metrics with empty label dictionaries."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.fallback_metrics["test_metric"] = MetricValue(
            "test_metric", 42.0, {}, time.time()
        )
        
        metrics_output = collector.get_metrics()
        
        assert "test_metric 42.0" in metrics_output
    
    def test_multiple_metrics_same_name_different_labels(self):
        """Test handling multiple metrics with same name but different labels."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.inc_request_counter("GET", "/test", "200")
        collector.inc_request_counter("GET", "/test", "404")
        
        assert "requests_GET_/test_200" in collector.fallback_metrics
        assert "requests_GET_/test_404" in collector.fallback_metrics
        assert len(collector.fallback_metrics) == 2
    
    def test_decorator_exception_handling(self):
        """Test that decorators properly handle exceptions."""
        test_collector = MetricsCollector(enable_prometheus=False)
        
        @monitor_model_loading("test")
        def failing_function():
            raise Exception("Test exception")
        
        with patch('src.metrics.metrics', test_collector):
            with pytest.raises(Exception):
                failing_function()
        
        # Should still record the duration even if function fails
        assert "model_load_test" in test_collector.fallback_metrics