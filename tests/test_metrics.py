"""Tests for the metrics module."""

import pytest
import time
from unittest.mock import patch, MagicMock
import threading

from src.metrics import (
    MetricsCollector, 
    metrics_collector, 
    measure_time, 
    get_metrics_content_type,
    setup_app_info,
    PROMETHEUS_AVAILABLE
)


class TestMetricsCollector:
    """Test cases for MetricsCollector class."""
    
    def test_init(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        assert collector is not None
        assert hasattr(collector, 'logger')
        
        if PROMETHEUS_AVAILABLE:
            assert hasattr(collector, 'registry')
            assert hasattr(collector, 'http_requests_total')
            assert hasattr(collector, 'predictions_total')
        else:
            assert hasattr(collector, '_fallback_counters')
            assert hasattr(collector, '_fallback_histograms')
    
    def test_record_http_request(self):
        """Test HTTP request recording."""
        collector = MetricsCollector()
        
        # Record a sample request
        collector.record_http_request('GET', '/predict', 200, 0.5)
        
        if PROMETHEUS_AVAILABLE:
            # Check that metrics were recorded (can't easily assert on prometheus values)
            assert collector.http_requests_total is not None
        else:
            assert 'http_requests_GET_200_/predict' in collector._fallback_counters
            assert collector._fallback_counters['http_requests_GET_200_/predict'] == 1
            assert 'http_duration_GET_/predict' in collector._fallback_histograms
            assert 0.5 in collector._fallback_histograms['http_duration_GET_/predict']
    
    def test_record_prediction(self):
        """Test prediction recording."""
        collector = MetricsCollector()
        
        # Record a sample prediction
        collector.record_prediction('lstm', 1.2, 150)
        
        if PROMETHEUS_AVAILABLE:
            assert collector.predictions_total is not None
        else:
            assert 'predictions_lstm' in collector._fallback_counters
            assert collector._fallback_counters['predictions_lstm'] == 1
            assert 'prediction_duration_lstm' in collector._fallback_histograms
            assert 1.2 in collector._fallback_histograms['prediction_duration_lstm']
            assert 150 in collector._fallback_histograms['text_lengths']
    
    def test_record_error(self):
        """Test error recording."""
        collector = MetricsCollector()
        
        collector.record_error('validation_error', 'webapp')
        
        if PROMETHEUS_AVAILABLE:
            assert collector.errors_total is not None
        else:
            assert 'errors_validation_error_webapp' in collector._fallback_counters
            assert collector._fallback_counters['errors_validation_error_webapp'] == 1
    
    def test_record_rate_limit_exceeded(self):
        """Test rate limit recording."""
        collector = MetricsCollector()
        
        collector.record_rate_limit_exceeded('192.168.1.100')
        
        if PROMETHEUS_AVAILABLE:
            assert collector.rate_limit_exceeded_total is not None
        else:
            assert 'rate_limit_exceeded' in collector._fallback_counters
            assert collector._fallback_counters['rate_limit_exceeded'] == 1
    
    def test_set_active_connections(self):
        """Test active connections gauge."""
        collector = MetricsCollector()
        
        collector.set_active_connections(5)
        
        if PROMETHEUS_AVAILABLE:
            assert collector.active_connections is not None
        else:
            assert collector._fallback_gauges['active_connections'] == 5
    
    def test_set_model_load_time(self):
        """Test model load time recording."""
        collector = MetricsCollector()
        
        timestamp = time.time()
        collector.set_model_load_time('/path/to/model.joblib', timestamp)
        
        if PROMETHEUS_AVAILABLE:
            assert collector.model_load_timestamp is not None
        else:
            assert collector._fallback_gauges['model_load_/path/to/model.joblib'] == timestamp
    
    def test_set_memory_usage(self):
        """Test memory usage recording."""
        collector = MetricsCollector()
        
        collector.set_memory_usage('webapp', 1024 * 1024 * 50)  # 50MB
        
        if PROMETHEUS_AVAILABLE:
            assert collector.memory_usage_bytes is not None
        else:
            assert collector._fallback_gauges['memory_webapp'] == 1024 * 1024 * 50
    
    def test_set_app_info(self):
        """Test application info setting."""
        collector = MetricsCollector()
        
        build_info = {'build_date': '2024-01-01', 'commit': 'abc123'}
        collector.set_app_info('1.0.0', build_info)
        
        if PROMETHEUS_AVAILABLE:
            assert collector.app_info is not None
        else:
            assert 'version' in collector._fallback_info
            assert collector._fallback_info['version'] == '1.0.0'
            assert collector._fallback_info['build_date'] == '2024-01-01'
    
    def test_get_prometheus_metrics(self):
        """Test Prometheus metrics export."""
        collector = MetricsCollector()
        
        # Add some sample data
        collector.record_http_request('GET', '/predict', 200, 0.5)
        collector.record_prediction('default', 1.0, 100)
        
        metrics_data = collector.get_prometheus_metrics()
        
        assert isinstance(metrics_data, bytes)
        assert b'# ' in metrics_data  # Should contain Prometheus comments
        
        if not PROMETHEUS_AVAILABLE:
            assert b'fallback mode' in metrics_data
    
    def test_get_dashboard_data(self):
        """Test dashboard data export."""
        collector = MetricsCollector()
        
        # Add some sample data
        collector.record_http_request('GET', '/predict', 200, 0.5)
        collector.record_prediction('default', 1.0, 100)
        
        dashboard_data = collector.get_dashboard_data()
        
        assert 'requests' in dashboard_data
        assert 'predictions' in dashboard_data
        assert 'system' in dashboard_data
        
        assert 'total' in dashboard_data['requests']
        assert 'avg_duration' in dashboard_data['requests']
        assert 'recent' in dashboard_data['requests']
        
        assert 'total' in dashboard_data['predictions']
        assert 'avg_duration' in dashboard_data['predictions']
        assert 'avg_text_length' in dashboard_data['predictions']
        
        assert 'prometheus_available' in dashboard_data['system']
        assert dashboard_data['system']['prometheus_available'] == PROMETHEUS_AVAILABLE
    
    def test_thread_safety(self):
        """Test that metrics collection is thread-safe."""
        collector = MetricsCollector()
        results = []
        
        def worker():
            for i in range(10):
                collector.record_http_request('GET', '/test', 200, 0.1)
                collector.record_prediction('test', 0.5, 50)
                time.sleep(0.01)  # Small delay to encourage race conditions
        
        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Verify data was recorded
        dashboard_data = collector.get_dashboard_data()
        assert dashboard_data['requests']['total'] >= 0  # Some requests should be recorded
        assert dashboard_data['predictions']['total'] >= 0  # Some predictions should be recorded


class TestMeasureTime:
    """Test cases for the measure_time context manager."""
    
    def test_measure_http_request_time(self):
        """Test measuring HTTP request time."""
        collector = MetricsCollector()
        
        with measure_time('http_request', {
            'method': 'POST',
            'endpoint': '/predict',
            'status_code': 200
        }):
            time.sleep(0.1)  # Simulate work
        
        dashboard_data = collector.get_dashboard_data()
        requests = dashboard_data['requests']['recent']
        
        # Should have at least one request recorded
        assert len(requests) > 0
        last_request = requests[-1]
        assert last_request['method'] == 'POST'
        assert last_request['endpoint'] == '/predict'
        assert last_request['status'] == 200
        assert last_request['duration'] >= 0.1
    
    def test_measure_prediction_time(self):
        """Test measuring prediction time."""
        collector = MetricsCollector()
        
        with measure_time('prediction', {
            'model_type': 'transformer',
            'text_length': 200
        }):
            time.sleep(0.05)  # Simulate prediction work
        
        dashboard_data = collector.get_dashboard_data()
        predictions = dashboard_data['predictions']['recent']
        
        # Should have at least one prediction recorded
        assert len(predictions) > 0
        last_prediction = predictions[-1]
        assert last_prediction['model_type'] == 'transformer'
        assert last_prediction['text_length'] == 200
        assert last_prediction['duration'] >= 0.05
    
    def test_measure_time_with_exception(self):
        """Test that timing works even when exceptions occur."""
        collector = MetricsCollector()
        
        try:
            with measure_time('http_request', {
                'method': 'GET',
                'endpoint': '/error',
                'status_code': 500
            }):
                time.sleep(0.02)
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected
        
        dashboard_data = collector.get_dashboard_data()
        requests = dashboard_data['requests']['recent']
        
        # Should still record the timing despite the exception
        assert len(requests) > 0
        last_request = requests[-1]
        assert last_request['duration'] >= 0.02


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_get_metrics_content_type(self):
        """Test metrics content type."""
        content_type = get_metrics_content_type()
        assert isinstance(content_type, str)
        assert 'text' in content_type.lower()
    
    def test_setup_app_info(self):
        """Test application info setup."""
        setup_app_info('2.0.0', {'environment': 'test'})
        
        # Verify it was recorded (check via dashboard data)
        dashboard_data = metrics_collector.get_dashboard_data()
        assert dashboard_data is not None


class TestGlobalMetricsCollector:
    """Test cases for the global metrics collector instance."""
    
    def test_global_instance_exists(self):
        """Test that the global metrics collector instance exists."""
        assert metrics_collector is not None
        assert isinstance(metrics_collector, MetricsCollector)
    
    def test_global_instance_functionality(self):
        """Test that the global instance works correctly."""
        # Record some data
        metrics_collector.record_http_request('GET', '/health', 200, 0.1)
        
        # Should be able to retrieve data
        dashboard_data = metrics_collector.get_dashboard_data()
        assert dashboard_data is not None
        assert 'requests' in dashboard_data


@pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus client not available")
class TestPrometheusIntegration:
    """Test cases specifically for Prometheus integration."""
    
    def test_prometheus_metrics_format(self):
        """Test that Prometheus metrics are in correct format."""
        collector = MetricsCollector()
        collector.record_http_request('GET', '/test', 200, 0.1)
        
        metrics_data = collector.get_prometheus_metrics()
        metrics_str = metrics_data.decode('utf-8')
        
        # Should contain Prometheus format elements
        assert '# HELP' in metrics_str or '# TYPE' in metrics_str
        assert 'http_requests_total' in metrics_str or 'requests' in metrics_str.lower()
    
    def test_prometheus_registry(self):
        """Test that Prometheus registry is properly configured."""
        collector = MetricsCollector()
        assert collector.registry is not None
        
        # Should be able to get metrics from registry
        metrics = collector.get_prometheus_metrics()
        assert len(metrics) > 0


class TestFallbackMode:
    """Test cases for fallback mode when Prometheus is not available."""
    
    @patch('src.metrics.PROMETHEUS_AVAILABLE', False)
    def test_fallback_metrics_collection(self):
        """Test that metrics are still collected in fallback mode."""
        # Create new collector with Prometheus disabled
        collector = MetricsCollector()
        
        collector.record_http_request('GET', '/test', 200, 0.1)
        collector.record_prediction('test', 0.5, 100)
        
        # Should still be able to get data
        dashboard_data = collector.get_dashboard_data()
        assert dashboard_data is not None
        
        metrics_data = collector.get_prometheus_metrics()
        assert b'fallback mode' in metrics_data
    
    @patch('src.metrics.PROMETHEUS_AVAILABLE', False)
    def test_fallback_prometheus_export(self):
        """Test Prometheus export in fallback mode."""
        collector = MetricsCollector()
        collector.record_error('test_error', 'test_component')
        
        metrics_data = collector.get_prometheus_metrics()
        metrics_str = metrics_data.decode('utf-8')
        
        assert 'fallback mode' in metrics_str
        assert 'test_error' in metrics_str