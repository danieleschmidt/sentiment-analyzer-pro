"""Prometheus metrics collection and export for Sentiment Analyzer Pro."""

import time
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from contextlib import contextmanager
import threading
import logging

# Try to import prometheus_client, gracefully handle if not available
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info, generate_latest, 
        CollectorRegistry, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Fallback mock classes for when prometheus_client is not installed
    class MockMetric:
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
    
    Counter = Histogram = Gauge = Info = MockMetric
    generate_latest = lambda *args: b"# Prometheus metrics not available\n"
    CONTENT_TYPE_LATEST = "text/plain"
    CollectorRegistry = lambda: None

# Thread-safe metrics storage
_metrics_lock = threading.Lock()
_request_durations = deque(maxlen=1000)
_prediction_durations = deque(maxlen=1000)


class MetricsCollector:
    """Centralized metrics collection for the application."""
    
    def __init__(self):
        """Initialize metrics collectors."""
        self.logger = logging.getLogger(__name__)
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Set up Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus client not available. Metrics will be collected but not exported.")
            self._setup_fallback_metrics()
            return
        
        # Request metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests by method and status',
            ['method', 'status', 'endpoint'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Prediction metrics
        self.predictions_total = Counter(
            'predictions_total',
            'Total number of predictions made',
            ['model_type'],
            registry=self.registry
        )
        
        self.prediction_duration_seconds = Histogram(
            'prediction_duration_seconds',
            'Prediction processing time in seconds',
            ['model_type'],
            registry=self.registry
        )
        
        # Application metrics
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.model_load_timestamp = Gauge(
            'model_load_timestamp',
            'Timestamp when model was last loaded',
            ['model_path'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'errors_total',
            'Total number of errors by type',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Rate limiting metrics
        self.rate_limit_exceeded_total = Counter(
            'rate_limit_exceeded_total',
            'Total number of rate limit violations',
            ['client_ip'],
            registry=self.registry
        )
        
        # Application info
        self.app_info = Info(
            'sentiment_analyzer_info',
            'Application information',
            registry=self.registry
        )
        
        # Text processing metrics
        self.text_length_histogram = Histogram(
            'text_length_characters',
            'Distribution of input text lengths',
            buckets=[10, 50, 100, 500, 1000, 5000, 10000],
            registry=self.registry
        )
        
        # Memory and performance metrics
        self.memory_usage_bytes = Gauge(
            'memory_usage_bytes',
            'Current memory usage in bytes',
            ['component'],
            registry=self.registry
        )
    
    def _setup_fallback_metrics(self):
        """Set up fallback metrics storage when Prometheus is not available."""
        self._fallback_counters = defaultdict(int)
        self._fallback_histograms = defaultdict(list)
        self._fallback_gauges = defaultdict(float)
        self._fallback_info = {}
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        if PROMETHEUS_AVAILABLE:
            self.http_requests_total.labels(
                method=method, 
                status=str(status_code), 
                endpoint=endpoint
            ).inc()
            self.http_request_duration_seconds.labels(
                method=method, 
                endpoint=endpoint
            ).observe(duration)
        else:
            self._fallback_counters[f'http_requests_{method}_{status_code}_{endpoint}'] += 1
            self._fallback_histograms[f'http_duration_{method}_{endpoint}'].append(duration)
        
        # Keep in-memory history for dashboard
        with _metrics_lock:
            _request_durations.append({
                'timestamp': time.time(),
                'method': method,
                'endpoint': endpoint,
                'status': status_code,
                'duration': duration
            })
    
    def record_prediction(self, model_type: str, duration: float, text_length: int):
        """Record prediction metrics."""
        if PROMETHEUS_AVAILABLE:
            self.predictions_total.labels(model_type=model_type).inc()
            self.prediction_duration_seconds.labels(model_type=model_type).observe(duration)
            self.text_length_histogram.observe(text_length)
        else:
            self._fallback_counters[f'predictions_{model_type}'] += 1
            self._fallback_histograms[f'prediction_duration_{model_type}'].append(duration)
            self._fallback_histograms['text_lengths'].append(text_length)
        
        # Keep in-memory history
        with _metrics_lock:
            _prediction_durations.append({
                'timestamp': time.time(),
                'model_type': model_type,
                'duration': duration,
                'text_length': text_length
            })
    
    def record_error(self, error_type: str, component: str):
        """Record error metrics."""
        if PROMETHEUS_AVAILABLE:
            self.errors_total.labels(error_type=error_type, component=component).inc()
        else:
            self._fallback_counters[f'errors_{error_type}_{component}'] += 1
    
    def record_rate_limit_exceeded(self, client_ip: str):
        """Record rate limit violation."""
        if PROMETHEUS_AVAILABLE:
            # Anonymize IP for privacy (keep only first two octets)
            anonymized_ip = '.'.join(client_ip.split('.')[:2]) + '.x.x' if client_ip else 'unknown'
            self.rate_limit_exceeded_total.labels(client_ip=anonymized_ip).inc()
        else:
            self._fallback_counters['rate_limit_exceeded'] += 1
    
    def set_active_connections(self, count: int):
        """Set current active connections count."""
        if PROMETHEUS_AVAILABLE:
            self.active_connections.set(count)
        else:
            self._fallback_gauges['active_connections'] = count
    
    def set_model_load_time(self, model_path: str, timestamp: float):
        """Record when a model was loaded."""
        if PROMETHEUS_AVAILABLE:
            self.model_load_timestamp.labels(model_path=model_path).set(timestamp)
        else:
            self._fallback_gauges[f'model_load_{model_path}'] = timestamp
    
    def set_memory_usage(self, component: str, bytes_used: int):
        """Record memory usage for a component."""
        if PROMETHEUS_AVAILABLE:
            self.memory_usage_bytes.labels(component=component).set(bytes_used)
        else:
            self._fallback_gauges[f'memory_{component}'] = bytes_used
    
    def set_app_info(self, version: str, build_info: Optional[Dict[str, str]] = None):
        """Set application information."""
        info_dict = {'version': version}
        if build_info:
            info_dict.update(build_info)
        
        if PROMETHEUS_AVAILABLE:
            self.app_info.info(info_dict)
        else:
            self._fallback_info.update(info_dict)
    
    def get_prometheus_metrics(self) -> bytes:
        """Get Prometheus-formatted metrics."""
        if PROMETHEUS_AVAILABLE and self.registry:
            return generate_latest(self.registry)
        else:
            # Return fallback metrics in Prometheus format
            lines = ["# Prometheus metrics (fallback mode)\n"]
            
            # Counters
            for key, value in self._fallback_counters.items():
                lines.append(f"# TYPE {key} counter\n")
                lines.append(f"{key} {value}\n")
            
            # Gauges
            for key, value in self._fallback_gauges.items():
                lines.append(f"# TYPE {key} gauge\n")
                lines.append(f"{key} {value}\n")
            
            return ''.join(lines).encode('utf-8')
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get metrics data for dashboard display."""
        with _metrics_lock:
            recent_requests = list(_request_durations)[-100:]  # Last 100 requests
            recent_predictions = list(_prediction_durations)[-100:]  # Last 100 predictions
        
        # Calculate basic statistics
        if recent_requests:
            avg_request_duration = sum(r['duration'] for r in recent_requests) / len(recent_requests)
            total_requests = len(recent_requests)
        else:
            avg_request_duration = 0
            total_requests = 0
        
        if recent_predictions:
            avg_prediction_duration = sum(p['duration'] for p in recent_predictions) / len(recent_predictions)
            total_predictions = len(recent_predictions)
            avg_text_length = sum(p['text_length'] for p in recent_predictions) / len(recent_predictions)
        else:
            avg_prediction_duration = 0
            total_predictions = 0
            avg_text_length = 0
        
        return {
            'requests': {
                'total': total_requests,
                'avg_duration': avg_request_duration,
                'recent': recent_requests
            },
            'predictions': {
                'total': total_predictions,
                'avg_duration': avg_prediction_duration,
                'avg_text_length': avg_text_length,
                'recent': recent_predictions
            },
            'system': {
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'timestamp': time.time()
            }
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()


@contextmanager
def measure_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Context manager for measuring execution time."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        if metric_name == 'prediction' and labels:
            metrics_collector.record_prediction(
                labels.get('model_type', 'unknown'),
                duration,
                labels.get('text_length', 0)
            )
        elif metric_name == 'http_request' and labels:
            metrics_collector.record_http_request(
                labels.get('method', 'unknown'),
                labels.get('endpoint', 'unknown'),
                labels.get('status_code', 200),
                duration
            )


def get_metrics_content_type() -> str:
    """Get the appropriate content type for metrics endpoint."""
    return CONTENT_TYPE_LATEST


def setup_app_info(version: str, build_info: Optional[Dict[str, str]] = None):
    """Initialize application information metrics."""
    metrics_collector.set_app_info(version, build_info)