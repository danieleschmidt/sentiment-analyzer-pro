"""Prometheus metrics for sentiment analyzer."""

import time
from typing import Dict, Any
from dataclasses import dataclass
from functools import wraps

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class MetricValue:
    """Simple metric value for when Prometheus is not available."""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: float


class MetricsCollector:
    """Centralized metrics collection with optional Prometheus support."""
    
    def __init__(self, enable_prometheus: bool = PROMETHEUS_AVAILABLE):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.fallback_metrics: Dict[str, MetricValue] = {}
        
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._init_prometheus_metrics()
        
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # API request metrics
        self.request_counter = Counter(
            'sentiment_requests_total',
            'Total number of sentiment analysis requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'sentiment_request_duration_seconds',
            'Time spent processing sentiment analysis requests',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Model metrics
        self.model_predictions = Counter(
            'sentiment_predictions_total',
            'Total number of predictions made',
            ['model_type', 'prediction'],
            registry=self.registry
        )
        
        self.model_load_duration = Histogram(
            'sentiment_model_load_duration_seconds',
            'Time spent loading models',
            ['model_type'],
            registry=self.registry
        )
        
        # System metrics
        self.active_connections = Gauge(
            'sentiment_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        # Training metrics
        self.training_duration = Histogram(
            'sentiment_training_duration_seconds',
            'Time spent training models',
            ['model_type'],
            registry=self.registry
        )
        
        self.training_accuracy = Gauge(
            'sentiment_training_accuracy',
            'Model training accuracy',
            ['model_type'],
            registry=self.registry
        )
    
    def inc_request_counter(self, method: str, endpoint: str, status: str):
        """Increment request counter."""
        if self.enable_prometheus:
            self.request_counter.labels(method=method, endpoint=endpoint, status=status).inc()
        else:
            key = f"requests_{method}_{endpoint}_{status}"
            self.fallback_metrics[key] = MetricValue(key, 1, 
                {"method": method, "endpoint": endpoint, "status": status}, time.time())
    
    def observe_request_duration(self, method: str, endpoint: str, duration: float):
        """Record request duration."""
        if self.enable_prometheus:
            self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        else:
            key = f"duration_{method}_{endpoint}"
            self.fallback_metrics[key] = MetricValue(key, duration,
                {"method": method, "endpoint": endpoint}, time.time())
    
    def inc_prediction_counter(self, model_type: str, prediction: str):
        """Increment prediction counter."""
        if self.enable_prometheus:
            self.model_predictions.labels(model_type=model_type, prediction=prediction).inc()
        else:
            key = f"predictions_{model_type}_{prediction}"
            self.fallback_metrics[key] = MetricValue(key, 1,
                {"model_type": model_type, "prediction": prediction}, time.time())
    
    def observe_model_load_duration(self, model_type: str, duration: float):
        """Record model load duration."""
        if self.enable_prometheus:
            self.model_load_duration.labels(model_type=model_type).observe(duration)
        else:
            key = f"model_load_{model_type}"
            self.fallback_metrics[key] = MetricValue(key, duration,
                {"model_type": model_type}, time.time())
    
    def set_active_connections(self, count: int):
        """Set active connections gauge."""
        if self.enable_prometheus:
            self.active_connections.set(count)
        else:
            self.fallback_metrics["active_connections"] = MetricValue(
                "active_connections", count, {}, time.time())
    
    def observe_training_duration(self, model_type: str, duration: float):
        """Record training duration."""
        if self.enable_prometheus:
            self.training_duration.labels(model_type=model_type).observe(duration)
        else:
            key = f"training_duration_{model_type}"
            self.fallback_metrics[key] = MetricValue(key, duration,
                {"model_type": model_type}, time.time())
    
    def set_training_accuracy(self, model_type: str, accuracy: float):
        """Set training accuracy gauge."""
        if self.enable_prometheus:
            self.training_accuracy.labels(model_type=model_type).set(accuracy)
        else:
            key = f"training_accuracy_{model_type}"
            self.fallback_metrics[key] = MetricValue(key, accuracy,
                {"model_type": model_type}, time.time())
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if self.enable_prometheus:
            return generate_latest(self.registry).decode('utf-8')
        else:
            # Return simple text format for fallback metrics
            lines = []
            for metric in self.fallback_metrics.values():
                labels = ','.join(f'{k}="{v}"' for k, v in metric.labels.items())
                if labels:
                    lines.append(f"{metric.name}{{{labels}}} {metric.value}")
                else:
                    lines.append(f"{metric.name} {metric.value}")
            return '\n'.join(lines)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary as dictionary."""
        if self.enable_prometheus:
            # In a real implementation, we'd parse the Prometheus metrics
            # For now, return a simple summary
            return {
                "prometheus_enabled": True,
                "metrics_endpoint": "/metrics",
                "collectors": len(self.registry._collector_to_names)
            }
        else:
            return {
                "prometheus_enabled": False,
                "fallback_metrics_count": len(self.fallback_metrics),
                "latest_metrics": list(self.fallback_metrics.keys())[-5:]  # Last 5 metrics
            }


# Global metrics instance
metrics = MetricsCollector()


def monitor_api_request(method: str, endpoint: str):
    """Decorator to monitor API requests."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                metrics.inc_request_counter(method, endpoint, status)
                metrics.observe_request_duration(method, endpoint, duration)
        
        return wrapper
    return decorator


def monitor_model_prediction(model_type: str):
    """Decorator to monitor model predictions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Try to extract prediction from result
            prediction = "unknown"
            if isinstance(result, dict) and "prediction" in result:
                prediction = str(result["prediction"])
            elif isinstance(result, str):
                prediction = result
            
            metrics.inc_prediction_counter(model_type, prediction)
            return result
        
        return wrapper
    return decorator


def monitor_model_loading(model_type: str):
    """Decorator to monitor model loading time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics.observe_model_load_duration(model_type, duration)
        
        return wrapper
    return decorator


def monitor_training(model_type: str):
    """Decorator to monitor model training."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Try to extract accuracy from result
                if isinstance(result, dict) and "accuracy" in result:
                    metrics.set_training_accuracy(model_type, result["accuracy"])
                
                return result
            finally:
                duration = time.time() - start_time
                metrics.observe_training_duration(model_type, duration)
        
        return wrapper
    return decorator