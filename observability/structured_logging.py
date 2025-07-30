"""
Structured logging configuration for comprehensive observability.

This module provides structured logging with correlation IDs, 
performance metrics, and integration with monitoring systems.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional

import structlog


class StructuredLogger:
    """Structured logger with correlation tracking and performance metrics."""
    
    def __init__(self, service_name: str = "sentiment-analyzer"):
        self.service_name = service_name
        self.setup_logging()
    
    def setup_logging(self):
        """Configure structured logging with processors."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                self.add_service_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=None,  # Use default
            level=logging.INFO,
        )
    
    def add_service_info(self, logger, method_name, event_dict):
        """Add service metadata to log entries."""
        event_dict["service"] = self.service_name
        event_dict["version"] = "0.1.0"  # Should be dynamic
        return event_dict
    
    def get_logger(self, name: str) -> structlog.BoundLogger:
        """Get a structured logger instance."""
        return structlog.get_logger(name)


class CorrelationContext:
    """Context manager for request correlation tracking."""
    
    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.start_time = time.time()
        self.logger = structlog.get_logger(__name__)
    
    def __enter__(self):
        # Bind correlation ID to logger context
        self.bound_logger = self.logger.bind(
            correlation_id=self.correlation_id,
            timestamp=datetime.utcnow().isoformat()
        )
        return self.bound_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.bound_logger.info(
                "request_completed",
                duration_ms=round(duration * 1000, 2),
                correlation_id=self.correlation_id
            )
        else:
            self.bound_logger.error(
                "request_failed",
                duration_ms=round(duration * 1000, 2),
                correlation_id=self.correlation_id,
                error_type=exc_type.__name__,
                error_message=str(exc_val)
            )


def log_performance(operation_name: str):
    """Decorator to log performance metrics for functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = structlog.get_logger(__name__)
            start_time = time.time()
            
            # Log start
            logger.info(
                "operation_started",
                operation=operation_name,
                function=func.__name__,
                module=func.__module__
            )
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log success
                logger.info(
                    "operation_completed",
                    operation=operation_name,
                    function=func.__name__,
                    duration_ms=round(duration * 1000, 2),
                    status="success"
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log error
                logger.error(
                    "operation_failed",
                    operation=operation_name,
                    function=func.__name__,
                    duration_ms=round(duration * 1000, 2),
                    status="error",
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator


class MetricsCollector:
    """Collect and export custom metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.logger = structlog.get_logger(__name__)
    
    def increment_counter(self, name: str, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        labels = labels or {}
        key = f"{name}_{hash(frozenset(labels.items()))}"
        
        self.metrics.setdefault(key, {
            'type': 'counter',
            'name': name,
            'value': 0,
            'labels': labels
        })
        
        self.metrics[key]['value'] += 1
        
        # Log metric update
        self.logger.info(
            "metric_updated",
            metric_type="counter",
            metric_name=name,
            metric_value=self.metrics[key]['value'],
            labels=labels
        )
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric."""
        labels = labels or {}
        
        self.logger.info(
            "metric_recorded",
            metric_type="histogram",
            metric_name=name,
            metric_value=value,
            labels=labels
        )
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics in Prometheus format."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': self.metrics
        }


# Global instances
structured_logger = StructuredLogger()
metrics_collector = MetricsCollector()

# Convenience functions
def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structured_logger.get_logger(name)

def correlation_context(correlation_id: Optional[str] = None) -> CorrelationContext:
    """Create a correlation context for request tracking."""
    return CorrelationContext(correlation_id)

def log_metric(metric_type: str, name: str, value: Any = 1, labels: Dict[str, str] = None):
    """Log a metric with structured logging."""
    if metric_type == "counter":
        metrics_collector.increment_counter(name, labels)
    elif metric_type == "histogram":
        metrics_collector.record_histogram(name, value, labels)


# Example usage
if __name__ == "__main__":
    logger = get_logger(__name__)
    
    # Basic logging
    logger.info("Application started", version="0.1.0")
    
    # Correlation context
    with correlation_context() as ctx_logger:
        ctx_logger.info("Processing request", user_id="12345")
        
        # Simulate work
        time.sleep(0.1)
        
        ctx_logger.info("Request processed successfully")
    
    # Performance logging
    @log_performance("model_prediction")
    def predict_sentiment(text: str):
        time.sleep(0.05)  # Simulate prediction
        return "positive"
    
    result = predict_sentiment("This is great!")
    logger.info("Prediction result", text="This is great!", result=result)
    
    # Metrics
    log_metric("counter", "predictions_total", labels={"model": "transformer"})
    log_metric("histogram", "prediction_duration_ms", 50.5, labels={"model": "transformer"})