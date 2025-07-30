"""
Distributed tracing configuration for microservices observability.

This module provides OpenTelemetry integration for tracing
requests across service boundaries with automatic instrumentation.
"""

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sklearn import SklearnInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
import os
from typing import Optional


class TracingConfig:
    """Configuration for distributed tracing."""
    
    def __init__(self, 
                 service_name: str = "sentiment-analyzer",
                 service_version: str = "0.1.0",
                 environment: str = "development"):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        
        # Jaeger configuration
        self.jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces")
        
        # OTLP configuration  
        self.otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
        
        # Sampling configuration
        self.sampling_rate = float(os.getenv("TRACE_SAMPLING_RATE", "1.0"))
        
        self.setup_tracing()
    
    def setup_tracing(self):
        """Configure OpenTelemetry tracing."""
        # Create resource with service information
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": self.service_version,
            "deployment.environment": self.environment,
        })
        
        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider(resource=resource))
        tracer_provider = trace.get_tracer_provider()
        
        # Configure exporters
        self._setup_exporters(tracer_provider)
        
        # Instrument libraries
        self._setup_instrumentation()
    
    def _setup_exporters(self, tracer_provider):
        """Set up trace exporters (Jaeger, OTLP)."""
        # Jaeger exporter
        if self.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
                collector_endpoint=self.jaeger_endpoint,
            )
            jaeger_processor = BatchSpanProcessor(jaeger_exporter)
            tracer_provider.add_span_processor(jaeger_processor)
        
        # OTLP exporter (for services like Honeycomb, Lightstep)
        if self.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.otlp_endpoint,
                headers=self._get_otlp_headers()
            )
            otlp_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(otlp_processor)
    
    def _get_otlp_headers(self) -> dict:
        """Get OTLP headers for authentication."""
        headers = {}
        
        # Honeycomb
        if honeycomb_key := os.getenv("HONEYCOMB_API_KEY"):
            headers["x-honeycomb-team"] = honeycomb_key
        
        # Custom headers
        if custom_headers := os.getenv("OTLP_HEADERS"):
            for header in custom_headers.split(","):
                key, value = header.split("=", 1)
                headers[key.strip()] = value.strip()
        
        return headers
    
    def _setup_instrumentation(self):
        """Set up automatic instrumentation for common libraries."""
        # Flask instrumentation
        FlaskInstrumentor().instrument()
        
        # Requests instrumentation (for HTTP calls)
        RequestsInstrumentor().instrument()
        
        # Scikit-learn instrumentation
        SklearnInstrumentor().instrument()
        
        # Logging instrumentation
        LoggingInstrumentor().instrument(set_logging_format=True)


class SentimentTracer:
    """Custom tracer for sentiment analysis operations."""
    
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
    
    def trace_prediction(self, text: str, model_name: str):
        """Create a span for sentiment prediction."""
        with self.tracer.start_as_current_span("sentiment_prediction") as span:
            # Add attributes
            span.set_attribute("sentiment.text_length", len(text))
            span.set_attribute("sentiment.model_name", model_name)
            span.set_attribute("sentiment.input_hash", hash(text) % 10000)
            
            return span
    
    def trace_model_loading(self, model_name: str, model_path: str):
        """Create a span for model loading."""
        with self.tracer.start_as_current_span("model_loading") as span:
            span.set_attribute("model.name", model_name)
            span.set_attribute("model.path", model_path)
            
            return span
    
    def trace_preprocessing(self, operation: str, text_count: int):
        """Create a span for text preprocessing."""
        with self.tracer.start_as_current_span(f"preprocessing_{operation}") as span:
            span.set_attribute("preprocessing.operation", operation)
            span.set_attribute("preprocessing.text_count", text_count)
            
            return span
    
    def add_prediction_result(self, span, prediction: str, confidence: float):
        """Add prediction results to current span."""
        span.set_attribute("sentiment.prediction", prediction)
        span.set_attribute("sentiment.confidence", confidence)
        
        # Add events for significant predictions
        if confidence < 0.6:
            span.add_event("low_confidence_prediction", {
                "confidence": confidence,
                "prediction": prediction
            })


def trace_sentiment_operation(operation_name: str):
    """Decorator to automatically trace sentiment analysis operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            
            with tracer.start_as_current_span(operation_name) as span:
                # Add function metadata
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                
                except Exception as e:
                    # Record exception in span
                    span.record_exception(e)
                    span.set_status(
                        trace.Status(
                            trace.StatusCode.ERROR,
                            description=str(e)
                        )
                    )
                    raise
        
        return wrapper
    return decorator


class TraceContextManager:
    """Context manager for custom span creation."""
    
    def __init__(self, operation_name: str, attributes: dict = None):
        self.operation_name = operation_name
        self.attributes = attributes or {}
        self.tracer = trace.get_tracer(__name__)
        self.span = None
    
    def __enter__(self):
        self.span = self.tracer.start_span(self.operation_name)
        
        # Add attributes
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type is not None:
                self.span.record_exception(exc_val)
                self.span.set_status(
                    trace.Status(
                        trace.StatusCode.ERROR,
                        description=str(exc_val)
                    )
                )
            else:
                self.span.set_status(trace.Status(trace.StatusCode.OK))
            
            self.span.end()


# Global tracer instance
sentiment_tracer = SentimentTracer()

# Convenience functions
def get_tracer(name: str = __name__):
    """Get a tracer instance."""
    return trace.get_tracer(name)

def create_span(operation_name: str, attributes: dict = None):
    """Create a custom span context manager."""
    return TraceContextManager(operation_name, attributes)

def get_current_span():
    """Get the current active span."""
    return trace.get_current_span()

def set_span_attribute(key: str, value):
    """Set attribute on current span."""
    current_span = trace.get_current_span()
    if current_span:
        current_span.set_attribute(key, value)


# Example usage
if __name__ == "__main__":
    # Initialize tracing
    config = TracingConfig()
    
    # Manual span creation
    with create_span("example_operation", {"user_id": "12345"}) as span:
        span.add_event("processing_started")
        
        # Simulate work
        import time
        time.sleep(0.1)
        
        span.add_event("processing_completed")
    
    # Using decorator
    @trace_sentiment_operation("example_prediction")
    def predict_example(text: str):
        set_span_attribute("text_length", len(text))
        return "positive"
    
    result = predict_example("This is a great product!")
    print(f"Prediction: {result}")