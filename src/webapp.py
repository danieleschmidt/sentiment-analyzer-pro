"""Flask web server for sentiment predictions."""
from __future__ import annotations

import argparse
import time
from functools import lru_cache, wraps
from typing import Any
from collections import defaultdict, deque
import hashlib
import threading
import psutil
import os

from flask import Flask, jsonify, request
from importlib import metadata
from pydantic import ValidationError

import joblib

from .config import Config
from .schemas import PredictRequest, BatchPredictRequest
from .models import SentimentModel
from .metrics import metrics, monitor_api_request, monitor_model_loading
from .logging_config import setup_logging, get_logger, log_security_event, log_api_request
from .auto_scaling import get_auto_scaler, ScalingMetrics

app = Flask(__name__)
logger = get_logger(__name__)

# Rate limiting - simple in-memory store
RATE_LIMIT_REQUESTS = defaultdict(lambda: deque())

# Performance tracking
REQUEST_COUNT = 0
PREDICTION_COUNT = 0


def _reset_counters():
    """Reset counters for testing purposes."""
    global REQUEST_COUNT, PREDICTION_COUNT, PREDICTION_CACHE
    with CACHE_LOCK:
        REQUEST_COUNT = 0
        PREDICTION_COUNT = 0
        PREDICTION_CACHE.clear()

# Prediction cache - Thread-safe LRU cache for predictions
PREDICTION_CACHE = {}
CACHE_LOCK = threading.RLock()
MAX_CACHE_SIZE = 1000

# Performance tracking for auto-scaling
REQUEST_TIMES = deque(maxlen=1000)
ERROR_COUNT = 0


def _get_system_metrics() -> ScalingMetrics:
    """Collect current system metrics for auto-scaling."""
    try:
        # CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Request rate (requests per second over last minute)
        now = time.time()
        recent_requests = [t for t in REQUEST_TIMES if now - t < 60]
        request_rate = len(recent_requests) / 60.0 if recent_requests else 0.0
        
        # Average response time
        if recent_requests:
            response_times = [r.get('duration', 0) for r in REQUEST_TIMES 
                            if isinstance(r, dict) and now - r.get('timestamp', 0) < 60]
            avg_response_time = sum(response_times) / len(response_times) * 1000 if response_times else 0
        else:
            avg_response_time = 0
        
        # Error rate over last minute
        recent_errors = ERROR_COUNT  # Simple approximation
        error_rate = (recent_errors / max(len(recent_requests), 1)) * 100
        
        return ScalingMetrics(
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            request_rate=request_rate,
            response_time_ms=avg_response_time,
            queue_depth=0,  # Not applicable for this simple setup
            error_rate=error_rate
        )
    except Exception as e:
        logger.warning(f"Failed to collect system metrics: {e}")
        return ScalingMetrics()

try:
    APP_VERSION = metadata.version("sentiment-analyzer-pro")
except metadata.PackageNotFoundError:  # pragma: no cover - local usage
    APP_VERSION = "0.0.0"


@lru_cache(maxsize=1)
@monitor_model_loading("sklearn")
def load_model(path: str | None = None) -> SentimentModel:
    """Load and cache a trained model from disk."""
    if path is None:
        path = Config.MODEL_PATH
    return joblib.load(path)


def _cache_prediction(text: str, prediction: str) -> str:
    """Cache prediction result with LRU eviction."""
    global PREDICTION_CACHE
    
    # Create cache key from text hash
    cache_key = hashlib.sha256(text.encode()).hexdigest()[:16]
    
    with CACHE_LOCK:
        # LRU eviction if cache is full
        if len(PREDICTION_CACHE) >= MAX_CACHE_SIZE:
            # Remove oldest entry (simple FIFO for performance)
            oldest_key = next(iter(PREDICTION_CACHE))
            del PREDICTION_CACHE[oldest_key]
        
        PREDICTION_CACHE[cache_key] = prediction
    
    return prediction


def _get_cached_prediction(text: str) -> str | None:
    """Get cached prediction if available."""
    cache_key = hashlib.sha256(text.encode()).hexdigest()[:16]
    
    with CACHE_LOCK:
        if cache_key in PREDICTION_CACHE:
            # Move to end for LRU
            prediction = PREDICTION_CACHE.pop(cache_key)
            PREDICTION_CACHE[cache_key] = prediction
            return prediction
    
    return None


def _check_rate_limit(client_ip: str) -> bool:
    """Check if client is within rate limits."""
    now = time.time()
    client_requests = RATE_LIMIT_REQUESTS[client_ip]
    
    # Remove old requests outside the window
    while client_requests and client_requests[0] <= now - Config.RATE_LIMIT_WINDOW:
        client_requests.popleft()
    
    # Check if under limit
    if len(client_requests) >= Config.RATE_LIMIT_MAX_REQUESTS:
        return False
    
    # Add current request
    client_requests.append(now)
    return True


@app.before_request
def _log_request() -> None:  # pragma: no cover - logging side effect
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    
    # Store request start time for duration calculation
    request.start_time = time.time()
    
    # Track request for metrics
    REQUEST_TIMES.append(time.time())
    
    # Rate limiting
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    if not _check_rate_limit(client_ip):
        log_security_event(
            logger, 
            'rate_limit_exceeded', 
            client_ip=client_ip,
            details={'path': request.path, 'method': request.method}
        )
        return jsonify({"error": "Rate limit exceeded"}), 429


@app.after_request
def _add_security_headers(response):
    """Add security headers and log API requests."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    
    # Log API request with duration and collect metrics
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        
        # Update request times with duration info for metrics
        if REQUEST_TIMES:
            REQUEST_TIMES[-1] = {
                'timestamp': request.start_time,
                'duration': duration
            }
        
        # Collect and record system metrics for auto-scaling
        try:
            auto_scaler = get_auto_scaler()
            system_metrics = _get_system_metrics()
            auto_scaler.record_metrics(system_metrics)
        except Exception as e:
            logger.debug(f"Failed to record auto-scaling metrics: {e}")
        
        log_api_request(
            logger, 
            request.method, 
            request.path, 
            response.status_code, 
            duration, 
            client_ip
        )
    
    return response


@app.route("/predict", methods=["POST"])
@monitor_api_request("POST", "/predict")
def predict():
    # Validate Content-Type
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json(silent=True) or {}
    try:
        req = PredictRequest(**data)
    except ValidationError as exc:
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        log_security_event(
            logger, 
            'validation_error', 
            client_ip=client_ip,
            details={'errors': exc.errors(), 'data_keys': list(data.keys())}
        )
        return jsonify({"error": "Invalid input", "details": exc.errors()}), 400
    except (TypeError, KeyError, AttributeError) as exc:
        logger.error("Request parsing error", extra={
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'request_data': str(data) if 'data' in locals() else 'unavailable'
        })
        return jsonify({"error": "Invalid request format"}), 400
    except Exception as exc:
        logger.error("Unexpected validation error", extra={
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'request_data': str(data) if 'data' in locals() else 'unavailable'
        })
        return jsonify({"error": "Internal server error during validation"}), 500
    
    try:
        start_time = time.time()
        
        # Check cache first for performance
        cached_prediction = _get_cached_prediction(req.text)
        if cached_prediction:
            cache_time = time.time() - start_time
            logger.info("Prediction served from cache", extra={
                'event_type': 'prediction_cached',
                'cache_time_seconds': cache_time,
                'text_length': len(req.text),
                'prediction': cached_prediction
            })
            return jsonify({"prediction": cached_prediction, "cached": True})
        
        # For testing, allow model path override via global variable
        model_path = getattr(load_model, '_test_model_path', None) or Config.MODEL_PATH
        model = load_model(model_path)
        prediction = model.predict([req.text])[0]
        
        # Cache the prediction for future requests
        _cache_prediction(req.text, prediction)
        
        # Record prediction in metrics
        metrics.inc_prediction_counter("sklearn", str(prediction))
        prediction_time = time.time() - start_time
        
        global PREDICTION_COUNT
        PREDICTION_COUNT += 1
        
        # Log prediction performance
        logger.info("Prediction completed", extra={
            'event_type': 'prediction',
            'prediction_time_seconds': prediction_time,
            'text_length': len(req.text),
            'prediction': prediction,
            'cached': False
        })
        
        return jsonify({"prediction": prediction, "cached": False})
    except FileNotFoundError as exc:
        logger.error("Model file not found", extra={
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'model_path': Config.MODEL_PATH
        })
        return jsonify({"error": "Model not available"}), 503
    except (ValueError, AttributeError) as exc:
        logger.error("Model prediction error", extra={
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'model_path': Config.MODEL_PATH,
            'text_length': len(req.text) if 'req' in locals() else 0
        })
        return jsonify({"error": "Invalid input for prediction"}), 400
    except Exception as exc:
        logger.error("Unexpected prediction error", extra={
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'model_path': Config.MODEL_PATH,
            'text_length': len(req.text) if 'req' in locals() else 0
        })
        return jsonify({"error": "Internal server error"}), 500


@app.route("/predict/batch", methods=["POST"])
@monitor_api_request("POST", "/predict/batch")
def predict_batch():
    """Handle batch predictions for improved performance."""
    # Validate Content-Type
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json(silent=True) or {}
    try:
        req = BatchPredictRequest(**data)
    except ValidationError as exc:
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        log_security_event(
            logger, 
            'validation_error_batch', 
            client_ip=client_ip,
            details={'errors': exc.errors(), 'data_keys': list(data.keys())}
        )
        return jsonify({"error": "Invalid input", "details": exc.errors()}), 400
    except Exception as exc:
        logger.error("Unexpected batch validation error", extra={
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'request_data': str(data) if 'data' in locals() else 'unavailable'
        })
        return jsonify({"error": "Internal server error during validation"}), 500
    
    try:
        start_time = time.time()
        results = []
        cache_hits = 0
        
        # Check cache for each text
        cached_results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(req.texts):
            cached_prediction = _get_cached_prediction(text)
            if cached_prediction:
                cached_results.append((i, cached_prediction))
                cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Batch predict uncached texts for better performance
        predictions = []
        if uncached_texts:
            model_path = getattr(load_model, '_test_model_path', None) or Config.MODEL_PATH
            model = load_model(model_path)
            predictions = model.predict(uncached_texts)
            
            # Cache new predictions
            for text, prediction in zip(uncached_texts, predictions):
                _cache_prediction(text, prediction)
        
        # Combine cached and new results in original order
        all_results = [None] * len(req.texts)
        
        # Add cached results
        for idx, prediction in cached_results:
            all_results[idx] = {"prediction": prediction, "cached": True}
        
        # Add new predictions
        for i, (idx, prediction) in enumerate(zip(uncached_indices, predictions)):
            all_results[idx] = {"prediction": prediction, "cached": False}
        
        # Record metrics
        for prediction_result in all_results:
            if not prediction_result["cached"]:
                metrics.inc_prediction_counter("sklearn", str(prediction_result["prediction"]))
        
        prediction_time = time.time() - start_time
        
        global PREDICTION_COUNT
        PREDICTION_COUNT += len(req.texts)
        
        # Log batch prediction performance
        logger.info("Batch prediction completed", extra={
            'event_type': 'batch_prediction',
            'prediction_time_seconds': prediction_time,
            'batch_size': len(req.texts),
            'cache_hits': cache_hits,
            'cache_hit_rate': cache_hits / len(req.texts) if req.texts else 0,
            'new_predictions': len(uncached_texts)
        })
        
        return jsonify({
            "predictions": all_results,
            "batch_size": len(req.texts),
            "cache_hits": cache_hits,
            "processing_time_seconds": round(prediction_time, 4)
        })
        
    except FileNotFoundError as exc:
        logger.error("Model file not found for batch prediction", extra={
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'model_path': Config.MODEL_PATH
        })
        return jsonify({"error": "Model not available"}), 503
    except Exception as exc:
        logger.error("Unexpected batch prediction error", extra={
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'batch_size': len(req.texts) if 'req' in locals() else 0
        })
        return jsonify({"error": "Internal server error"}), 500


@app.route("/", methods=["GET"])
@monitor_api_request("GET", "/")
def index():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/version", methods=["GET"])
@monitor_api_request("GET", "/version")
def version():
    """Return the running package version."""
    return jsonify({"version": APP_VERSION})


@app.route("/metrics", methods=["GET"])
def metrics_endpoint() -> Any:
    """Prometheus metrics endpoint."""
    return metrics.get_metrics(), 200, {'Content-Type': 'text/plain'}


@app.route("/metrics/summary", methods=["GET"])
@monitor_api_request("GET", "/metrics/summary")
def metrics_summary() -> Any:
    """Return enhanced service metrics summary."""
    with CACHE_LOCK:
        cache_size = len(PREDICTION_CACHE)
        cache_hit_rate = getattr(metrics_summary, '_cache_hits', 0) / max(1, PREDICTION_COUNT) * 100
    
    base_metrics = {
        "requests": REQUEST_COUNT, 
        "predictions": PREDICTION_COUNT,
        "cache_size": cache_size,
        "cache_hit_rate_percent": round(cache_hit_rate, 2),
        "max_cache_size": MAX_CACHE_SIZE
    }
    enhanced_metrics = metrics.get_summary()
    return jsonify({**base_metrics, **enhanced_metrics})


@app.route("/health", methods=["GET"])
@monitor_api_request("GET", "/health")
def health_check() -> Any:
    """Comprehensive health check endpoint."""
    try:
        # Basic service health
        health_status = {"status": "healthy", "timestamp": time.time()}
        
        # System metrics
        system_metrics = _get_system_metrics()
        health_status["system"] = {
            "cpu_usage_percent": system_metrics.cpu_usage,
            "memory_usage_percent": system_metrics.memory_usage,
            "request_rate_per_second": system_metrics.request_rate
        }
        
        # Model availability
        try:
            model = load_model()
            health_status["model"] = {"status": "loaded", "type": "sklearn"}
        except Exception as e:
            health_status["model"] = {"status": "error", "error": str(e)}
            health_status["status"] = "degraded"
        
        # Auto-scaling status
        auto_scaler = get_auto_scaler()
        health_status["auto_scaling"] = auto_scaler.get_status()
        
        # Cache status
        with CACHE_LOCK:
            health_status["cache"] = {
                "size": len(PREDICTION_CACHE),
                "max_size": MAX_CACHE_SIZE,
                "utilization_percent": (len(PREDICTION_CACHE) / MAX_CACHE_SIZE) * 100
            }
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": time.time()
        }), 503


@app.route("/scaling/status", methods=["GET"])
@monitor_api_request("GET", "/scaling/status")
def scaling_status() -> Any:
    """Get detailed auto-scaling status and metrics."""
    try:
        auto_scaler = get_auto_scaler()
        status = auto_scaler.get_status()
        
        # Add recent system metrics
        current_metrics = _get_system_metrics()
        avg_metrics = auto_scaler.get_average_metrics()
        
        return jsonify({
            "auto_scaling": status,
            "current_metrics": current_metrics.__dict__,
            "average_metrics": avg_metrics.__dict__ if avg_metrics else None,
            "thresholds": {
                "scale_up": {
                    "cpu_percent": 70.0,
                    "memory_percent": 80.0,
                    "response_time_ms": 200.0,
                    "request_rate_per_second": 100.0
                },
                "scale_down": {
                    "cpu_percent": 30.0,
                    "memory_percent": 40.0,
                    "response_time_ms": 50.0,
                    "request_rate_per_second": 20.0
                }
            }
        })
    except Exception as e:
        logger.error(f"Scaling status check failed: {e}")
        return jsonify({"error": str(e)}), 500


def main(argv: list[str] | None = None) -> None:
    global MODEL_PATH
    parser = argparse.ArgumentParser(description="Run prediction web server")
    parser.add_argument("--model", default=MODEL_PATH, help="Trained model path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--structured-logs", action="store_true", 
                       help="Enable structured JSON logging")
    args = parser.parse_args(argv)

    # Configure logging
    setup_logging(level=args.log_level, structured=args.structured_logs)
    
    MODEL_PATH = args.model
    logger.info("Starting web server", extra={
        'host': args.host,
        'port': args.port,
        'model_path': MODEL_PATH,
        'structured_logs': args.structured_logs
    })
    
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()
