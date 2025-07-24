"""Flask web server for sentiment predictions."""
from __future__ import annotations

import argparse
import logging
import os
import time
from functools import lru_cache
from typing import Any
from collections import defaultdict, deque

from flask import Flask, jsonify, request
from importlib import metadata
from pydantic import ValidationError

import joblib

from .config import Config
from .schemas import PredictRequest
from .models import SentimentModel
from .logging_config import setup_logging, get_logger, log_security_event, log_api_request
from .metrics import metrics_collector, get_metrics_content_type, setup_app_info
from .dashboard import add_dashboard_routes
from .profiling import profiler, profile_function, start_profiling

app = Flask(__name__)
logger = get_logger(__name__)

# Rate limiting - simple in-memory store
RATE_LIMIT_REQUESTS = defaultdict(lambda: deque())

REQUEST_COUNT = 0
PREDICTION_COUNT = 0

try:
    APP_VERSION = metadata.version("sentiment-analyzer-pro")
except metadata.PackageNotFoundError:  # pragma: no cover - local usage
    APP_VERSION = "0.0.0"


@lru_cache(maxsize=1)
@profile_function('load_model')
def load_model(path: str | None = None) -> SentimentModel:
    """Load and cache a trained model from disk."""
    if path is None:
        path = Config.MODEL_PATH
    return joblib.load(path)


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
    
    # Rate limiting
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    if not _check_rate_limit(client_ip):
        log_security_event(
            logger, 
            'rate_limit_exceeded', 
            client_ip=client_ip,
            details={'path': request.path, 'method': request.method}
        )
        metrics_collector.record_rate_limit_exceeded(client_ip)
        metrics_collector.record_error('rate_limit', 'webapp')
        return jsonify({"error": "Rate limit exceeded"}), 429


@app.after_request
def _add_security_headers(response):
    """Add security headers and log API requests."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    
    # Log API request with duration and record metrics
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        log_api_request(
            logger, 
            request.method, 
            request.path, 
            response.status_code, 
            duration, 
            client_ip
        )
        # Record metrics
        metrics_collector.record_http_request(
            request.method,
            request.path,
            response.status_code,
            duration
        )
    
    return response


@app.route("/predict", methods=["POST"])
@profile_function('predict_endpoint')
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
        metrics_collector.record_error('validation_error', 'webapp')
        return jsonify({"error": "Invalid input", "details": exc.errors()}), 400
    except (TypeError, KeyError, AttributeError) as exc:
        logger.error("Request parsing error", extra={
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'request_data': str(data) if 'data' in locals() else 'unavailable'
        })
        metrics_collector.record_error('parsing_error', 'webapp')
        return jsonify({"error": "Invalid request format"}), 400
    except Exception as exc:
        logger.error("Unexpected validation error", extra={
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'request_data': str(data) if 'data' in locals() else 'unavailable'
        })
        metrics_collector.record_error('unexpected_validation_error', 'webapp')
        return jsonify({"error": "Internal server error during validation"}), 500
    
    try:
        start_time = time.time()
        # For testing, allow model path override via global variable
        model_path = getattr(load_model, '_test_model_path', None) or Config.MODEL_PATH
        model = load_model(model_path)
        prediction = model.predict([req.text])[0]
        prediction_time = time.time() - start_time
        
        global PREDICTION_COUNT
        PREDICTION_COUNT += 1
        
        # Log prediction performance and record metrics
        logger.info("Prediction completed", extra={
            'event_type': 'prediction',
            'prediction_time_seconds': prediction_time,
            'text_length': len(req.text),
            'prediction': prediction
        })
        
        # Record prediction metrics
        metrics_collector.record_prediction(
            'default',  # model type - could be expanded to detect actual model type
            prediction_time,
            len(req.text)
        )
        
        return jsonify({"prediction": prediction})
    except FileNotFoundError as exc:
        logger.error("Model file not found", extra={
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'model_path': Config.MODEL_PATH
        })
        metrics_collector.record_error('model_not_found', 'webapp')
        return jsonify({"error": "Model not available"}), 503
    except (ValueError, AttributeError) as exc:
        logger.error("Model prediction error", extra={
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'model_path': Config.MODEL_PATH,
            'text_length': len(req.text) if 'req' in locals() else 0
        })
        metrics_collector.record_error('prediction_error', 'webapp')
        return jsonify({"error": "Invalid input for prediction"}), 400
    except Exception as exc:
        logger.error("Unexpected prediction error", extra={
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'model_path': Config.MODEL_PATH,
            'text_length': len(req.text) if 'req' in locals() else 0
        })
        metrics_collector.record_error('unexpected_prediction_error', 'webapp')
        return jsonify({"error": "Internal server error"}), 500


@app.route("/", methods=["GET"])
def index():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/version", methods=["GET"])
def version():
    """Return the running package version."""
    return jsonify({"version": APP_VERSION})


@app.route("/metrics", methods=["GET"])
def metrics() -> Any:
    """Return Prometheus-formatted metrics."""
    from flask import Response
    metrics_data = metrics_collector.get_prometheus_metrics()
    return Response(metrics_data, mimetype=get_metrics_content_type())


@app.route("/metrics/json", methods=["GET"])
def metrics_json() -> Any:
    """Return metrics in JSON format for dashboard."""
    dashboard_data = metrics_collector.get_dashboard_data()
    # Include legacy counters for backward compatibility
    dashboard_data['legacy'] = {
        'requests': REQUEST_COUNT, 
        'predictions': PREDICTION_COUNT
    }
    return jsonify(dashboard_data)


@app.route("/performance", methods=["GET"])
def performance() -> Any:
    """Return performance profiling data."""
    from .profiling import get_performance_report
    performance_data = get_performance_report()
    return jsonify(performance_data)


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
    
    # Initialize application metrics
    setup_app_info(APP_VERSION, {
        'host': args.host,
        'port': str(args.port),
        'log_level': args.log_level,
        'structured_logs': str(args.structured_logs)
    })
    
    # Start profiling with system monitoring
    start_profiling(enable_monitoring=True, monitor_interval=10.0)
    
    MODEL_PATH = args.model
    logger.info("Starting web server", extra={
        'host': args.host,
        'port': args.port,
        'model_path': MODEL_PATH,
        'structured_logs': args.structured_logs
    })
    
    # Add dashboard routes
    add_dashboard_routes(app)
    
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()
