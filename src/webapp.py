"""Flask web server for sentiment predictions."""
from __future__ import annotations

import argparse
import time
from functools import lru_cache
from typing import Any
from collections import defaultdict, deque

from flask import Flask, jsonify, request
from importlib import metadata
from pydantic import ValidationError as PydanticValidationError

import joblib

from .config import Config
from .schemas import PredictRequest, ValidationError, SecurityError
from .models import SentimentModel
from .metrics import metrics, monitor_api_request, monitor_model_loading
from .logging_config import setup_logging, get_logger, log_security_event, log_api_request
from .performance_optimization import (
    performance_monitor, resource_limiter, circuit_breaker, time_it,
    optimize_batch_prediction, preprocess_text_optimized
)
from .security_enhancements import (
    security_middleware, secure_endpoint, security_audit_logger,
    input_validator, compliance_manager
)

app = Flask(__name__)
logger = get_logger(__name__)

# Initialize security middleware
security_middleware.init_app(app)

# Rate limiting - simple in-memory store
RATE_LIMIT_REQUESTS = defaultdict(lambda: deque())

REQUEST_COUNT = 0
PREDICTION_COUNT = 0

try:
    APP_VERSION = metadata.version("sentiment-analyzer-pro")
except metadata.PackageNotFoundError:  # pragma: no cover - local usage
    APP_VERSION = "0.0.0"


@lru_cache(maxsize=1)
@monitor_model_loading("sklearn")
def load_model(path: str | None = None):
    """Load and cache a trained model from disk."""
    if path is None:
        path = Config.MODEL_PATH
    
    model = joblib.load(path)
    
    # Wrap simple sklearn pipeline in our SentimentModel for consistency
    if hasattr(model, 'predict') and not isinstance(model, SentimentModel):
        return SentimentModel(pipeline=model)
    
    return model


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
        return jsonify({"error": "Rate limit exceeded"}), 429


@app.after_request
def _add_security_headers(response):
    """Add security headers and log API requests."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    
    # Log API request with duration
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
    
    return response


@app.errorhandler(ValidationError)
def handle_validation_error(error):
    """Handle validation errors with structured response."""
    from .schemas import ErrorResponse, ValidationError
    response = ErrorResponse(
        error=error.message,
        error_code=error.error_code,
        details=error.details
    )
    return jsonify(response.model_dump()), 400


@app.errorhandler(SecurityError) 
def handle_security_error(error):
    """Handle security errors with logging."""
    from .schemas import ErrorResponse, SecurityError
    log_security_event(
        logger,
        'security_violation',
        error_message=error.message,
        details=error.details
    )
    response = ErrorResponse(
        error="Security violation detected",
        error_code=error.error_code,
        details={"message": "Request blocked for security reasons"}
    )
    return jsonify(response.model_dump()), 403


@app.errorhandler(Exception)
def handle_general_error(error):
    """Handle unexpected errors gracefully."""
    from .schemas import ErrorResponse
    logger.error(f"Unexpected error: {str(error)}", exc_info=True)
    response = ErrorResponse(
        error="Internal server error",
        error_code="INTERNAL_ERROR",
        details={"type": type(error).__name__}
    )
    return jsonify(response.model_dump()), 500


@app.route("/predict", methods=["POST"])
@monitor_api_request("POST", "/predict")
@time_it("predict_endpoint")
@secure_endpoint()
def predict():
    # Validate Content-Type
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json(silent=True) or {}
    try:
        req = PredictRequest(**data)
    except PydanticValidationError as exc:
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
    
    # Use resource limiter to prevent overload
    try:
        with resource_limiter:
            start_time = time.time()
            
            # Load model with circuit breaker protection
            try:
                model_path = getattr(load_model, '_test_model_path', None) or Config.MODEL_PATH
                model = circuit_breaker.call(load_model, model_path)
            except FileNotFoundError:
                logger.error(f"Model file not found: {model_path}")
                return jsonify({"error": "Model not available", "code": "MODEL_NOT_FOUND"}), 503
            except RuntimeError as exc:
                if "Circuit breaker" in str(exc):
                    logger.error("Circuit breaker is open - model unavailable")
                    return jsonify({"error": "Service temporarily unavailable", "code": "CIRCUIT_BREAKER_OPEN"}), 503
                raise
            except Exception as exc:
                logger.error(f"Model loading failed: {exc}")
                return jsonify({"error": "Model loading failed", "code": "MODEL_ERROR"}), 503
        
            # Make prediction with validation and optimization
            try:
                if not hasattr(model, 'predict'):
                    raise AttributeError("Model does not have predict method")
                
                # Preprocess text for better performance
                processed_text = preprocess_text_optimized(req.text)
                prediction = model.predict([processed_text])
                # Handle both single result and list results
                if isinstance(prediction, list):
                    prediction = prediction[0]
                
                # Validate prediction output
                if not isinstance(prediction, str):
                    prediction = str(prediction)
                
                # Get confidence if available
                confidence = None
                probabilities = None
                
                if hasattr(model.pipeline, 'predict_proba') and req.return_probabilities:
                    try:
                        proba = model.pipeline.predict_proba([processed_text])[0]
                        classes = model.pipeline.classes_
                        probabilities = {classes[i]: float(proba[i]) for i in range(len(classes))}
                        confidence = float(max(proba))
                    except Exception as exc:
                        logger.warning(f"Could not get probabilities: {exc}")
                
            except (ValueError, AttributeError, IndexError) as exc:
                logger.error(f"Prediction failed: {exc}")
                return jsonify({"error": "Prediction failed", "code": "PREDICTION_ERROR", "details": str(exc)}), 500
            except Exception as exc:
                logger.error(f"Unexpected prediction error: {exc}")
                return jsonify({"error": "Internal prediction error", "code": "INTERNAL_ERROR"}), 500
            
            # Record prediction in metrics
            prediction_time = time.time() - start_time
            metrics.inc_prediction_counter("sklearn", prediction)
        
        global PREDICTION_COUNT
        PREDICTION_COUNT += 1
        
        # Log prediction performance
        logger.info("Prediction completed", extra={
            'event_type': 'prediction',
            'prediction_time_seconds': prediction_time,
            'text_length': len(req.text),
            'prediction': prediction
        })
        
        # Build response with optional fields
        response_data = {
            "prediction": prediction,
            "processing_time_ms": round(prediction_time * 1000, 2)
        }
        
        if confidence is not None:
            response_data["confidence"] = confidence
        
        if probabilities is not None:
            response_data["probabilities"] = probabilities
        
            return jsonify(response_data)
            
    except RuntimeError as exc:
        if "Too many concurrent requests" in str(exc):
            logger.warning("Request rejected due to resource limits")
            return jsonify({"error": "Service overloaded", "code": "RATE_LIMITED"}), 503
        raise
    except Exception as exc:
        logger.error("Unexpected error in predict endpoint", extra={
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'text_length': len(req.text) if 'req' in locals() else 0
        })
        return jsonify({"error": "Internal server error", "code": "INTERNAL_ERROR"}), 500


@app.route("/predict/batch", methods=["POST"])
@monitor_api_request("POST", "/predict/batch")
@secure_endpoint()
def predict_batch():
    """Handle batch prediction requests."""
    from .schemas import BatchPredictRequest, BatchPredictionResponse, PredictionResponse
    
    # Validate Content-Type
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json(silent=True) or {}
    try:
        req = BatchPredictRequest(**data)
    except PydanticValidationError as exc:
        logger.warning("Batch validation error", extra={'errors': exc.errors()})
        return jsonify({"error": "Invalid input", "details": exc.errors()}), 400
    
    try:
        start_time = time.time()
        
        # Load model
        try:
            model_path = getattr(load_model, '_test_model_path', None) or Config.MODEL_PATH
            model = load_model(model_path)
        except Exception as exc:
            logger.error(f"Model loading failed for batch: {exc}")
            return jsonify({"error": "Model not available", "code": "MODEL_ERROR"}), 503
        
        # Process batch predictions
        predictions = []
        for i, text in enumerate(req.texts):
            try:
                pred_start = time.time()
                prediction = model.predict([text])[0]
                pred_time = time.time() - pred_start
                
                # Get probabilities if requested
                confidence = None
                probabilities = None
                
                if hasattr(model.pipeline, 'predict_proba') and req.return_probabilities:
                    try:
                        proba = model.pipeline.predict_proba([text])[0]
                        classes = model.pipeline.classes_
                        probabilities = {classes[j]: float(proba[j]) for j in range(len(classes))}
                        confidence = float(max(proba))
                    except Exception:
                        pass  # Skip probabilities if error
                
                pred_response = PredictionResponse(
                    prediction=str(prediction),
                    confidence=confidence,
                    probabilities=probabilities,
                    processing_time_ms=round(pred_time * 1000, 2)
                )
                predictions.append(pred_response)
                
                # Update metrics
                metrics.inc_prediction_counter("sklearn", str(prediction))
                
            except Exception as exc:
                logger.error(f"Failed to predict text {i}: {exc}")
                pred_response = PredictionResponse(
                    prediction="error",
                    confidence=None,
                    probabilities=None,
                    processing_time_ms=0.0
                )
                predictions.append(pred_response)
        
        total_time = time.time() - start_time
        
        global PREDICTION_COUNT
        PREDICTION_COUNT += len(predictions)
        
        # Build response
        response = BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=round(total_time * 1000, 2)
        )
        
        logger.info("Batch prediction completed", extra={
            'batch_size': len(req.texts),
            'total_time_seconds': total_time,
            'successful_predictions': len([p for p in predictions if p.prediction != "error"])
        })
        
        return jsonify(response.model_dump())
        
    except Exception as exc:
        logger.error(f"Unexpected batch prediction error: {exc}")
        return jsonify({"error": "Internal server error", "code": "BATCH_ERROR"}), 500


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
    """Return comprehensive service metrics summary."""
    base_metrics = {"requests": REQUEST_COUNT, "predictions": PREDICTION_COUNT}
    enhanced_metrics = metrics.get_summary()
    
    # Get performance metrics
    perf_metrics = performance_monitor.get_all_stats()
    
    # Get resource metrics
    resource_metrics = resource_limiter.get_stats()
    
    # Get circuit breaker state
    circuit_state = circuit_breaker.get_state()
    
    # Get model cache stats if available
    try:
        model = load_model()
        cache_stats = model.get_cache_stats()
    except:
        cache_stats = {}
    
    return jsonify({
        **base_metrics, 
        **enhanced_metrics,
        "performance": perf_metrics,
        "resources": resource_metrics,
        "circuit_breaker": circuit_state,
        "cache": cache_stats
    })


@app.route("/health/detailed", methods=["GET"])
def health_detailed():
    """Detailed health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": APP_VERSION
    }
    
    # Check model availability
    try:
        model = load_model()
        health_status["model"] = "available"
        health_status["model_cache_stats"] = model.get_cache_stats()
    except Exception as e:
        health_status["model"] = f"unavailable: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check circuit breaker
    cb_state = circuit_breaker.get_state()
    if cb_state["state"] != "CLOSED":
        health_status["status"] = "degraded"
        health_status["circuit_breaker"] = cb_state
    
    # Check resource usage
    resource_stats = resource_limiter.get_stats()
    if resource_stats["available_slots"] < 10:  # Less than 10 slots available
        health_status["status"] = "degraded"
    health_status["resources"] = resource_stats
    
    return jsonify(health_status)


@app.route("/security/audit", methods=["GET"])
@secure_endpoint(require_auth=True, permission="admin")
def security_audit():
    """Get security audit information (admin only)."""
    return jsonify({
        "audit_status": "enabled",
        "security_features": [
            "input_validation",
            "rate_limiting", 
            "security_headers",
            "audit_logging",
            "content_type_validation",
            "suspicious_activity_detection"
        ],
        "compliance": {
            "gdpr_ready": True,
            "ccpa_ready": True,
            "audit_logging": True
        }
    })


@app.route("/privacy/data-export", methods=["POST"])
@secure_endpoint(require_auth=True)
def request_data_export():
    """Handle GDPR data export requests."""
    user_id = getattr(request, 'user_id', 'anonymous')
    
    result = compliance_manager.generate_data_export(user_id)
    
    return jsonify(result)


@app.route("/privacy/data-deletion", methods=["POST"])
@secure_endpoint(require_auth=True)
def request_data_deletion():
    """Handle GDPR data deletion requests."""
    user_id = getattr(request, 'user_id', 'anonymous')
    
    result = compliance_manager.handle_data_deletion_request(user_id)
    
    return jsonify(result)


@app.route("/security/validate", methods=["POST"])
def validate_input():
    """Endpoint to validate input without processing."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    
    if 'text' in data:
        validation = input_validator.validate_text_input(data['text'])
        return jsonify({
            "is_valid": validation['is_valid'],
            "warnings": validation['warnings'],
            "sanitized_length": len(validation['sanitized_text'])
        })
    
    elif 'texts' in data:
        validation = input_validator.validate_batch_input(data['texts'])
        return jsonify({
            "is_valid": validation['is_valid'],
            "warnings": validation['warnings'],
            "invalid_indices": validation['invalid_indices']
        })
    
    else:
        return jsonify({"error": "No text or texts field provided"}), 400


def main(argv: list[str] | None = None) -> None:
    global MODEL_PATH
    parser = argparse.ArgumentParser(description="Run prediction web server")
    parser.add_argument("--model", default=Config.MODEL_PATH, help="Trained model path")
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
