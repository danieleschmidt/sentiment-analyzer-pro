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
from pydantic import ValidationError as PydanticValidationError

import joblib

from .config import Config
from .schemas import PredictRequest, BatchPredictRequest, ValidationError, SecurityError
from .models import SentimentModel
from .metrics import metrics, monitor_api_request, monitor_model_loading
from .logging_config import setup_logging, get_logger, log_security_event, log_api_request
from .auto_scaling import get_auto_scaler, ScalingMetrics
from .performance_optimization import (
    performance_monitor, resource_limiter, circuit_breaker, time_it,
    optimize_batch_prediction, preprocess_text_optimized
)
from .security_enhancements import (
    security_middleware, secure_endpoint, security_audit_logger,
    input_validator, compliance_manager
)
from .i18n import t, set_language, get_supported_languages
from .compliance import get_compliance_manager, DataProcessingPurpose
from .multi_region_deployment import route_request, get_load_balancer

app = Flask(__name__)
logger = get_logger(__name__)

# Initialize security middleware
security_middleware.init_app(app)

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
def load_model(path: str | None = None):
    """Load and cache a trained model from disk."""
    if path is None:
        path = Config.MODEL_PATH
    
    model = joblib.load(path)
    
    # Wrap simple sklearn pipeline in our SentimentModel for consistency
    if hasattr(model, 'predict') and not isinstance(model, SentimentModel):
        return SentimentModel(pipeline=model)
    
    return model


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
                
                # Cache the prediction for future requests
                _cache_prediction(req.text, prediction)
                
                # Get confidence if available
                confidence = None
                probabilities = None
                
                if hasattr(model, 'pipeline') and hasattr(model.pipeline, 'predict_proba') and req.return_probabilities:
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
            'prediction': prediction,
            'cached': False
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
@time_it("batch_predict_endpoint")
@secure_endpoint()
def predict_batch():
    """Handle batch prediction requests with enhanced performance."""
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
        with resource_limiter:
            start_time = time.time()
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
            
            # Load model if needed
            predictions = []
            if uncached_texts:
                try:
                    model_path = getattr(load_model, '_test_model_path', None) or Config.MODEL_PATH
                    model = circuit_breaker.call(load_model, model_path)
                except Exception as exc:
                    logger.error(f"Model loading failed for batch: {exc}")
                    return jsonify({"error": "Model not available", "code": "MODEL_ERROR"}), 503
                
                # Optimize batch prediction
                if hasattr(model, 'predict'):
                    # Preprocess all texts
                    processed_texts = [preprocess_text_optimized(text) for text in uncached_texts]
                    predictions = optimize_batch_prediction(model, processed_texts)
                    
                    # Cache new predictions
                    for text, prediction in zip(uncached_texts, predictions):
                        _cache_prediction(text, str(prediction))
            
            # Build responses with probabilities if requested
            prediction_responses = []
            all_indices = list(range(len(req.texts)))
            
            # Process cached results
            for idx, prediction in cached_results:
                pred_response = PredictionResponse(
                    prediction=str(prediction),
                    confidence=None,
                    probabilities=None,
                    processing_time_ms=0.0  # Cached results have no processing time
                )
                prediction_responses.append((idx, pred_response))
            
            # Process new predictions
            if uncached_texts and hasattr(model, 'pipeline') and hasattr(model.pipeline, 'predict_proba') and req.return_probabilities:
                try:
                    processed_texts = [preprocess_text_optimized(text) for text in uncached_texts]
                    probas = model.pipeline.predict_proba(processed_texts)
                    classes = model.pipeline.classes_
                    
                    for i, (idx, prediction, proba) in enumerate(zip(uncached_indices, predictions, probas)):
                        probabilities = {classes[j]: float(proba[j]) for j in range(len(classes))}
                        confidence = float(max(proba))
                        
                        pred_response = PredictionResponse(
                            prediction=str(prediction),
                            confidence=confidence,
                            probabilities=probabilities,
                            processing_time_ms=0.0  # Will be calculated for the batch
                        )
                        prediction_responses.append((idx, pred_response))
                        
                        # Update metrics
                        metrics.inc_prediction_counter("sklearn", str(prediction))
                except Exception as exc:
                    logger.warning(f"Could not get batch probabilities: {exc}")
                    # Fall back to predictions without probabilities
                    for idx, prediction in zip(uncached_indices, predictions):
                        pred_response = PredictionResponse(
                            prediction=str(prediction),
                            confidence=None,
                            probabilities=None,
                            processing_time_ms=0.0
                        )
                        prediction_responses.append((idx, pred_response))
                        metrics.inc_prediction_counter("sklearn", str(prediction))
            else:
                # No probabilities requested or not available
                for idx, prediction in zip(uncached_indices, predictions):
                    pred_response = PredictionResponse(
                        prediction=str(prediction),
                        confidence=None,
                        probabilities=None,
                        processing_time_ms=0.0
                    )
                    prediction_responses.append((idx, pred_response))
                    metrics.inc_prediction_counter("sklearn", str(prediction))
            
            # Sort responses by original index
            prediction_responses.sort(key=lambda x: x[0])
            sorted_predictions = [pred for _, pred in prediction_responses]
            
            total_time = time.time() - start_time
            
            global PREDICTION_COUNT
            PREDICTION_COUNT += len(req.texts)
            
            # Build response
            response = BatchPredictionResponse(
                predictions=sorted_predictions,
                total_processing_time_ms=round(total_time * 1000, 2)
            )
            
            logger.info("Batch prediction completed", extra={
                'batch_size': len(req.texts),
                'total_time_seconds': total_time,
                'cache_hits': cache_hits,
                'cache_hit_rate': cache_hits / len(req.texts) if req.texts else 0,
                'new_predictions': len(uncached_texts)
            })
            
            return jsonify(response.model_dump())
            
    except RuntimeError as exc:
        if "Too many concurrent requests" in str(exc):
            logger.warning("Batch request rejected due to resource limits")
            return jsonify({"error": "Service overloaded", "code": "RATE_LIMITED"}), 503
        raise
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
        
        # Circuit breaker status
        cb_state = circuit_breaker.get_state()
        if cb_state["state"] != "CLOSED":
            health_status["status"] = "degraded"
        health_status["circuit_breaker"] = cb_state
        
        # Resource usage
        resource_stats = resource_limiter.get_stats()
        if resource_stats["available_slots"] < 10:  # Less than 10 slots available
            health_status["status"] = "degraded"
        health_status["resources"] = resource_stats
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": time.time()
        }), 503


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


@app.route('/i18n/languages')
def get_languages():
    """Get supported languages."""
    return jsonify({
        "supported_languages": get_supported_languages(),
        "message": t("processing")
    })


@app.route('/i18n/set/<language>')
def set_app_language(language):
    """Set application language."""
    try:
        set_language(language)
        return jsonify({
            "success": True,
            "language": language,
            "message": t("processing")
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route('/compliance/consent', methods=['POST'])
@secure_endpoint()
def record_user_consent():
    """Record user consent for data processing."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": t("invalid_input")}), 400
        
        compliance_mgr = get_compliance_manager()
        consent = compliance_mgr.record_consent(
            user_id=data.get('user_id'),
            purpose=DataProcessingPurpose(data.get('purpose', 'sentiment_analysis')),
            granted=data.get('granted', False)
        )
        
        return jsonify({
            "success": True,
            "consent_id": consent.user_id,
            "message": t("processing")
        })
    except Exception as e:
        logger.error(f"Consent recording failed: {e}")
        return jsonify({"error": t("error_occurred")}), 500


@app.route('/compliance/data/<user_id>')
@secure_endpoint()
def get_user_compliance_data(user_id):
    """Get user's data for compliance (data portability)."""
    try:
        compliance_mgr = get_compliance_manager()
        user_data = compliance_mgr.get_user_data(user_id)
        return jsonify(user_data)
    except Exception as e:
        logger.error(f"Data retrieval failed: {e}")
        return jsonify({"error": t("error_occurred")}), 500


@app.route('/regions/route', methods=['POST'])
def route_global_request():
    """Route request to optimal region."""
    try:
        data = request.get_json()
        user_location = data.get('location') if data else None
        
        routing_info = route_request(
            request_data=data or {},
            user_location=user_location
        )
        
        return jsonify({
            "routing": routing_info,
            "message": t("processing")
        })
    except Exception as e:
        logger.error(f"Request routing failed: {e}")
        return jsonify({"error": t("error_occurred")}), 500


@app.route('/regions/stats')
def get_region_stats():
    """Get global region statistics."""
    try:
        load_balancer = get_load_balancer()
        stats = load_balancer.get_load_balancer_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        return jsonify({"error": t("error_occurred")}), 500


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
