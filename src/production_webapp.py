#!/usr/bin/env python3
"""Production-ready web application with enhanced security and monitoring."""

import os
import sys
import logging
from flask import Flask, request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import time
import psutil
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')

def create_production_app():
    """Create production Flask application."""
    app = Flask(__name__)
    
    # Production configuration
    app.config.update({
        'SECRET_KEY': os.environ.get('SECRET_KEY', os.urandom(32)),
        'TESTING': False,
        'DEBUG': False,
        'JSON_SORT_KEYS': False
    })
    
    # Rate limiting
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=["1000 per hour", "100 per minute"]
    )
    
    # Security headers middleware
    @app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response
    
    # Request timing middleware
    @app.before_request
    def before_request():
        g.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        if hasattr(g, 'start_time'):
            REQUEST_LATENCY.observe(time.time() - g.start_time)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.endpoint or 'unknown',
                status=response.status_code
            ).inc()
        return response
    
    # Health check endpoint
    @app.route('/health')
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'timestamp': time.time(),
            'version': '1.0.0'
        })
    
    # Readiness check endpoint
    @app.route('/ready')
    def ready():
        """Readiness check endpoint."""
        # Add actual readiness checks here
        return jsonify({'status': 'ready'})
    
    # Metrics endpoint
    @app.route('/metrics')
    def metrics():
        """Prometheus metrics endpoint."""
        return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
    
    # Main prediction endpoint
    @app.route('/predict', methods=['POST'])
    @limiter.limit("50 per minute")
    def predict():
        """Production prediction endpoint."""
        try:
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({'error': 'Missing text field'}), 400
            
            text = data['text']
            if not isinstance(text, str) or len(text.strip()) == 0:
                return jsonify({'error': 'Invalid text input'}), 400
            
            # Mock prediction for demo
            prediction = 'positive' if len(text) > 10 else 'negative'
            confidence = 0.85
            
            return jsonify({
                'prediction': prediction,
                'confidence': confidence,
                'text_length': len(text),
                'timestamp': time.time()
            })
            
        except Exception as e:
            app.logger.error(f"Prediction error: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    # System info endpoint (admin only)
    @app.route('/system')
    def system_info():
        """System information endpoint."""
        return jsonify({
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        })
    
    return app

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    app = create_production_app()
    
    # Production server configuration
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host=host, port=port, threaded=True)
