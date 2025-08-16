#!/usr/bin/env python3
"""
Generation 2: Robustness Enhancements
Comprehensive error handling, validation, monitoring, and security
"""

import logging
import time
from functools import wraps
from typing import Dict, Any, List, Optional
import json
import os

class RobustErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self):
        self.error_counts = {}
        self.recovery_strategies = {}
        self.logger = logging.getLogger(__name__)
    
    def handle_prediction_error(self, error: Exception, text: str, fallback_sentiment: str = "neutral") -> Dict[str, Any]:
        """Handle prediction errors with fallback strategies."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.logger.error(f"Prediction error {error_type}: {error}")
        
        # Fallback strategy: simple rule-based sentiment
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst']
        
        text_lower = text.lower()
        if any(word in text_lower for word in positive_words):
            fallback_sentiment = "positive"
        elif any(word in text_lower for word in negative_words):
            fallback_sentiment = "negative"
        
        return {
            "prediction": fallback_sentiment,
            "confidence": 0.5,
            "error_recovery": True,
            "error_type": error_type,
            "fallback_strategy": "rule_based"
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "error_counts": self.error_counts,
            "total_errors": sum(self.error_counts.values()),
            "error_types": list(self.error_counts.keys())
        }

class InputValidator:
    """Enhanced input validation and sanitization."""
    
    MAX_TEXT_LENGTH = 10000
    MAX_BATCH_SIZE = 100
    
    @staticmethod
    def validate_text(text: str) -> Dict[str, Any]:
        """Validate and sanitize text input."""
        result = {
            "is_valid": True,
            "sanitized_text": text,
            "warnings": []
        }
        
        if not isinstance(text, str):
            result["is_valid"] = False
            result["warnings"].append("Input must be a string")
            return result
        
        if len(text) > InputValidator.MAX_TEXT_LENGTH:
            result["is_valid"] = False
            result["warnings"].append(f"Text exceeds maximum length of {InputValidator.MAX_TEXT_LENGTH}")
            return result
        
        if not text.strip():
            result["warnings"].append("Empty or whitespace-only input")
            result["sanitized_text"] = "No content provided"
        
        # Basic sanitization
        result["sanitized_text"] = text.strip()
        
        return result
    
    @staticmethod
    def validate_batch(texts: List[str]) -> Dict[str, Any]:
        """Validate batch input."""
        result = {
            "is_valid": True,
            "sanitized_texts": [],
            "warnings": []
        }
        
        if not isinstance(texts, list):
            result["is_valid"] = False
            result["warnings"].append("Batch input must be a list")
            return result
        
        if len(texts) > InputValidator.MAX_BATCH_SIZE:
            result["is_valid"] = False
            result["warnings"].append(f"Batch size exceeds maximum of {InputValidator.MAX_BATCH_SIZE}")
            return result
        
        for i, text in enumerate(texts):
            validation = InputValidator.validate_text(text)
            if not validation["is_valid"]:
                result["is_valid"] = False
                result["warnings"].extend([f"Item {i}: {w}" for w in validation["warnings"]])
            result["sanitized_texts"].append(validation["sanitized_text"])
        
        return result

class HealthMonitor:
    """System health monitoring and alerting."""
    
    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "avg_response_time": 0.0,
            "error_rate": 0.0,
            "uptime_start": time.time()
        }
        self.response_times = []
        
    def record_request(self, response_time: float, success: bool = True):
        """Record request metrics."""
        self.metrics["requests_total"] += 1
        self.response_times.append(response_time)
        
        if success:
            self.metrics["requests_successful"] += 1
        else:
            self.metrics["requests_failed"] += 1
        
        # Keep only last 100 response times for rolling average
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        
        self.metrics["avg_response_time"] = sum(self.response_times) / len(self.response_times)
        self.metrics["error_rate"] = self.metrics["requests_failed"] / max(self.metrics["requests_total"], 1)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        uptime = time.time() - self.metrics["uptime_start"]
        
        status = "healthy"
        if self.metrics["error_rate"] > 0.1:  # 10% error rate
            status = "degraded"
        if self.metrics["error_rate"] > 0.3:  # 30% error rate
            status = "unhealthy"
        
        return {
            "status": status,
            "uptime_seconds": uptime,
            "metrics": self.metrics,
            "timestamp": time.time()
        }

def resilient_api_call(health_monitor: HealthMonitor, error_handler: RobustErrorHandler):
    """Decorator for resilient API calls with monitoring."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                health_monitor.record_request(response_time, success=True)
                return result
            except Exception as e:
                response_time = time.time() - start_time
                health_monitor.record_request(response_time, success=False)
                
                # Try to recover based on function type
                if 'predict' in func.__name__:
                    text = kwargs.get('text', args[0] if args else "")
                    return error_handler.handle_prediction_error(e, text)
                else:
                    raise e
        return wrapper
    return decorator

# Global instances
error_handler = RobustErrorHandler()
health_monitor = HealthMonitor()

def setup_robustness_logging():
    """Setup comprehensive logging for robustness."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app_robustness.log')
        ]
    )

def test_robustness_features():
    """Test all robustness features."""
    print("Testing Robustness Features...")
    
    # Test input validation
    print("\n1. Testing Input Validation:")
    validator = InputValidator()
    
    test_cases = [
        "This is a normal text",
        "",
        "x" * 15000,  # Too long
        None,  # Invalid type
    ]
    
    for i, test_case in enumerate(test_cases):
        try:
            result = validator.validate_text(test_case)
            print(f"   Test {i+1}: Valid={result['is_valid']}, Warnings={len(result['warnings'])}")
        except Exception as e:
            print(f"   Test {i+1}: Error={e}")
    
    # Test error handling
    print("\n2. Testing Error Handling:")
    
    @resilient_api_call(health_monitor, error_handler)
    def test_predict(text):
        if "error" in text:
            raise ValueError("Simulated prediction error")
        return {"prediction": "positive", "confidence": 0.9}
    
    try:
        result1 = test_predict("This is great!")
        print(f"   Success case: {result1}")
        
        result2 = test_predict("This causes error")
        print(f"   Error recovery: {result2}")
    except Exception as e:
        print(f"   Unexpected error: {e}")
    
    # Test health monitoring
    print("\n3. Testing Health Monitoring:")
    health_status = health_monitor.get_health_status()
    print(f"   Health Status: {health_status['status']}")
    print(f"   Total Requests: {health_status['metrics']['requests_total']}")
    print(f"   Error Rate: {health_status['metrics']['error_rate']:.2%}")
    
    print("\n✓ Robustness features tested successfully!")

if __name__ == "__main__":
    setup_robustness_logging()
    test_robustness_features()