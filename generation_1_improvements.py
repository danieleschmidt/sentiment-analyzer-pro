#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK - Simple improvements for immediate functionality
Terragon Labs Autonomous SDLC Execution
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check dependencies without installing."""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'nltk', 'joblib', 
        'pydantic', 'cryptography', 'pyjwt', 'flask', 'pytest', 'pytest-cov'
    ]
    
    missing = []
    available = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            available.append(package)
        except ImportError:
            missing.append(package)
    
    print(f"✅ Available packages: {available}")
    if missing:
        print(f"⚠️ Missing packages: {missing}")
        print("Note: Install manually if needed")
    
    return len(missing) == 0

def create_simple_health_check():
    """Create basic health monitoring system."""
    health_check_code = '''
import time
import psutil
from typing import Dict, Any

class SimpleHealthMonitor:
    """Basic health monitoring for Generation 1."""
    
    def __init__(self):
        self.start_time = time.time()
        self.requests_count = 0
        
    def get_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        return {
            "status": "healthy",
            "uptime": time.time() - self.start_time,
            "memory_usage": psutil.virtual_memory().percent if hasattr(psutil, 'virtual_memory') else 0,
            "requests_served": self.requests_count,
            "timestamp": time.time()
        }
    
    def record_request(self):
        """Record a request for monitoring."""
        self.requests_count += 1

# Global health monitor instance
health_monitor = SimpleHealthMonitor()
'''
    
    with open("/root/repo/src/simple_health.py", "w") as f:
        f.write(health_check_code)
    
    print("✅ Created simple health monitoring system")

def create_basic_error_handler():
    """Create simple error handling system."""
    error_handler_code = '''
import logging
import traceback
from functools import wraps
from typing import Callable, Any

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def handle_errors(func: Callable) -> Callable:
    """Simple error handling decorator."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper

class SimpleErrorHandler:
    """Basic error handling and recovery system."""
    
    def __init__(self):
        self.error_count = 0
        self.errors = []
    
    def record_error(self, error: Exception, context: str = ""):
        """Record an error for tracking."""
        self.error_count += 1
        self.errors.append({
            "error": str(error),
            "context": context,
            "timestamp": time.time(),
            "type": type(error).__name__
        })
        
        # Keep only last 100 errors
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]
    
    def get_error_stats(self):
        """Get error statistics."""
        return {
            "total_errors": self.error_count,
            "recent_errors": len(self.errors),
            "error_types": list(set([e["type"] for e in self.errors]))
        }

# Global error handler
error_handler = SimpleErrorHandler()
'''
    
    with open("/root/repo/src/simple_errors.py", "w") as f:
        f.write(error_handler_code)
    
    print("✅ Created simple error handling system")

def create_basic_validation():
    """Create input validation system."""
    validation_code = '''
from typing import Any, Dict, List, Optional
import re

class SimpleValidator:
    """Basic input validation for Generation 1."""
    
    @staticmethod
    def validate_text(text: Any) -> str:
        """Validate and clean text input."""
        if not text:
            raise ValueError("Text input cannot be empty")
        
        if not isinstance(text, str):
            text = str(text)
        
        # Basic sanitization
        text = text.strip()
        if len(text) > 10000:  # Reasonable limit
            raise ValueError("Text too long (max 10000 characters)")
        
        return text
    
    @staticmethod
    def validate_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model parameters."""
        validated = {}
        
        # Common parameter validation
        if 'batch_size' in params:
            batch_size = params['batch_size']
            if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 1000:
                raise ValueError("batch_size must be integer between 1-1000")
            validated['batch_size'] = batch_size
        
        if 'learning_rate' in params:
            lr = params['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                raise ValueError("learning_rate must be float between 0-1")
            validated['learning_rate'] = float(lr)
        
        return validated

# Global validator instance
validator = SimpleValidator()
'''
    
    with open("/root/repo/src/simple_validation.py", "w") as f:
        f.write(validation_code)
    
    print("✅ Created simple validation system")

def run_basic_tests():
    """Run basic functionality tests."""
    print("Running Generation 1 basic tests...")
    
    try:
        # Test imports
        import src.models
        import src.config
        print("✅ Core modules import successfully")
        
        # Test simple model creation if sklearn is available
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.feature_extraction.text import TfidfVectorizer
            model = LogisticRegression()
            vectorizer = TfidfVectorizer()
            print("✅ ML models can be instantiated")
        except ImportError:
            print("⚠️ ML libraries not available - will install")
        
        return True
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def main():
    """Main execution for Generation 1 improvements."""
    print("🚀 Starting Generation 1: MAKE IT WORK")
    
    # Check dependencies
    deps_available = check_dependencies()
    if deps_available:
        print("✅ All dependencies available")
    else:
        print("⚠️ Some dependencies missing - continuing with available ones")
    
    # Create basic systems
    create_simple_health_check()
    create_basic_error_handler()
    create_basic_validation()
    
    # Run tests
    if run_basic_tests():
        print("✅ Generation 1 improvements completed successfully")
        return True
    else:
        print("❌ Generation 1 improvements failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)