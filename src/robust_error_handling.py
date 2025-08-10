"""
Robust error handling and logging system
Generation 2: Make It Robust - Comprehensive error handling
"""
import logging
import traceback
import time
import json
from typing import Any, Dict, Optional, Callable, Union
from functools import wraps
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    VALIDATION = "validation"
    PROCESSING = "processing"
    MODEL = "model"
    DATA = "data"
    SYSTEM = "system"
    SECURITY = "security"
    EXTERNAL = "external"

@dataclass
class ErrorContext:
    """Context information for errors"""
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: Dict[str, Any]
    traceback: Optional[str]
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "details": self.details,
            "traceback": self.traceback,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "session_id": self.session_id
        }

class RobustLogger:
    """Enhanced logging with error tracking and metrics"""
    
    def __init__(self, name: str = "sentiment_analyzer", 
                 log_file: Optional[str] = None,
                 level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
        
        # Error tracking
        self.error_log = []
        self.error_counts = {
            ErrorSeverity.LOW: 0,
            ErrorSeverity.MEDIUM: 0,
            ErrorSeverity.HIGH: 0,
            ErrorSeverity.CRITICAL: 0
        }
    
    def log_error(self, error_context: ErrorContext):
        """Log an error with context"""
        # Update counters
        self.error_counts[error_context.severity] += 1
        
        # Store error for analysis
        self.error_log.append(error_context)
        
        # Keep only last 1000 errors to prevent memory issues
        if len(self.error_log) > 1000:
            self.error_log = self.error_log[-1000:]
        
        # Log based on severity
        log_message = f"[{error_context.category.value}] {error_context.message}"
        if error_context.details:
            log_message += f" | Details: {json.dumps(error_context.details, default=str)}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            if error_context.traceback:
                self.logger.critical(f"Traceback: {error_context.traceback}")
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "by_severity": {k.value: v for k, v in self.error_counts.items()},
            "recent_errors": [err.to_dict() for err in self.error_log[-10:]],
            "categories": self._get_category_breakdown()
        }
    
    def _get_category_breakdown(self) -> Dict[str, int]:
        """Get error breakdown by category"""
        category_counts = {}
        for error in self.error_log:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts

# Global logger instance
_global_logger: Optional[RobustLogger] = None

def get_logger(name: str = "sentiment_analyzer") -> RobustLogger:
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = RobustLogger(name)
    return _global_logger

class RobustErrorHandler:
    """Centralized error handling with retry logic"""
    
    def __init__(self, logger: Optional[RobustLogger] = None):
        self.logger = logger or get_logger()
        self.retry_attempts = {}
        self.max_retries = 3
        self.backoff_factor = 2.0
    
    def handle_error(self, 
                    error: Exception,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.PROCESSING,
                    context: Optional[Dict[str, Any]] = None,
                    user_id: Optional[str] = None,
                    request_id: Optional[str] = None) -> ErrorContext:
        """Handle an error with context"""
        
        error_context = ErrorContext(
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=str(error),
            details=context or {},
            traceback=traceback.format_exc(),
            user_id=user_id,
            request_id=request_id
        )
        
        self.logger.log_error(error_context)
        return error_context
    
    def retry_with_backoff(self,
                          func: Callable,
                          max_retries: int = 3,
                          backoff_factor: float = 2.0,
                          exceptions: tuple = (Exception,),
                          context: Optional[Dict[str, Any]] = None):
        """Retry function with exponential backoff"""
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                if attempt == max_retries:
                    # Final attempt failed
                    self.handle_error(
                        e,
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.PROCESSING,
                        context={
                            **(context or {}),
                            "retry_attempts": attempt + 1,
                            "max_retries": max_retries
                        }
                    )
                    raise
                
                # Wait before retry with exponential backoff
                wait_time = backoff_factor ** attempt
                time.sleep(wait_time)
                
                self.logger.logger.warning(
                    f"Retry attempt {attempt + 1}/{max_retries} after {wait_time}s delay"
                )

def robust_function(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                   category: ErrorCategory = ErrorCategory.PROCESSING,
                   max_retries: int = 0,
                   exceptions: tuple = (Exception,)):
    """Decorator for robust error handling"""
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = RobustErrorHandler()
            
            def execute():
                return func(*args, **kwargs)
            
            try:
                if max_retries > 0:
                    return error_handler.retry_with_backoff(
                        execute,
                        max_retries=max_retries,
                        exceptions=exceptions,
                        context={
                            "function": func.__name__,
                            "args": str(args)[:100],
                            "kwargs": str(kwargs)[:100]
                        }
                    )
                else:
                    return execute()
                    
            except Exception as e:
                error_handler.handle_error(
                    e,
                    severity=severity,
                    category=category,
                    context={
                        "function": func.__name__,
                        "args": str(args)[:100],
                        "kwargs": str(kwargs)[:100]
                    }
                )
                raise
        
        return wrapper
    return decorator

class CircuitBreaker:
    """Circuit breaker pattern for external dependencies"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                self.failure_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            # Success resets the circuit
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
            self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise

def setup_robust_logging(log_file: str = "sentiment_analyzer.log", 
                        level: str = "INFO") -> RobustLogger:
    """Setup robust logging system"""
    
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = RobustLogger(log_file=log_file, level=level)
    
    # Set as global logger
    global _global_logger
    _global_logger = logger
    
    return logger

# Input validation decorators
def validate_input(validation_func: Callable[[Any], bool], 
                  error_message: str = "Invalid input"):
    """Decorator for input validation"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate all arguments
            all_args = list(args) + list(kwargs.values())
            for arg in all_args:
                if not validation_func(arg):
                    error_handler = RobustErrorHandler()
                    error_handler.handle_error(
                        ValueError(error_message),
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.VALIDATION,
                        context={
                            "function": func.__name__,
                            "invalid_input": str(arg)[:100]
                        }
                    )
                    raise ValueError(error_message)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_text_input(text: Any) -> bool:
    """Validate text input for sentiment analysis"""
    return isinstance(text, str) and len(text.strip()) > 0 and len(text) <= 10000

if __name__ == "__main__":
    # Test the robust error handling system
    logger = setup_robust_logging()
    
    @robust_function(severity=ErrorSeverity.HIGH, max_retries=2)
    @validate_input(validate_text_input, "Text must be non-empty string")
    def test_function(text: str):
        if "error" in text.lower():
            raise ValueError("Test error")
        return f"Processed: {text}"
    
    # Test successful case
    result = test_function("Hello world")
    print(f"Success: {result}")
    
    # Test error case
    try:
        test_function("")
    except Exception as e:
        print(f"Expected error: {e}")
    
    # Print error summary
    print("\nError Summary:")
    summary = logger.get_error_summary()
    print(json.dumps(summary, indent=2, default=str))