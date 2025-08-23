
import time
import functools
import threading
from typing import Callable, Any, Dict, Optional, Type
from enum import Enum
import random

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise Exception("Circuit breaker is OPEN")
                else:
                    self.state = CircuitState.HALF_OPEN
            
            try:
                result = func(*args, **kwargs)
                self.on_success()
                return result
            except Exception as e:
                self.on_failure()
                raise
    
    def on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, 
                      backoff_multiplier: float = 2.0, exceptions: tuple = (Exception,)):
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    delay = base_delay * (backoff_multiplier ** attempt)
                    # Add jitter
                    jitter = delay * 0.1 * random.random()
                    time.sleep(delay + jitter)
            
            raise last_exception
        return wrapper
    return decorator

class RobustErrorHandler:
    """Advanced error handling with recovery strategies."""
    
    def __init__(self):
        self.error_stats = {}
        self.circuit_breakers = {}
        self._lock = threading.Lock()
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker()
        return self.circuit_breakers[name]
    
    def record_error(self, operation: str, error: Exception, context: Dict[str, Any] = None):
        """Record error with context for analysis."""
        with self._lock:
            if operation not in self.error_stats:
                self.error_stats[operation] = {
                    "count": 0,
                    "last_error": None,
                    "error_types": {},
                    "contexts": []
                }
            
            stats = self.error_stats[operation]
            stats["count"] += 1
            stats["last_error"] = {
                "message": str(error),
                "type": type(error).__name__,
                "timestamp": time.time()
            }
            
            error_type = type(error).__name__
            stats["error_types"][error_type] = stats["error_types"].get(error_type, 0) + 1
            
            if context:
                stats["contexts"].append({
                    "context": context,
                    "timestamp": time.time()
                })
                # Keep only last 10 contexts
                stats["contexts"] = stats["contexts"][-10:]

# Global error handler
robust_error_handler = RobustErrorHandler()
