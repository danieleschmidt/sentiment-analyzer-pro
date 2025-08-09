"""Advanced error handling with circuit breakers, retries, and recovery mechanisms."""

import asyncio
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better classification."""
    NETWORK = "network"
    DATABASE = "database"
    VALIDATION = "validation"
    PROCESSING = "processing"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    SYSTEM = "system"
    EXTERNAL_SERVICE = "external_service"

@dataclass
class ErrorContext:
    """Enhanced error context with metadata."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: float
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    stack_trace: Optional[str] = None
    recovery_suggestions: List[str] = None

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.success_count = 0
        self.total_requests = 0
        self._lock = threading.Lock()
    
    def _can_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _record_success(self):
        """Record successful request."""
        with self._lock:
            self.failure_count = 0
            self.success_count += 1
            self.total_requests += 1
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")
    
    def _record_failure(self):
        """Record failed request."""
        with self._lock:
            self.failure_count += 1
            self.total_requests += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logger.error(f"Circuit breaker '{self.name}' opened due to {self.failure_count} failures")
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.error(f"Circuit breaker '{self.name}' reopened due to failure during half-open state")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._can_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                else:
                    raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exception as e:
            self._record_failure()
            raise e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "total_requests": self.total_requests,
                "failure_rate": self.failure_count / max(self.total_requests, 1),
                "last_failure_time": self.last_failure_time
            }

class RetryStrategy:
    """Advanced retry strategy with exponential backoff and jitter."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    def execute(
        self,
        func: Callable,
        *args,
        retryable_exceptions: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e
                if attempt == self.max_attempts:
                    logger.error(f"All {self.max_attempts} retry attempts failed")
                    break
                
                delay = self.calculate_delay(attempt)
                logger.warning(f"Attempt {attempt} failed, retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Non-retryable exception occurred: {e}")
                raise e
        
        raise last_exception

class ErrorRecoveryManager:
    """Manages error recovery strategies and fallback mechanisms."""
    
    def __init__(self):
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def register_recovery_strategy(
        self,
        category: ErrorCategory,
        strategy: Callable
    ):
        """Register a recovery strategy for error category."""
        if category not in self.recovery_strategies:
            self.recovery_strategies[category] = []
        self.recovery_strategies[category].append(strategy)
    
    def register_fallback_handler(self, name: str, handler: Callable):
        """Register a fallback handler."""
        self.fallback_handlers[name] = handler
    
    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
        return self.circuit_breakers[name]
    
    def log_error(self, error_context: ErrorContext):
        """Log error with context."""
        with self._lock:
            self.error_history.append(error_context)
            
            # Keep only last 1000 errors
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-1000:]
        
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[error_context.severity]
        
        logger.log(
            log_level,
            f"[{error_context.error_id}] {error_context.category.value}: {error_context.message}",
            extra={
                "error_context": error_context.details,
                "user_id": error_context.user_id,
                "request_id": error_context.request_id
            }
        )
    
    def attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from error."""
        strategies = self.recovery_strategies.get(error_context.category, [])
        
        for strategy in strategies:
            try:
                if strategy(error_context):
                    logger.info(f"Recovery successful for error {error_context.error_id}")
                    return True
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
        
        return False
    
    def execute_with_fallback(
        self,
        primary_func: Callable,
        fallback_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with fallback on failure."""
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary function failed, attempting fallback: {e}")
            
            if fallback_name in self.fallback_handlers:
                try:
                    return self.fallback_handlers[fallback_name](*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise e
            else:
                logger.error(f"No fallback handler registered for '{fallback_name}'")
                raise e
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            if not self.error_history:
                return {"total_errors": 0}
            
            category_counts = {}
            severity_counts = {}
            
            for error in self.error_history:
                category = error.category.value
                severity = error.severity.value
                
                category_counts[category] = category_counts.get(category, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            return {
                "total_errors": len(self.error_history),
                "category_breakdown": category_counts,
                "severity_breakdown": severity_counts,
                "circuit_breaker_stats": {
                    name: cb.get_stats()
                    for name, cb in self.circuit_breakers.items()
                }
            }

# Global error recovery manager
_global_error_manager = ErrorRecoveryManager()

def robust_operation(
    category: ErrorCategory = ErrorCategory.PROCESSING,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    circuit_breaker: Optional[str] = None,
    retry_strategy: Optional[RetryStrategy] = None,
    fallback: Optional[str] = None
):
    """Decorator for robust operation execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            error_id = f"{func.__name__}_{int(time.time())}"
            
            try:
                # Execute with circuit breaker if specified
                if circuit_breaker:
                    cb = _global_error_manager.get_circuit_breaker(circuit_breaker)
                    if retry_strategy:
                        return retry_strategy.execute(cb.call, func, *args, **kwargs)
                    else:
                        return cb.call(func, *args, **kwargs)
                
                # Execute with retry if specified
                if retry_strategy:
                    return retry_strategy.execute(func, *args, **kwargs)
                
                # Execute with fallback if specified
                if fallback:
                    return _global_error_manager.execute_with_fallback(
                        func, fallback, *args, **kwargs
                    )
                
                # Simple execution
                return func(*args, **kwargs)
                
            except Exception as e:
                error_context = ErrorContext(
                    error_id=error_id,
                    category=category,
                    severity=severity,
                    message=str(e),
                    details={
                        "function": func.__name__,
                        "args": str(args)[:200],
                        "kwargs": str(kwargs)[:200]
                    },
                    timestamp=time.time(),
                    stack_trace=traceback.format_exc()
                )
                
                _global_error_manager.log_error(error_context)
                
                # Attempt recovery for high/critical errors
                if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                    if _global_error_manager.attempt_recovery(error_context):
                        return func(*args, **kwargs)
                
                raise e
        
        return wrapper
    return decorator

@contextmanager
def error_boundary(
    category: ErrorCategory = ErrorCategory.PROCESSING,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
):
    """Context manager for error boundary handling."""
    error_id = f"boundary_{int(time.time())}"
    
    try:
        yield
    except Exception as e:
        error_context = ErrorContext(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(e),
            details={"context": "error_boundary"},
            timestamp=time.time(),
            stack_trace=traceback.format_exc()
        )
        
        _global_error_manager.log_error(error_context)
        raise e

def get_error_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager."""
    return _global_error_manager

# Default retry strategies
DEFAULT_RETRY = RetryStrategy(max_attempts=3, base_delay=1.0)
AGGRESSIVE_RETRY = RetryStrategy(max_attempts=5, base_delay=0.5, max_delay=30.0)
CONSERVATIVE_RETRY = RetryStrategy(max_attempts=2, base_delay=2.0, max_delay=10.0)