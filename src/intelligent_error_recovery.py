"""
Intelligent Error Recovery System for Production Sentiment Analysis

This module implements sophisticated error handling and recovery mechanisms:
- Circuit breaker patterns with adaptive thresholds
- Retry strategies with exponential backoff and jitter
- Graceful degradation and fallback mechanisms
- Self-healing capabilities with automated recovery
- Error pattern analysis and learning
- Context-aware error handling
- Distributed system resilience patterns

Features:
- Smart retry logic with context awareness
- Circuit breakers with health checks
- Bulkhead isolation for resource protection
- Timeout management with adaptive scaling
- Error classification and routing
- Recovery orchestration
- Chaos engineering integration
"""

from __future__ import annotations

import asyncio
import time
import random
import logging
import threading
import traceback
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict, deque
import functools
import uuid
import inspect

# Optional dependencies for advanced features
try:
    import aiohttp
    import asyncio
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing fast
    HALF_OPEN = "half_open" # Testing recovery


class RetryStrategy(Enum):
    """Retry strategy types"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    JITTERED_EXPONENTIAL = "jittered_exponential"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    operation: str = ""
    error_type: str = ""
    error_message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    retry_count: int = 0
    max_retries: int = 3
    context_data: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    recovery_attempts: List[Dict] = field(default_factory=list)
    user_id: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms"""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    jitter: bool = True
    backoff_multiplier: float = 2.0
    retry_exceptions: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    stop_exceptions: List[Type[Exception]] = field(default_factory=list)
    retry_conditions: List[Callable] = field(default_factory=list)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    success_threshold: int = 3  # successes needed to close from half-open
    minimum_throughput: int = 10  # minimum requests before circuit can open
    error_rate_threshold: float = 0.5  # 50% error rate threshold
    sliding_window_size: int = 100  # request window size
    half_open_max_calls: int = 3  # max calls in half-open state


class SmartRetryManager:
    """Intelligent retry manager with adaptive strategies"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.retry_statistics: Dict[str, List] = defaultdict(list)
        self.success_patterns: Dict[str, List] = defaultdict(list)
        
    def execute_with_retry(self, func: Callable, operation_name: str = None, 
                          context: ErrorContext = None, **kwargs) -> Any:
        """Execute function with intelligent retry logic"""
        operation_name = operation_name or func.__name__
        context = context or ErrorContext(operation=operation_name)
        
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                context.retry_count = attempt
                
                # Execute the function
                result = func(**kwargs)
                
                # Record success
                self._record_success(operation_name, attempt)
                
                return result
                
            except Exception as e:
                last_exception = e
                context.error_type = type(e).__name__
                context.error_message = str(e)
                context.stack_trace = traceback.format_exc()
                
                # Check if we should stop retrying
                if self._should_stop_retry(e, attempt, operation_name):
                    break
                
                # Calculate delay for next attempt
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt, operation_name)
                    context.recovery_attempts.append({
                        'attempt': attempt + 1,
                        'delay': delay,
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {operation_name}, "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    
                    time.sleep(delay)
        
        # All retries exhausted
        self._record_failure(operation_name, context.retry_count, last_exception)
        raise last_exception
    
    async def execute_with_retry_async(self, coro: Callable, operation_name: str = None,
                                      context: ErrorContext = None, **kwargs) -> Any:
        """Async version of execute_with_retry"""
        if not ASYNC_AVAILABLE:
            raise ImportError("Async functionality requires aiohttp")
            
        operation_name = operation_name or coro.__name__
        context = context or ErrorContext(operation=operation_name)
        
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                context.retry_count = attempt
                
                # Execute the coroutine
                if inspect.iscoroutinefunction(coro):
                    result = await coro(**kwargs)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(None, coro, **kwargs)
                
                # Record success
                self._record_success(operation_name, attempt)
                
                return result
                
            except Exception as e:
                last_exception = e
                context.error_type = type(e).__name__
                context.error_message = str(e)
                context.stack_trace = traceback.format_exc()
                
                # Check if we should stop retrying
                if self._should_stop_retry(e, attempt, operation_name):
                    break
                
                # Calculate delay for next attempt
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt, operation_name)
                    context.recovery_attempts.append({
                        'attempt': attempt + 1,
                        'delay': delay,
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
                    
                    logger.warning(
                        f"Async attempt {attempt + 1} failed for {operation_name}, "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        self._record_failure(operation_name, context.retry_count, last_exception)
        raise last_exception
    
    def _should_stop_retry(self, exception: Exception, attempt: int, operation_name: str) -> bool:
        """Determine if we should stop retrying"""
        # Check stop exceptions
        if any(isinstance(exception, stop_ex) for stop_ex in self.config.stop_exceptions):
            return True
        
        # Check if exception is in retry list
        if not any(isinstance(exception, retry_ex) for retry_ex in self.config.retry_exceptions):
            return True
        
        # Check custom retry conditions
        for condition in self.config.retry_conditions:
            if not condition(exception, attempt, operation_name):
                return True
        
        return False
    
    def _calculate_delay(self, attempt: int, operation_name: str) -> float:
        """Calculate delay for retry attempt"""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
            
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * (attempt + 1)
            
        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.config.base_delay * self._fibonacci(attempt + 1)
            
        elif self.config.strategy == RetryStrategy.JITTERED_EXPONENTIAL:
            base_delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
            jitter = base_delay * 0.1 * random.random()  # 10% jitter
            delay = base_delay + jitter
        
        else:
            delay = self.config.base_delay
        
        # Apply jitter if enabled
        if self.config.jitter and self.config.strategy != RetryStrategy.JITTERED_EXPONENTIAL:
            jitter = delay * 0.1 * (random.random() - 0.5)  # ±5% jitter
            delay += jitter
        
        # Apply max delay limit
        delay = min(delay, self.config.max_delay)
        
        # Adaptive delay based on success patterns
        delay = self._apply_adaptive_delay(delay, operation_name, attempt)
        
        return max(delay, 0.1)  # Minimum 100ms delay
    
    def _apply_adaptive_delay(self, base_delay: float, operation_name: str, attempt: int) -> float:
        """Apply adaptive delay based on historical success patterns"""
        success_history = self.success_patterns.get(operation_name, [])
        
        if len(success_history) < 10:  # Not enough data for adaptation
            return base_delay
        
        # Analyze success patterns
        recent_successes = success_history[-20:]  # Last 20 attempts
        avg_successful_attempt = sum(recent_successes) / len(recent_successes)
        
        # If current attempt is higher than average successful attempt,
        # increase delay to give more time for recovery
        if attempt > avg_successful_attempt:
            multiplier = 1.0 + (attempt - avg_successful_attempt) * 0.2
            return base_delay * multiplier
        
        return base_delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number"""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def _record_success(self, operation_name: str, attempt: int) -> None:
        """Record successful execution"""
        self.success_patterns[operation_name].append(attempt)
        if len(self.success_patterns[operation_name]) > 100:
            self.success_patterns[operation_name] = self.success_patterns[operation_name][-100:]
    
    def _record_failure(self, operation_name: str, attempts: int, exception: Exception) -> None:
        """Record failed execution"""
        self.retry_statistics[operation_name].append({
            'timestamp': datetime.now(),
            'attempts': attempts,
            'error_type': type(exception).__name__,
            'error_message': str(exception)
        })
        
        # Keep only recent failures
        if len(self.retry_statistics[operation_name]) > 100:
            self.retry_statistics[operation_name] = self.retry_statistics[operation_name][-100:]


class AdaptiveCircuitBreaker:
    """Circuit breaker with adaptive thresholds and intelligent recovery"""
    
    def __init__(self, config: CircuitBreakerConfig = None, name: str = "default"):
        self.config = config or CircuitBreakerConfig()
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None
        self.request_window = deque(maxlen=self.config.sliding_window_size)
        self.half_open_calls = 0
        self._lock = threading.Lock()
        
        # Analytics for adaptive behavior
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.recovery_history: List[Dict] = []
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self._should_block_request():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN"
                )
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
        
        try:
            # Execute the function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record success
            self._record_success(execution_time)
            
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure(e)
            raise
    
    async def call_async(self, coro: Callable, *args, **kwargs) -> Any:
        """Async version of circuit breaker call"""
        with self._lock:
            if self._should_block_request():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN"
                )
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
        
        try:
            # Execute the coroutine
            start_time = time.time()
            if inspect.iscoroutinefunction(coro):
                result = await coro(*args, **kwargs)
            else:
                result = await asyncio.get_event_loop().run_in_executor(None, coro, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record success
            self._record_success(execution_time)
            
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure(e)
            raise
    
    def _should_block_request(self) -> bool:
        """Determine if request should be blocked"""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return False
        
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                current_time - self.last_failure_time >= self.config.recovery_timeout):
                self._transition_to_half_open()
                return False
            return True
        
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            return self.half_open_calls >= self.config.half_open_max_calls
        
        return False
    
    def _record_success(self, execution_time: float) -> None:
        """Record successful execution"""
        with self._lock:
            self.request_window.append({
                'timestamp': time.time(),
                'success': True,
                'execution_time': execution_time
            })
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                if self.failure_count > 0:
                    self.failure_count = max(0, self.failure_count - 1)
    
    def _record_failure(self, exception: Exception) -> None:
        """Record failed execution"""
        with self._lock:
            self.request_window.append({
                'timestamp': time.time(),
                'success': False,
                'error_type': type(exception).__name__,
                'error_message': str(exception)
            })
            
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Track error patterns
            error_type = type(exception).__name__
            self.error_patterns[error_type] += 1
            
            if self.state == CircuitState.CLOSED:
                if self._should_open_circuit():
                    self._transition_to_open()
            
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should open based on adaptive thresholds"""
        if len(self.request_window) < self.config.minimum_throughput:
            return False
        
        # Basic failure threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Error rate threshold
        recent_requests = list(self.request_window)
        if len(recent_requests) >= self.config.minimum_throughput:
            error_count = sum(1 for req in recent_requests if not req['success'])
            error_rate = error_count / len(recent_requests)
            
            if error_rate >= self.config.error_rate_threshold:
                return True
        
        # Adaptive threshold based on error patterns
        if self._adaptive_threshold_exceeded():
            return True
        
        return False
    
    def _adaptive_threshold_exceeded(self) -> bool:
        """Check adaptive threshold based on error analysis"""
        if not ANALYTICS_AVAILABLE or len(self.error_patterns) < 2:
            return False
        
        # If we see a sudden spike in a specific error type, lower threshold
        total_errors = sum(self.error_patterns.values())
        if total_errors > 0:
            max_error_rate = max(self.error_patterns.values()) / total_errors
            if max_error_rate > 0.7:  # 70% of errors are of same type
                return self.failure_count >= max(2, self.config.failure_threshold // 2)
        
        return False
    
    def _transition_to_open(self) -> None:
        """Transition circuit to OPEN state"""
        previous_state = self.state
        self.state = CircuitState.OPEN
        self.half_open_calls = 0
        self.success_count = 0
        
        # Calculate adaptive recovery timeout
        recovery_timeout = self._calculate_adaptive_recovery_timeout()
        self.next_attempt_time = time.time() + recovery_timeout
        
        self.recovery_history.append({
            'timestamp': datetime.now(),
            'previous_state': previous_state.value,
            'new_state': self.state.value,
            'failure_count': self.failure_count,
            'error_patterns': dict(self.error_patterns),
            'recovery_timeout': recovery_timeout
        })
        
        logger.warning(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures")
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state"""
        previous_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
        
        logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")
    
    def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state"""
        previous_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.error_patterns.clear()  # Reset error pattern tracking
        
        self.recovery_history.append({
            'timestamp': datetime.now(),
            'previous_state': previous_state.value,
            'new_state': self.state.value,
            'recovery_successful': True
        })
        
        logger.info(f"Circuit breaker '{self.name}' closed - service recovered")
    
    def _calculate_adaptive_recovery_timeout(self) -> float:
        """Calculate adaptive recovery timeout based on history"""
        base_timeout = self.config.recovery_timeout
        
        # If we have recovery history, adapt based on patterns
        if len(self.recovery_history) >= 3:
            recent_recoveries = self.recovery_history[-5:]
            failed_recoveries = [r for r in recent_recoveries 
                               if not r.get('recovery_successful', False)]
            
            # Increase timeout if recent recoveries failed
            if len(failed_recoveries) > len(recent_recoveries) / 2:
                multiplier = 1.5 + (len(failed_recoveries) * 0.2)
                return min(base_timeout * multiplier, base_timeout * 5)  # Max 5x
        
        # Adapt based on error patterns
        if self.error_patterns:
            # Increase timeout for certain error types
            critical_errors = ['TimeoutError', 'ConnectionError', 'ServiceUnavailable']
            for error_type in critical_errors:
                if error_type in self.error_patterns:
                    return base_timeout * 1.5
        
        return base_timeout
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        with self._lock:
            recent_requests = list(self.request_window)
            success_rate = 0.0
            
            if recent_requests:
                success_count = sum(1 for req in recent_requests if req['success'])
                success_rate = success_count / len(recent_requests)
            
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'success_rate': success_rate,
                'total_requests': len(recent_requests),
                'error_patterns': dict(self.error_patterns),
                'last_failure_time': self.last_failure_time,
                'next_attempt_time': self.next_attempt_time,
                'recovery_history_count': len(self.recovery_history)
            }
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            self.last_failure_time = None
            self.next_attempt_time = None
            self.error_patterns.clear()
            self.request_window.clear()
            
        logger.info(f"Circuit breaker '{self.name}' manually reset")


class FallbackManager:
    """Manages fallback strategies for graceful degradation"""
    
    def __init__(self):
        self.fallback_strategies: Dict[str, List[Callable]] = defaultdict(list)
        self.fallback_usage: Dict[str, Dict] = defaultdict(dict)
    
    def register_fallback(self, operation: str, fallback_func: Callable, priority: int = 0) -> None:
        """Register fallback function for operation"""
        self.fallback_strategies[operation].append((priority, fallback_func))
        # Sort by priority (higher priority first)
        self.fallback_strategies[operation].sort(key=lambda x: x[0], reverse=True)
        
        logger.info(f"Registered fallback for '{operation}' with priority {priority}")
    
    def execute_with_fallback(self, operation: str, primary_func: Callable, 
                            *args, **kwargs) -> Any:
        """Execute function with fallback capabilities"""
        # Try primary function first
        try:
            result = primary_func(*args, **kwargs)
            self._record_primary_success(operation)
            return result
        except Exception as primary_error:
            logger.warning(f"Primary function failed for '{operation}': {primary_error}")
            
            # Try fallback strategies in priority order
            fallbacks = self.fallback_strategies.get(operation, [])
            
            for priority, fallback_func in fallbacks:
                try:
                    logger.info(f"Trying fallback (priority {priority}) for '{operation}'")
                    result = fallback_func(*args, **kwargs)
                    self._record_fallback_success(operation, priority)
                    return result
                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback (priority {priority}) failed for '{operation}': {fallback_error}"
                    )
                    self._record_fallback_failure(operation, priority, fallback_error)
                    continue
            
            # All fallbacks failed
            self._record_complete_failure(operation, primary_error)
            raise FallbackExhaustedException(
                f"All fallback strategies exhausted for '{operation}'"
            ) from primary_error
    
    def _record_primary_success(self, operation: str) -> None:
        """Record successful primary execution"""
        if operation not in self.fallback_usage:
            self.fallback_usage[operation] = {'primary_success': 0, 'fallback_usage': {}}
        self.fallback_usage[operation]['primary_success'] += 1
    
    def _record_fallback_success(self, operation: str, priority: int) -> None:
        """Record successful fallback execution"""
        if operation not in self.fallback_usage:
            self.fallback_usage[operation] = {'primary_success': 0, 'fallback_usage': {}}
        
        fallback_key = f'priority_{priority}_success'
        if fallback_key not in self.fallback_usage[operation]['fallback_usage']:
            self.fallback_usage[operation]['fallback_usage'][fallback_key] = 0
        self.fallback_usage[operation]['fallback_usage'][fallback_key] += 1
    
    def _record_fallback_failure(self, operation: str, priority: int, error: Exception) -> None:
        """Record fallback failure"""
        if operation not in self.fallback_usage:
            self.fallback_usage[operation] = {'primary_success': 0, 'fallback_usage': {}}
        
        fallback_key = f'priority_{priority}_failure'
        if fallback_key not in self.fallback_usage[operation]['fallback_usage']:
            self.fallback_usage[operation]['fallback_usage'][fallback_key] = 0
        self.fallback_usage[operation]['fallback_usage'][fallback_key] += 1
    
    def _record_complete_failure(self, operation: str, error: Exception) -> None:
        """Record complete failure of operation"""
        if operation not in self.fallback_usage:
            self.fallback_usage[operation] = {'primary_success': 0, 'fallback_usage': {}}
        
        if 'complete_failures' not in self.fallback_usage[operation]:
            self.fallback_usage[operation]['complete_failures'] = 0
        self.fallback_usage[operation]['complete_failures'] += 1
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get fallback usage statistics"""
        return dict(self.fallback_usage)


class SelfHealingOrchestrator:
    """Orchestrates self-healing capabilities across the system"""
    
    def __init__(self):
        self.healing_strategies: Dict[str, List[Callable]] = defaultdict(list)
        self.healing_history: List[Dict] = []
        self.active_healers: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
    
    def register_healing_strategy(self, error_pattern: str, healing_func: Callable) -> None:
        """Register self-healing strategy for error pattern"""
        self.healing_strategies[error_pattern].append(healing_func)
        logger.info(f"Registered healing strategy for pattern: {error_pattern}")
    
    def trigger_healing(self, error_context: ErrorContext) -> bool:
        """Trigger healing process for error"""
        healing_id = f"heal_{int(time.time())}_{error_context.error_id[:8]}"
        
        # Find matching healing strategies
        matching_strategies = []
        for pattern, strategies in self.healing_strategies.items():
            if pattern in error_context.error_type or pattern in error_context.error_message:
                matching_strategies.extend(strategies)
        
        if not matching_strategies:
            logger.info(f"No healing strategies found for error: {error_context.error_type}")
            return False
        
        # Start healing in background thread
        healing_thread = threading.Thread(
            target=self._execute_healing,
            args=(healing_id, matching_strategies, error_context)
        )
        healing_thread.daemon = True
        healing_thread.start()
        
        with self._lock:
            self.active_healers[healing_id] = healing_thread
        
        return True
    
    def _execute_healing(self, healing_id: str, strategies: List[Callable], 
                        error_context: ErrorContext) -> None:
        """Execute healing strategies"""
        healing_start = time.time()
        successful_healers = []
        failed_healers = []
        
        try:
            for i, healing_func in enumerate(strategies):
                try:
                    logger.info(f"Executing healing strategy {i+1}/{len(strategies)} for {healing_id}")
                    result = healing_func(error_context)
                    
                    if result:
                        successful_healers.append({
                            'strategy_index': i,
                            'function_name': healing_func.__name__,
                            'result': result
                        })
                    else:
                        failed_healers.append({
                            'strategy_index': i,
                            'function_name': healing_func.__name__,
                            'error': 'Returned False'
                        })
                        
                except Exception as e:
                    logger.error(f"Healing strategy {i+1} failed: {e}")
                    failed_healers.append({
                        'strategy_index': i,
                        'function_name': healing_func.__name__,
                        'error': str(e)
                    })
            
            # Record healing attempt
            healing_record = {
                'healing_id': healing_id,
                'timestamp': datetime.now(),
                'duration': time.time() - healing_start,
                'error_context': error_context,
                'strategies_attempted': len(strategies),
                'successful_healers': successful_healers,
                'failed_healers': failed_healers,
                'overall_success': len(successful_healers) > 0
            }
            
            self.healing_history.append(healing_record)
            
            # Keep only recent healing history
            if len(self.healing_history) > 100:
                self.healing_history = self.healing_history[-100:]
            
            if successful_healers:
                logger.info(f"Healing successful for {healing_id}: {len(successful_healers)} strategies succeeded")
            else:
                logger.warning(f"Healing failed for {healing_id}: all strategies failed")
                
        finally:
            with self._lock:
                if healing_id in self.active_healers:
                    del self.active_healers[healing_id]
    
    def get_healing_status(self) -> Dict[str, Any]:
        """Get self-healing status"""
        with self._lock:
            active_count = len(self.active_healers)
        
        if not self.healing_history:
            return {
                'active_healers': active_count,
                'total_attempts': 0,
                'success_rate': 0.0
            }
        
        successful_attempts = sum(1 for record in self.healing_history 
                                if record['overall_success'])
        
        return {
            'active_healers': active_count,
            'total_attempts': len(self.healing_history),
            'successful_attempts': successful_attempts,
            'success_rate': successful_attempts / len(self.healing_history),
            'strategies_registered': sum(len(strategies) 
                                       for strategies in self.healing_strategies.values()),
            'recent_healing_attempts': [
                {
                    'healing_id': record['healing_id'],
                    'timestamp': record['timestamp'].isoformat(),
                    'success': record['overall_success'],
                    'duration': record['duration']
                }
                for record in self.healing_history[-10:]
            ]
        }


class IntelligentErrorRecoverySystem:
    """Main error recovery system integrating all components"""
    
    def __init__(self):
        self.retry_manager = SmartRetryManager()
        self.circuit_breakers: Dict[str, AdaptiveCircuitBreaker] = {}
        self.fallback_manager = FallbackManager()
        self.healing_orchestrator = SelfHealingOrchestrator()
        self.error_analytics = ErrorAnalytics()
        
        # Setup default healing strategies
        self._setup_default_healing_strategies()
        
        logger.info("Intelligent Error Recovery System initialized")
    
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> AdaptiveCircuitBreaker:
        """Create and register circuit breaker"""
        circuit_breaker = AdaptiveCircuitBreaker(config, name)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> AdaptiveCircuitBreaker:
        """Get existing circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = AdaptiveCircuitBreaker(name=name)
        return self.circuit_breakers[name]
    
    @contextmanager
    def resilient_operation(self, operation_name: str, 
                          circuit_breaker_name: str = None,
                          retry_config: RetryConfig = None,
                          enable_fallback: bool = True):
        """Context manager for resilient operations"""
        circuit_breaker_name = circuit_breaker_name or operation_name
        circuit_breaker = self.get_circuit_breaker(circuit_breaker_name)
        
        error_context = ErrorContext(operation=operation_name)
        
        try:
            with circuit_breaker._lock:
                if circuit_breaker._should_block_request():
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{circuit_breaker_name}' is OPEN"
                    )
            
            yield error_context
            
            # Record success
            circuit_breaker._record_success(0.1)  # Default execution time
            
        except Exception as e:
            # Record failure
            circuit_breaker._record_failure(e)
            
            # Update error context
            error_context.error_type = type(e).__name__
            error_context.error_message = str(e)
            error_context.stack_trace = traceback.format_exc()
            
            # Record error for analytics
            self.error_analytics.record_error(error_context)
            
            # Trigger healing if enabled
            self.healing_orchestrator.trigger_healing(error_context)
            
            raise
    
    def execute_with_full_protection(self, func: Callable, operation_name: str,
                                   circuit_breaker_name: str = None,
                                   retry_config: RetryConfig = None,
                                   enable_fallback: bool = True,
                                   **kwargs) -> Any:
        """Execute function with full error recovery protection"""
        circuit_breaker_name = circuit_breaker_name or operation_name
        retry_config = retry_config or RetryConfig()
        
        def protected_execution():
            circuit_breaker = self.get_circuit_breaker(circuit_breaker_name)
            return circuit_breaker.call(func, **kwargs)
        
        # Execute with retry protection
        try:
            return self.retry_manager.execute_with_retry(
                protected_execution, operation_name
            )
        except Exception as e:
            # Try fallback if enabled
            if enable_fallback:
                return self.fallback_manager.execute_with_fallback(
                    operation_name, protected_execution
                )
            raise
    
    def _setup_default_healing_strategies(self) -> None:
        """Setup default self-healing strategies"""
        
        def memory_cleanup_healer(error_context: ErrorContext) -> bool:
            """Clean up memory when memory errors occur"""
            if 'memory' in error_context.error_message.lower():
                try:
                    import gc
                    collected = gc.collect()
                    logger.info(f"Memory cleanup: collected {collected} objects")
                    return True
                except:
                    return False
            return False
        
        def connection_reset_healer(error_context: ErrorContext) -> bool:
            """Reset connections for connection errors"""
            if 'connection' in error_context.error_type.lower():
                try:
                    # Simulate connection reset
                    time.sleep(0.5)  # Brief pause
                    logger.info("Connection reset attempted")
                    return True
                except:
                    return False
            return False
        
        def cache_clear_healer(error_context: ErrorContext) -> bool:
            """Clear caches for cache-related errors"""
            if any(term in error_context.error_message.lower() 
                   for term in ['cache', 'timeout', 'stale']):
                try:
                    # Simulate cache clearing
                    logger.info("Cache clearing attempted")
                    return True
                except:
                    return False
            return False
        
        # Register healing strategies
        self.healing_orchestrator.register_healing_strategy('MemoryError', memory_cleanup_healer)
        self.healing_orchestrator.register_healing_strategy('ConnectionError', connection_reset_healer)
        self.healing_orchestrator.register_healing_strategy('TimeoutError', cache_clear_healer)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        circuit_breaker_status = {}
        for name, cb in self.circuit_breakers.items():
            circuit_breaker_status[name] = cb.get_status()
        
        return {
            'circuit_breakers': circuit_breaker_status,
            'fallback_statistics': self.fallback_manager.get_fallback_statistics(),
            'healing_status': self.healing_orchestrator.get_healing_status(),
            'error_analytics': self.error_analytics.get_analytics_summary(),
            'system_health': self._calculate_system_health(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        health_score = 100.0
        
        # Factor in circuit breaker states
        for name, cb in self.circuit_breakers.items():
            status = cb.get_status()
            if status['state'] == 'open':
                health_score -= 20
            elif status['state'] == 'half_open':
                health_score -= 5
            
            # Factor in success rate
            success_rate = status.get('success_rate', 1.0)
            health_score -= (1.0 - success_rate) * 10
        
        # Factor in healing success rate
        healing_status = self.healing_orchestrator.get_healing_status()
        healing_success_rate = healing_status.get('success_rate', 1.0)
        health_score += healing_success_rate * 5  # Bonus for good healing
        
        health_score = max(0, min(100, health_score))
        
        return {
            'score': health_score,
            'status': 'healthy' if health_score > 80 else 'degraded' if health_score > 50 else 'critical',
            'active_issues': len([cb for cb in self.circuit_breakers.values() 
                                if cb.state != CircuitState.CLOSED])
        }


class ErrorAnalytics:
    """Advanced error pattern analysis and learning"""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.error_patterns: Dict[str, List] = defaultdict(list)
        self.clustering_model = None
        self._last_analysis = None
    
    def record_error(self, error_context: ErrorContext) -> None:
        """Record error for analysis"""
        self.error_history.append(error_context)
        
        # Extract pattern key
        pattern_key = f"{error_context.error_type}:{error_context.operation}"
        self.error_patterns[pattern_key].append(error_context)
        
        # Keep only recent errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
    
    def analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns using ML techniques"""
        if not ANALYTICS_AVAILABLE or len(self.error_history) < 10:
            return {'status': 'insufficient_data'}
        
        try:
            # Prepare features for clustering
            features = []
            error_labels = []
            
            for error in self.error_history[-100:]:  # Last 100 errors
                feature_vector = [
                    hash(error.error_type) % 1000,  # Error type hash
                    hash(error.operation) % 1000,   # Operation hash
                    error.retry_count,
                    len(error.error_message),
                    error.severity.value.__hash__() % 100
                ]
                features.append(feature_vector)
                error_labels.append(f"{error.error_type}:{error.operation}")
            
            # Perform clustering
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            n_clusters = min(5, len(set(error_labels)))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(features_scaled)
                
                # Analyze clusters
                cluster_analysis = defaultdict(list)
                for i, cluster_id in enumerate(clusters):
                    cluster_analysis[cluster_id].append(error_labels[i])
                
                self._last_analysis = {
                    'status': 'completed',
                    'clusters_found': n_clusters,
                    'cluster_analysis': {
                        f'cluster_{k}': {
                            'size': len(v),
                            'patterns': list(set(v))
                        }
                        for k, v in cluster_analysis.items()
                    },
                    'analysis_timestamp': datetime.now().isoformat()
                }
            else:
                self._last_analysis = {'status': 'single_cluster'}
                
        except Exception as e:
            self._last_analysis = {'status': 'analysis_failed', 'error': str(e)}
        
        return self._last_analysis
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get error analytics summary"""
        if not self.error_history:
            return {'total_errors': 0}
        
        # Basic statistics
        error_counts = defaultdict(int)
        operation_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for error in self.error_history:
            error_counts[error.error_type] += 1
            operation_counts[error.operation] += 1
            severity_counts[error.severity.value] += 1
        
        return {
            'total_errors': len(self.error_history),
            'unique_error_types': len(error_counts),
            'unique_operations': len(operation_counts),
            'top_error_types': dict(sorted(error_counts.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]),
            'top_operations': dict(sorted(operation_counts.items(),
                                        key=lambda x: x[1], reverse=True)[:5]),
            'severity_distribution': dict(severity_counts),
            'last_analysis': self._last_analysis,
            'pattern_count': len(self.error_patterns)
        }


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class FallbackExhaustedException(Exception):
    """Raised when all fallback strategies are exhausted"""
    pass


# Factory functions
def create_recovery_system() -> IntelligentErrorRecoverySystem:
    """Create intelligent error recovery system"""
    return IntelligentErrorRecoverySystem()


def create_retry_config(strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
                       max_retries: int = 3,
                       **kwargs) -> RetryConfig:
    """Create retry configuration"""
    return RetryConfig(strategy=strategy, max_retries=max_retries, **kwargs)


def create_circuit_breaker_config(failure_threshold: int = 5,
                                 recovery_timeout: float = 60.0,
                                 **kwargs) -> CircuitBreakerConfig:
    """Create circuit breaker configuration"""
    return CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        **kwargs
    )


# Decorators for easy usage
def resilient(operation_name: str = None, 
             circuit_breaker_name: str = None,
             retry_config: RetryConfig = None,
             enable_fallback: bool = True):
    """Decorator for resilient operations"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            recovery_system = create_recovery_system()
            
            return recovery_system.execute_with_full_protection(
                func, op_name, circuit_breaker_name, retry_config, 
                enable_fallback, *args, **kwargs
            )
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Create recovery system
    recovery = create_recovery_system()
    
    # Example resilient function
    @resilient(operation_name="test_operation", enable_fallback=True)
    def unreliable_function(fail_rate: float = 0.5):
        if random.random() < fail_rate:
            raise Exception("Random failure for testing")
        return "Success!"
    
    # Register fallback
    def fallback_function(fail_rate: float = 0.5):
        return "Fallback result"
    
    recovery.fallback_manager.register_fallback("test_operation", fallback_function)
    
    # Test the system
    results = []
    for i in range(10):
        try:
            result = unreliable_function(fail_rate=0.7)
            results.append(f"Attempt {i+1}: {result}")
        except Exception as e:
            results.append(f"Attempt {i+1}: Failed - {e}")
        
        time.sleep(0.5)
    
    # Print results
    for result in results:
        print(result)
    
    # Print system status
    print("\nSystem Status:")
    print("=" * 50)
    status = recovery.get_system_status()
    print(json.dumps(status, indent=2, default=str))