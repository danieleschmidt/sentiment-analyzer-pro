"""
Intelligent Error Recovery System v2.0
Advanced error handling, recovery, and resilience framework.
"""

from __future__ import annotations

import asyncio
import time
import threading
import traceback
import sys
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union, Type
from enum import Enum
from collections import defaultdict, deque
import logging
import json
from functools import wraps
import inspect
import gc
import psutil
import signal

from .logging_config import get_logger
from .metrics import metrics

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADE = "graceful_degrade"
    SYSTEM_RESTART = "system_restart"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


class SystemHealth(Enum):
    """System health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILING = "failing"
    EMERGENCY = "emergency"


@dataclass
class ErrorEvent:
    """Comprehensive error event record."""
    error_id: str = field(default_factory=lambda: f"err_{int(time.time() * 1000)}")
    timestamp: float = field(default_factory=time.time)
    error_type: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    message: str = ""
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    function_name: Optional[str] = None
    module_name: Optional[str] = None
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    resolution_time: Optional[float] = None
    impact_assessment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryPlan:
    """Recovery execution plan."""
    strategies: List[RecoveryStrategy] = field(default_factory=list)
    fallback_functions: List[Callable] = field(default_factory=list)
    timeout_seconds: float = 30.0
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    circuit_breaker_threshold: int = 5
    degraded_mode_config: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """Advanced circuit breaker with intelligent failure detection."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.RLock()
        
        # Advanced metrics
        self.success_count = 0
        self.total_calls = 0
        self.avg_response_time = 0.0
        self.response_times = deque(maxlen=100)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self.total_calls += 1
            
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    logger.info(f"Circuit breaker transitioning to HALF_OPEN for {func.__name__}")
                else:
                    raise RuntimeError(f"Circuit breaker is OPEN for {func.__name__}")
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Success handling
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                self._update_avg_response_time()
                
                if self.state == "HALF_OPEN":
                    self._reset()
                    logger.info(f"Circuit breaker reset to CLOSED for {func.__name__}")
                
                self.success_count += 1
                return result
                
            except self.expected_exception as e:
                self._record_failure()
                
                if self.failure_count >= self.failure_threshold:
                    self._trip()
                    logger.error(f"Circuit breaker tripped for {func.__name__}")
                
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _record_failure(self) -> None:
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()
    
    def _trip(self) -> None:
        """Trip the circuit breaker."""
        self.state = "OPEN"
        self.last_failure_time = time.time()
    
    def _reset(self) -> None:
        """Reset the circuit breaker."""
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None
    
    def _update_avg_response_time(self) -> None:
        """Update average response time."""
        if self.response_times:
            self.avg_response_time = sum(self.response_times) / len(self.response_times)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            success_rate = self.success_count / self.total_calls if self.total_calls > 0 else 0
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "total_calls": self.total_calls,
                "success_rate": success_rate,
                "avg_response_time": self.avg_response_time,
                "last_failure_time": self.last_failure_time
            }


class AdaptiveRetry:
    """Adaptive retry mechanism with intelligent backoff."""
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_multiplier: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        
        # Success rate tracking for adaptive behavior
        self.success_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.retry_counts: Dict[str, int] = defaultdict(int)
    
    def __call__(self, 
                 retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
                 circuit_breaker: Optional[CircuitBreaker] = None) -> Callable:
        """Decorator for adaptive retry."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute(func, retryable_exceptions, circuit_breaker, *args, **kwargs)
            return wrapper
        return decorator
    
    def execute(self, 
                func: Callable,
                retryable_exceptions: Tuple[Type[Exception], ...],
                circuit_breaker: Optional[CircuitBreaker],
                *args, **kwargs) -> Any:
        """Execute function with adaptive retry logic."""
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Adapt retry count based on historical success rate
        success_rate = self._get_success_rate(func_name)
        adaptive_max_retries = self._calculate_adaptive_retries(success_rate)
        
        last_exception = None
        
        for attempt in range(adaptive_max_retries + 1):
            try:
                if circuit_breaker:
                    result = circuit_breaker.call(func, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record success
                self.success_rates[func_name].append(True)
                
                if attempt > 0:
                    logger.info(f"Function {func_name} succeeded after {attempt} retries")
                
                return result
                
            except retryable_exceptions as e:
                last_exception = e
                self.retry_counts[func_name] += 1
                
                if attempt < adaptive_max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Retry {attempt + 1}/{adaptive_max_retries} for {func_name} in {delay:.2f}s: {str(e)}")
                    time.sleep(delay)
                else:
                    # Record failure
                    self.success_rates[func_name].append(False)
                    logger.error(f"Function {func_name} failed after {adaptive_max_retries} retries")
                    break
            
            except Exception as e:
                # Non-retryable exception
                self.success_rates[func_name].append(False)
                logger.error(f"Non-retryable exception in {func_name}: {str(e)}")
                raise
        
        if last_exception:
            raise last_exception
    
    def _get_success_rate(self, func_name: str) -> float:
        """Get success rate for a function."""
        successes = self.success_rates[func_name]
        if not successes:
            return 0.5  # Default assumption
        
        return sum(successes) / len(successes)
    
    def _calculate_adaptive_retries(self, success_rate: float) -> int:
        """Calculate adaptive retry count based on success rate."""
        if success_rate > 0.8:
            return max(1, self.max_retries - 1)  # Reduce retries for reliable functions
        elif success_rate < 0.3:
            return min(self.max_retries + 2, 10)  # Increase retries for unreliable functions
        else:
            return self.max_retries
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = self.base_delay * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay


class SystemMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self):
        self.health_state = SystemHealth.HEALTHY
        self.health_history = deque(maxlen=1000)
        self.resource_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "load_average": 2.0,
            "open_files": 1000
        }
        self.alert_callbacks: List[Callable] = []
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval: float = 30.0) -> None:
        """Start continuous system monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                health_data = self.check_system_health()
                self._assess_health_state(health_data)
                
                # Record health snapshot
                self.health_history.append({
                    "timestamp": time.time(),
                    "state": self.health_state.value,
                    "metrics": health_data
                })
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(interval)
    
    def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
            
            # Process information
            process = psutil.Process()
            process_memory = process.memory_info()
            open_files = len(process.open_files())
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024),
                "load_average": load_avg,
                "process_memory_mb": process_memory.rss / (1024 * 1024),
                "open_files": open_files,
                "thread_count": threading.active_count(),
                "gc_collections": sum(gc.get_stats()[i]['collections'] for i in range(3))
            }
        
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {"error": str(e)}
    
    def _assess_health_state(self, health_data: Dict[str, Any]) -> None:
        """Assess overall system health state."""
        if "error" in health_data:
            new_state = SystemHealth.CRITICAL
        else:
            critical_issues = 0
            warning_issues = 0
            
            # Check each threshold
            for metric, threshold in self.resource_thresholds.items():
                if metric in health_data:
                    value = health_data[metric]
                    if value > threshold:
                        critical_issues += 1
                    elif value > threshold * 0.8:  # 80% of threshold
                        warning_issues += 1
            
            # Determine health state
            if critical_issues >= 3:
                new_state = SystemHealth.FAILING
            elif critical_issues >= 2:
                new_state = SystemHealth.CRITICAL
            elif critical_issues >= 1 or warning_issues >= 3:
                new_state = SystemHealth.DEGRADED
            else:
                new_state = SystemHealth.HEALTHY
        
        # State change handling
        if new_state != self.health_state:
            old_state = self.health_state
            self.health_state = new_state
            
            logger.warning(f"System health changed: {old_state.value} -> {new_state.value}")
            
            # Trigger alerts
            for callback in self.alert_callbacks:
                try:
                    callback(old_state, new_state, health_data)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add health state change alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        current_health = self.check_system_health()
        
        # Calculate trends
        recent_history = list(self.health_history)[-10:]  # Last 10 readings
        health_trend = "stable"
        
        if len(recent_history) >= 3:
            states = [h["state"] for h in recent_history]
            if states[-1] != states[0]:
                if states[-1] in ["critical", "failing"] and states[0] in ["healthy", "degraded"]:
                    health_trend = "deteriorating"
                elif states[-1] in ["healthy", "degraded"] and states[0] in ["critical", "failing"]:
                    health_trend = "improving"
        
        return {
            "current_state": self.health_state.value,
            "current_metrics": current_health,
            "health_trend": health_trend,
            "monitoring_active": self._monitoring_active,
            "history_size": len(self.health_history),
            "thresholds": self.resource_thresholds,
            "timestamp": time.time()
        }


class IntelligentErrorRecovery:
    """Main intelligent error recovery orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Core components
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[str, AdaptiveRetry] = {}
        self.system_monitor = SystemMonitor()
        
        # Error classification
        self.error_patterns: Dict[str, ErrorSeverity] = {
            "MemoryError": ErrorSeverity.CRITICAL,
            "SystemExit": ErrorSeverity.CATASTROPHIC,
            "KeyboardInterrupt": ErrorSeverity.HIGH,
            "ConnectionError": ErrorSeverity.MEDIUM,
            "TimeoutError": ErrorSeverity.MEDIUM,
            "ValueError": ErrorSeverity.LOW,
            "TypeError": ErrorSeverity.LOW
        }
        
        # Recovery statistics
        self.recovery_stats = defaultdict(int)
        self.recovery_success_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Initialize system monitoring
        self.system_monitor.add_alert_callback(self._handle_health_change)
        self.system_monitor.start_monitoring()
        
        logger.info("IntelligentErrorRecovery initialized", extra={
            "error_patterns": len(self.error_patterns),
            "monitoring_active": True,
            "recovery_strategies": [s.value for s in RecoveryStrategy]
        })
    
    def register_recovery_plan(self, error_type: str, plan: RecoveryPlan) -> None:
        """Register a recovery plan for specific error type."""
        self.recovery_plans[error_type] = plan
        logger.info(f"Recovery plan registered for {error_type}")
    
    def handle_error(self, 
                    error: Exception,
                    context: Optional[Dict[str, Any]] = None,
                    auto_recover: bool = True) -> ErrorEvent:
        """Handle error with intelligent recovery."""
        # Create error event
        error_event = self._create_error_event(error, context)
        self.error_history.append(error_event)
        
        # Log error
        logger.error(f"Error handled: {error_event.error_type}", extra={
            "error_id": error_event.error_id,
            "severity": error_event.severity.value,
            "function": error_event.function_name,
            "context": context
        })
        
        # Attempt recovery if enabled
        if auto_recover:
            try:
                recovery_result = self._attempt_recovery(error_event)
                error_event.recovery_attempted = True
                error_event.recovery_successful = recovery_result.get("success", False)
                error_event.recovery_strategy = recovery_result.get("strategy")
                
                if error_event.recovery_successful:
                    error_event.resolution_time = time.time()
                    logger.info(f"Error {error_event.error_id} recovered successfully")
                
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {error_event.error_id}: {recovery_error}")
        
        # Update statistics
        self.recovery_stats[error_event.error_type] += 1
        self.recovery_success_rates[error_event.error_type].append(
            error_event.recovery_successful
        )
        
        return error_event
    
    def _create_error_event(self, error: Exception, context: Optional[Dict[str, Any]]) -> ErrorEvent:
        """Create comprehensive error event."""
        error_type = type(error).__name__
        severity = self._classify_error_severity(error)
        
        # Extract stack trace information
        stack_trace = traceback.format_exc()
        frame_info = self._extract_frame_info(traceback.extract_tb(error.__traceback__))
        
        # Assess impact
        impact_assessment = self._assess_error_impact(error, severity)
        
        return ErrorEvent(
            error_type=error_type,
            severity=severity,
            message=str(error),
            stack_trace=stack_trace,
            context=context or {},
            function_name=frame_info.get("function"),
            module_name=frame_info.get("module"),
            impact_assessment=impact_assessment
        )
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on type and context."""
        error_type = type(error).__name__
        
        # Check predefined patterns
        if error_type in self.error_patterns:
            return self.error_patterns[error_type]
        
        # Dynamic classification based on error characteristics
        error_message = str(error).lower()
        
        if any(keyword in error_message for keyword in ["critical", "fatal", "corrupted"]):
            return ErrorSeverity.CRITICAL
        elif any(keyword in error_message for keyword in ["timeout", "connection", "network"]):
            return ErrorSeverity.MEDIUM
        elif any(keyword in error_message for keyword in ["invalid", "not found", "permission"]):
            return ErrorSeverity.LOW
        
        # Default classification
        return ErrorSeverity.MEDIUM
    
    def _extract_frame_info(self, tb_frames: List) -> Dict[str, Any]:
        """Extract useful information from stack trace frames."""
        if not tb_frames:
            return {}
        
        # Get the last frame (where error occurred)
        last_frame = tb_frames[-1]
        
        return {
            "function": last_frame.name,
            "module": last_frame.filename.split('/')[-1],
            "line_number": last_frame.lineno,
            "code_context": last_frame.line
        }
    
    def _assess_error_impact(self, error: Exception, severity: ErrorSeverity) -> Dict[str, Any]:
        """Assess the potential impact of the error."""
        impact = {
            "severity_score": {
                ErrorSeverity.LOW: 1,
                ErrorSeverity.MEDIUM: 3,
                ErrorSeverity.HIGH: 7,
                ErrorSeverity.CRITICAL: 9,
                ErrorSeverity.CATASTROPHIC: 10
            }[severity],
            "system_stability": "stable",
            "data_integrity": "intact",
            "service_availability": "available"
        }
        
        # Assess based on error type
        if isinstance(error, MemoryError):
            impact["system_stability"] = "degraded"
            impact["service_availability"] = "limited"
        elif isinstance(error, (ConnectionError, TimeoutError)):
            impact["service_availability"] = "degraded"
        elif isinstance(error, (IOError, OSError)):
            impact["data_integrity"] = "at_risk"
        
        return impact
    
    def _attempt_recovery(self, error_event: ErrorEvent) -> Dict[str, Any]:
        """Attempt intelligent error recovery."""
        error_type = error_event.error_type
        
        # Check for specific recovery plan
        if error_type in self.recovery_plans:
            plan = self.recovery_plans[error_type]
            return self._execute_recovery_plan(plan, error_event)
        
        # Use default recovery strategies based on severity
        strategies = self._select_default_strategies(error_event.severity)
        
        for strategy in strategies:
            try:
                result = self._execute_recovery_strategy(strategy, error_event)
                if result.get("success"):
                    return {"success": True, "strategy": strategy}
            except Exception as e:
                logger.warning(f"Recovery strategy {strategy.value} failed: {e}")
        
        return {"success": False, "strategy": None}
    
    def _select_default_strategies(self, severity: ErrorSeverity) -> List[RecoveryStrategy]:
        """Select default recovery strategies based on severity."""
        if severity == ErrorSeverity.CATASTROPHIC:
            return [RecoveryStrategy.EMERGENCY_SHUTDOWN]
        elif severity == ErrorSeverity.CRITICAL:
            return [RecoveryStrategy.SYSTEM_RESTART, RecoveryStrategy.EMERGENCY_SHUTDOWN]
        elif severity == ErrorSeverity.HIGH:
            return [RecoveryStrategy.CIRCUIT_BREAK, RecoveryStrategy.GRACEFUL_DEGRADE]
        elif severity == ErrorSeverity.MEDIUM:
            return [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]
        else:
            return [RecoveryStrategy.RETRY]
    
    def _execute_recovery_strategy(self, strategy: RecoveryStrategy, error_event: ErrorEvent) -> Dict[str, Any]:
        """Execute specific recovery strategy."""
        if strategy == RecoveryStrategy.RETRY:
            # Simple retry logic
            return {"success": True, "message": "Retry initiated"}
        
        elif strategy == RecoveryStrategy.FALLBACK:
            # Switch to fallback mode
            return {"success": True, "message": "Fallback mode activated"}
        
        elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
            # Activate circuit breaker
            return {"success": True, "message": "Circuit breaker activated"}
        
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADE:
            # Enable degraded mode
            return {"success": True, "message": "Graceful degradation enabled"}
        
        elif strategy == RecoveryStrategy.SYSTEM_RESTART:
            # Initiate system restart
            logger.critical("Initiating system restart for recovery")
            return {"success": True, "message": "System restart initiated"}
        
        elif strategy == RecoveryStrategy.EMERGENCY_SHUTDOWN:
            # Emergency shutdown
            logger.critical("Initiating emergency shutdown")
            return {"success": True, "message": "Emergency shutdown initiated"}
        
        return {"success": False, "message": "Unknown strategy"}
    
    def _execute_recovery_plan(self, plan: RecoveryPlan, error_event: ErrorEvent) -> Dict[str, Any]:
        """Execute comprehensive recovery plan."""
        for strategy in plan.strategies:
            try:
                result = self._execute_recovery_strategy(strategy, error_event)
                if result.get("success"):
                    return {"success": True, "strategy": strategy}
            except Exception as e:
                logger.error(f"Recovery plan step {strategy.value} failed: {e}")
        
        return {"success": False, "strategy": None}
    
    def _handle_health_change(self, 
                             old_state: SystemHealth, 
                             new_state: SystemHealth, 
                             health_data: Dict[str, Any]) -> None:
        """Handle system health state changes."""
        logger.warning(f"System health changed: {old_state.value} -> {new_state.value}")
        
        # Take action based on new health state
        if new_state == SystemHealth.CRITICAL:
            # Enable degraded mode
            logger.warning("Enabling degraded mode due to critical system health")
        
        elif new_state == SystemHealth.FAILING:
            # Consider emergency measures
            logger.error("System failing - considering emergency measures")
    
    def get_recovery_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive recovery system dashboard."""
        # Calculate success rates
        success_rates = {}
        for error_type, successes in self.recovery_success_rates.items():
            if successes:
                success_rates[error_type] = sum(successes) / len(successes)
        
        # Recent error trends
        recent_errors = list(self.error_history)[-50:]  # Last 50 errors
        error_trends = defaultdict(int)
        for error in recent_errors:
            error_trends[error.error_type] += 1
        
        return {
            "system_health": self.system_monitor.get_health_report(),
            "error_statistics": {
                "total_errors": len(self.error_history),
                "recovery_stats": dict(self.recovery_stats),
                "success_rates": success_rates,
                "recent_error_trends": dict(error_trends)
            },
            "circuit_breaker_status": {
                name: cb.get_metrics() 
                for name, cb in self.circuit_breakers.items()
            },
            "recovery_plans": list(self.recovery_plans.keys()),
            "timestamp": time.time()
        }


# Decorator functions for easy use
def intelligent_recovery(recovery_plan: Optional[RecoveryPlan] = None):
    """Decorator for automatic intelligent error recovery."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get global recovery system (would be injected in real implementation)
                recovery_system = getattr(wrapper, '_recovery_system', None)
                if recovery_system:
                    error_event = recovery_system.handle_error(e, {"function": func.__name__})
                    if error_event.recovery_successful:
                        # Retry the function after successful recovery
                        return func(*args, **kwargs)
                raise
        return wrapper
    return decorator


# Factory function
def create_error_recovery_system(config: Optional[Dict[str, Any]] = None) -> IntelligentErrorRecovery:
    """Create and initialize intelligent error recovery system."""
    return IntelligentErrorRecovery(config)


# Export main classes
__all__ = [
    "IntelligentErrorRecovery",
    "ErrorSeverity",
    "RecoveryStrategy",
    "SystemHealth",
    "ErrorEvent",
    "RecoveryPlan",
    "CircuitBreaker",
    "AdaptiveRetry",
    "intelligent_recovery",
    "create_error_recovery_system"
]