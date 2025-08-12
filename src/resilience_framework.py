"""Comprehensive Resilience Framework - Bulletproof error handling and recovery."""

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
import threading
import json
from pathlib import Path

import psutil
import numpy as np

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of system failures."""

    NETWORK = "network"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    MODEL = "model"
    DATA = "data"
    CONFIG = "config"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure types."""

    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADE = "graceful_degrade"
    FAIL_FAST = "fail_fast"
    ASYNC_RETRY = "async_retry"


@dataclass
class FailureEvent:
    """Represents a system failure event."""

    failure_type: FailureType
    component: str
    error_message: str
    timestamp: datetime
    stack_trace: str
    recovery_strategy: RecoveryStrategy
    context: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""

    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open
    success_count: int = 0


class ResilienceFramework:
    """Comprehensive resilience and error handling framework."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.failure_history: List[FailureEvent] = []
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.recovery_handlers: Dict[FailureType, Callable] = {}
        self.health_monitors: Dict[str, Callable] = {}
        self.fallback_strategies: Dict[str, Callable] = {}

        # Initialize monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.system_metrics: Dict[str, float] = {}

        # Setup default recovery handlers
        self._setup_default_handlers()

        # Setup health monitors
        self._setup_health_monitors()

        # Start monitoring
        self.start_monitoring()

    def _default_config(self) -> Dict[str, Any]:
        """Default resilience configuration."""
        return {
            "max_retries": 3,
            "retry_delay": 1.0,
            "exponential_backoff": True,
            "circuit_breaker_threshold": 5,
            "circuit_breaker_timeout": 60,
            "health_check_interval": 30,
            "memory_threshold_mb": 1000,
            "cpu_threshold_percent": 80,
            "disk_threshold_percent": 85,
            "enable_graceful_degradation": True,
            "enable_async_recovery": True,
            "log_failures": True,
            "persist_failure_history": True,
        }

    def _setup_default_handlers(self):
        """Setup default recovery handlers for each failure type."""
        self.recovery_handlers = {
            FailureType.NETWORK: self._handle_network_failure,
            FailureType.MEMORY: self._handle_memory_failure,
            FailureType.CPU: self._handle_cpu_failure,
            FailureType.DISK: self._handle_disk_failure,
            FailureType.MODEL: self._handle_model_failure,
            FailureType.DATA: self._handle_data_failure,
            FailureType.CONFIG: self._handle_config_failure,
            FailureType.EXTERNAL: self._handle_external_failure,
            FailureType.UNKNOWN: self._handle_unknown_failure,
        }

    def _setup_health_monitors(self):
        """Setup health monitoring functions."""
        self.health_monitors = {
            "memory": self._monitor_memory,
            "cpu": self._monitor_cpu,
            "disk": self._monitor_disk,
            "model_health": self._monitor_model_health,
            "data_quality": self._monitor_data_quality,
        }

    def start_monitoring(self):
        """Start system health monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Resilience monitoring started")

    def stop_monitoring(self):
        """Stop system health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("Resilience monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._run_health_checks()
                time.sleep(self.config["health_check_interval"])
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(10)  # Recovery delay

    def _run_health_checks(self):
        """Run all health checks."""
        for name, monitor in self.health_monitors.items():
            try:
                result = monitor()
                self.system_metrics[name] = result

                # Check for threshold violations
                self._check_thresholds(name, result)

            except Exception as e:
                logger.warning(f"Health check {name} failed: {e}")
                self._trigger_failure(FailureType.UNKNOWN, name, str(e))

    def _check_thresholds(self, metric_name: str, value: float):
        """Check if metrics violate thresholds."""
        thresholds = {
            "memory": self.config["memory_threshold_mb"],
            "cpu": self.config["cpu_threshold_percent"],
            "disk": self.config["disk_threshold_percent"],
        }

        if metric_name in thresholds and value > thresholds[metric_name]:
            failure_type = getattr(
                FailureType, metric_name.upper(), FailureType.UNKNOWN
            )
            self._trigger_failure(
                failure_type, metric_name, f"Threshold exceeded: {value}"
            )

    @contextmanager
    def resilient_operation(
        self, component: str, failure_type: FailureType = FailureType.UNKNOWN
    ):
        """Context manager for resilient operations."""
        try:
            yield
        except Exception as e:
            self._handle_failure(failure_type, component, e)
            raise

    @asynccontextmanager
    async def async_resilient_operation(
        self, component: str, failure_type: FailureType = FailureType.UNKNOWN
    ):
        """Async context manager for resilient operations."""
        try:
            yield
        except Exception as e:
            await self._handle_failure_async(failure_type, component, e)
            raise

    def with_retry(
        self,
        max_retries: Optional[int] = None,
        delay: Optional[float] = None,
        exponential_backoff: bool = True,
        failure_type: FailureType = FailureType.UNKNOWN,
    ):
        """Decorator for retry logic."""
        max_retries = max_retries or self.config["max_retries"]
        delay = delay or self.config["retry_delay"]

        def decorator(func):
            def wrapper(*args, **kwargs):
                last_exception = None

                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e

                        if attempt < max_retries:
                            wait_time = delay * (
                                2**attempt if exponential_backoff else 1
                            )
                            logger.warning(
                                f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s"
                            )
                            time.sleep(wait_time)
                        else:
                            self._trigger_failure(failure_type, func.__name__, str(e))

                raise last_exception

            return wrapper

        return decorator

    def with_circuit_breaker(
        self,
        component: str,
        failure_threshold: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        """Decorator for circuit breaker pattern."""
        failure_threshold = (
            failure_threshold or self.config["circuit_breaker_threshold"]
        )
        timeout = timeout or self.config["circuit_breaker_timeout"]

        def decorator(func):
            def wrapper(*args, **kwargs):
                breaker = self._get_circuit_breaker(component)

                # Check circuit state
                if breaker.state == "open":
                    if self._should_attempt_reset(breaker, timeout):
                        breaker.state = "half_open"
                        logger.info(
                            f"Circuit breaker {component} entering half-open state"
                        )
                    else:
                        raise Exception(f"Circuit breaker {component} is open")

                try:
                    result = func(*args, **kwargs)

                    # Success in half-open state
                    if breaker.state == "half_open":
                        breaker.success_count += 1
                        if breaker.success_count >= 3:  # Configurable threshold
                            self._close_circuit_breaker(component)

                    return result

                except Exception as e:
                    # Failure handling
                    self._record_circuit_failure(component, breaker)

                    if breaker.failure_count >= failure_threshold:
                        self._open_circuit_breaker(component, breaker)

                    raise

            return wrapper

        return decorator

    def with_fallback(self, fallback_func: Callable, component: str = "unknown"):
        """Decorator for fallback strategy."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(
                        f"Primary function {func.__name__} failed: {e}. Using fallback."
                    )
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        self._trigger_failure(
                            FailureType.EXTERNAL,
                            component,
                            f"Both primary and fallback failed: {e}, {fallback_error}",
                        )
                        raise

            return wrapper

        return decorator

    def with_graceful_degradation(
        self, degraded_func: Callable, component: str = "unknown"
    ):
        """Decorator for graceful degradation."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.config["enable_graceful_degradation"]:
                    return func(*args, **kwargs)

                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(
                        f"Function {func.__name__} failed: {e}. Degrading gracefully."
                    )
                    try:
                        return degraded_func(*args, **kwargs)
                    except Exception:
                        # If degraded function also fails, try to return safe default
                        return self._get_safe_default(func.__name__)

            return wrapper

        return decorator

    def _handle_failure(
        self, failure_type: FailureType, component: str, error: Exception
    ):
        """Handle synchronous failures."""
        self._trigger_failure(failure_type, component, str(error))

    async def _handle_failure_async(
        self, failure_type: FailureType, component: str, error: Exception
    ):
        """Handle asynchronous failures."""
        self._trigger_failure(failure_type, component, str(error))

        if self.config["enable_async_recovery"]:
            asyncio.create_task(self._async_recovery(failure_type, component))

    def _trigger_failure(
        self, failure_type: FailureType, component: str, error_message: str
    ):
        """Trigger failure handling workflow."""
        failure_event = FailureEvent(
            failure_type=failure_type,
            component=component,
            error_message=error_message,
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            recovery_strategy=self._determine_recovery_strategy(failure_type),
            context=self._get_failure_context(),
        )

        # Log failure
        if self.config["log_failures"]:
            logger.error(f"Failure in {component}: {error_message}")

        # Store failure
        self.failure_history.append(failure_event)

        # Execute recovery
        if failure_type in self.recovery_handlers:
            try:
                self.recovery_handlers[failure_type](failure_event)
            except Exception as e:
                logger.error(f"Recovery handler failed: {e}")

        # Persist if configured
        if self.config["persist_failure_history"]:
            self._persist_failure_event(failure_event)

    def _determine_recovery_strategy(
        self, failure_type: FailureType
    ) -> RecoveryStrategy:
        """Determine appropriate recovery strategy based on failure type."""
        strategies = {
            FailureType.NETWORK: RecoveryStrategy.RETRY,
            FailureType.MEMORY: RecoveryStrategy.GRACEFUL_DEGRADE,
            FailureType.CPU: RecoveryStrategy.GRACEFUL_DEGRADE,
            FailureType.DISK: RecoveryStrategy.FALLBACK,
            FailureType.MODEL: RecoveryStrategy.FALLBACK,
            FailureType.DATA: RecoveryStrategy.RETRY,
            FailureType.CONFIG: RecoveryStrategy.FAIL_FAST,
            FailureType.EXTERNAL: RecoveryStrategy.CIRCUIT_BREAK,
            FailureType.UNKNOWN: RecoveryStrategy.RETRY,
        }
        return strategies.get(failure_type, RecoveryStrategy.RETRY)

    def _get_failure_context(self) -> Dict[str, Any]:
        """Get current system context for failure analysis."""
        return {
            "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
            "cpu_percent": psutil.cpu_percent(),
            "disk_usage": psutil.disk_usage("/").percent,
            "load_average": psutil.getloadavg()
            if hasattr(psutil, "getloadavg")
            else None,
            "timestamp": datetime.now().isoformat(),
            "active_threads": threading.active_count(),
        }

    # Recovery handlers for each failure type
    def _handle_network_failure(self, failure_event: FailureEvent):
        """Handle network-related failures."""
        logger.info(f"Handling network failure: {failure_event.error_message}")
        # Implement network-specific recovery logic
        time.sleep(2)  # Brief delay for network recovery

    def _handle_memory_failure(self, failure_event: FailureEvent):
        """Handle memory-related failures."""
        logger.info(f"Handling memory failure: {failure_event.error_message}")
        # Force garbage collection
        import gc

        gc.collect()

        # Clear caches if available
        self._clear_caches()

    def _handle_cpu_failure(self, failure_event: FailureEvent):
        """Handle CPU-related failures."""
        logger.info(f"Handling CPU failure: {failure_event.error_message}")
        # Reduce processing load
        self._reduce_processing_load()

    def _handle_disk_failure(self, failure_event: FailureEvent):
        """Handle disk-related failures."""
        logger.info(f"Handling disk failure: {failure_event.error_message}")
        # Clean up temporary files
        self._cleanup_temp_files()

    def _handle_model_failure(self, failure_event: FailureEvent):
        """Handle model-related failures."""
        logger.info(f"Handling model failure: {failure_event.error_message}")
        # Fallback to simpler model
        self._switch_to_fallback_model()

    def _handle_data_failure(self, failure_event: FailureEvent):
        """Handle data-related failures."""
        logger.info(f"Handling data failure: {failure_event.error_message}")
        # Use cached or default data
        self._use_fallback_data()

    def _handle_config_failure(self, failure_event: FailureEvent):
        """Handle configuration-related failures."""
        logger.info(f"Handling config failure: {failure_event.error_message}")
        # Reset to default configuration
        self._reset_to_default_config()

    def _handle_external_failure(self, failure_event: FailureEvent):
        """Handle external service failures."""
        logger.info(f"Handling external failure: {failure_event.error_message}")
        # Use local alternatives
        self._use_local_alternatives()

    def _handle_unknown_failure(self, failure_event: FailureEvent):
        """Handle unknown failures."""
        logger.info(f"Handling unknown failure: {failure_event.error_message}")
        # Generic recovery approach
        self._generic_recovery()

    async def _async_recovery(self, failure_type: FailureType, component: str):
        """Perform asynchronous recovery operations."""
        await asyncio.sleep(5)  # Wait before recovery attempt

        try:
            # Attempt to restore component
            if hasattr(self, f"_restore_{component}"):
                restore_func = getattr(self, f"_restore_{component}")
                await restore_func()

            logger.info(f"Async recovery completed for {component}")

        except Exception as e:
            logger.error(f"Async recovery failed for {component}: {e}")

    # Health monitoring functions
    def _monitor_memory(self) -> float:
        """Monitor memory usage."""
        memory = psutil.virtual_memory()
        return memory.used / 1024 / 1024  # MB

    def _monitor_cpu(self) -> float:
        """Monitor CPU usage."""
        return psutil.cpu_percent(interval=1)

    def _monitor_disk(self) -> float:
        """Monitor disk usage."""
        return psutil.disk_usage("/").percent

    def _monitor_model_health(self) -> float:
        """Monitor model health (placeholder)."""
        # Implementation would check model response times, accuracy, etc.
        return 0.0  # Placeholder

    def _monitor_data_quality(self) -> float:
        """Monitor data quality (placeholder)."""
        # Implementation would check data freshness, completeness, etc.
        return 0.0  # Placeholder

    # Circuit breaker helpers
    def _get_circuit_breaker(self, component: str) -> CircuitBreakerState:
        """Get or create circuit breaker for component."""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreakerState()
        return self.circuit_breakers[component]

    def _should_attempt_reset(self, breaker: CircuitBreakerState, timeout: int) -> bool:
        """Check if circuit breaker should attempt reset."""
        if breaker.last_failure_time is None:
            return True

        return datetime.now() - breaker.last_failure_time > timedelta(seconds=timeout)

    def _record_circuit_failure(self, component: str, breaker: CircuitBreakerState):
        """Record circuit breaker failure."""
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now()
        breaker.success_count = 0

    def _open_circuit_breaker(self, component: str, breaker: CircuitBreakerState):
        """Open circuit breaker."""
        breaker.state = "open"
        logger.warning(
            f"Circuit breaker {component} opened after {breaker.failure_count} failures"
        )

    def _close_circuit_breaker(self, component: str):
        """Close circuit breaker."""
        breaker = self.circuit_breakers[component]
        breaker.state = "closed"
        breaker.failure_count = 0
        breaker.success_count = 0
        logger.info(f"Circuit breaker {component} closed")

    # Helper methods for recovery actions
    def _clear_caches(self):
        """Clear various caches to free memory."""
        # Implementation would clear application-specific caches
        logger.info("Clearing caches to free memory")

    def _reduce_processing_load(self):
        """Reduce CPU processing load."""
        # Implementation would reduce batch sizes, pause non-critical tasks, etc.
        logger.info("Reducing processing load")

    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        # Implementation would clean up temp directories
        logger.info("Cleaning up temporary files")

    def _switch_to_fallback_model(self):
        """Switch to a fallback model."""
        # Implementation would switch to a simpler, more reliable model
        logger.info("Switching to fallback model")

    def _use_fallback_data(self):
        """Use fallback data sources."""
        # Implementation would use cached or default data
        logger.info("Using fallback data")

    def _reset_to_default_config(self):
        """Reset to default configuration."""
        # Implementation would reset configuration to safe defaults
        logger.info("Resetting to default configuration")

    def _use_local_alternatives(self):
        """Use local alternatives for external services."""
        # Implementation would use local alternatives
        logger.info("Using local alternatives")

    def _generic_recovery(self):
        """Generic recovery approach for unknown failures."""
        # Implementation would perform generic recovery steps
        logger.info("Performing generic recovery")

    def _get_safe_default(self, function_name: str) -> Any:
        """Get safe default return value for failed function."""
        defaults = {
            "predict": "neutral",
            "classify": "unknown",
            "score": 0.0,
            "list": [],
            "dict": {},
            "string": "",
            "number": 0,
        }

        for key, value in defaults.items():
            if key in function_name.lower():
                return value

        return None  # Ultimate fallback

    def _persist_failure_event(self, failure_event: FailureEvent):
        """Persist failure event to disk."""
        try:
            failures_dir = Path("logs/failures")
            failures_dir.mkdir(parents=True, exist_ok=True)

            filename = f"failure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = failures_dir / filename

            # Convert to serializable format
            event_data = {
                "failure_type": failure_event.failure_type.value,
                "component": failure_event.component,
                "error_message": failure_event.error_message,
                "timestamp": failure_event.timestamp.isoformat(),
                "stack_trace": failure_event.stack_trace,
                "recovery_strategy": failure_event.recovery_strategy.value,
                "context": failure_event.context,
                "resolved": failure_event.resolved,
            }

            with open(filepath, "w") as f:
                json.dump(event_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to persist failure event: {e}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        return {
            "monitoring_active": self.monitoring_active,
            "metrics": self.system_metrics.copy(),
            "circuit_breakers": {
                name: {
                    "state": breaker.state,
                    "failure_count": breaker.failure_count,
                    "success_count": breaker.success_count,
                }
                for name, breaker in self.circuit_breakers.items()
            },
            "recent_failures": len(
                [
                    f
                    for f in self.failure_history
                    if (datetime.now() - f.timestamp).seconds < 3600
                ]
            ),
            "total_failures": len(self.failure_history),
        }

    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get failure statistics."""
        if not self.failure_history:
            return {"total_failures": 0}

        failure_types = {}
        components = {}
        recovery_strategies = {}

        for failure in self.failure_history:
            # Count by type
            failure_types[failure.failure_type.value] = (
                failure_types.get(failure.failure_type.value, 0) + 1
            )

            # Count by component
            components[failure.component] = components.get(failure.component, 0) + 1

            # Count by recovery strategy
            recovery_strategies[failure.recovery_strategy.value] = (
                recovery_strategies.get(failure.recovery_strategy.value, 0) + 1
            )

        recent_failures = [
            f
            for f in self.failure_history
            if (datetime.now() - f.timestamp).seconds < 86400
        ]  # Last 24 hours

        return {
            "total_failures": len(self.failure_history),
            "recent_failures": len(recent_failures),
            "failure_by_type": failure_types,
            "failure_by_component": components,
            "recovery_strategies": recovery_strategies,
            "average_recovery_time": self._calculate_average_recovery_time(),
        }

    def _calculate_average_recovery_time(self) -> float:
        """Calculate average recovery time for resolved failures."""
        resolved_failures = [
            f for f in self.failure_history if f.resolved and f.resolution_time
        ]

        if not resolved_failures:
            return 0.0

        total_time = sum(
            (f.resolution_time - f.timestamp).total_seconds() for f in resolved_failures
        )

        return total_time / len(resolved_failures)


# Global resilience framework instance
_resilience_framework = None


def get_resilience_framework() -> ResilienceFramework:
    """Get global resilience framework instance."""
    global _resilience_framework
    if _resilience_framework is None:
        _resilience_framework = ResilienceFramework()
    return _resilience_framework


def resilient(
    component: str = "unknown", failure_type: FailureType = FailureType.UNKNOWN
):
    """Decorator for making functions resilient."""
    framework = get_resilience_framework()
    return framework.resilient_operation(component, failure_type)


def with_retry(
    max_retries: int = 3, delay: float = 1.0, exponential_backoff: bool = True
):
    """Decorator for retry functionality."""
    framework = get_resilience_framework()
    return framework.with_retry(max_retries, delay, exponential_backoff)


def with_circuit_breaker(component: str, failure_threshold: int = 5, timeout: int = 60):
    """Decorator for circuit breaker pattern."""
    framework = get_resilience_framework()
    return framework.with_circuit_breaker(component, failure_threshold, timeout)


def with_fallback(fallback_func: Callable, component: str = "unknown"):
    """Decorator for fallback strategy."""
    framework = get_resilience_framework()
    return framework.with_fallback(fallback_func, component)


def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    framework = get_resilience_framework()
    return framework.get_system_health()


def get_failure_statistics() -> Dict[str, Any]:
    """Get failure statistics."""
    framework = get_resilience_framework()
    return framework.get_failure_statistics()
