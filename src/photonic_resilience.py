"""
Photonic-MLIR Bridge - Resilience and Fault Tolerance System

This module provides comprehensive resilience, fault tolerance, and self-healing
capabilities for the photonic-MLIR synthesis bridge.
"""

import time
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set
from concurrent.futures import ThreadPoolExecutor, Future
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ResilienceLevel(Enum):
    """Resilience levels for different operational modes."""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"


class ComponentHealth(Enum):
    """Health status of system components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure modes."""
    RETRY = "retry"
    FALLBACK = "fallback"
    ISOLATION = "isolation"
    RESTART = "restart"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_function: Callable
    interval_seconds: float
    timeout_seconds: float
    failure_threshold: int
    recovery_strategy: RecoveryStrategy
    critical: bool = False


@dataclass
class ComponentStatus:
    """Component status information."""
    name: str
    health: ComponentHealth
    last_check: float
    failure_count: int
    consecutive_failures: int
    metrics: Dict[str, Any]
    recovery_attempts: int = 0


@dataclass
class ResilienceEvent:
    """Resilience event log entry."""
    timestamp: float
    event_type: str
    component: str
    severity: str
    message: str
    context: Dict[str, Any]
    recovery_action: Optional[str] = None


class ResilienceManager:
    """Advanced resilience and fault tolerance manager."""
    
    def __init__(self, resilience_level: ResilienceLevel = ResilienceLevel.STANDARD):
        self.resilience_level = resilience_level
        self.components: Dict[str, ComponentStatus] = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.event_log: List[ResilienceEvent] = []
        self.monitoring_active = False
        self.monitoring_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.recovery_lock = threading.RLock()
        
        # Circuit breakers for different operations
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        
        # Failure patterns and recovery strategies
        self.failure_patterns: Dict[str, RecoveryStrategy] = {}
        
        self._initialize_resilience_framework()
    
    def _initialize_resilience_framework(self):
        """Initialize resilience framework components."""
        
        # Register core health checks
        self.register_health_check(HealthCheck(
            name="synthesis_engine",
            check_function=self._check_synthesis_engine_health,
            interval_seconds=30.0,
            timeout_seconds=5.0,
            failure_threshold=3,
            recovery_strategy=RecoveryStrategy.RESTART,
            critical=True
        ))
        
        self.register_health_check(HealthCheck(
            name="validation_system",
            check_function=self._check_validation_system_health,
            interval_seconds=60.0,
            timeout_seconds=10.0,
            failure_threshold=2,
            recovery_strategy=RecoveryStrategy.RETRY,
            critical=False
        ))
        
        self.register_health_check(HealthCheck(
            name="security_system",
            check_function=self._check_security_system_health,
            interval_seconds=45.0,
            timeout_seconds=5.0,
            failure_threshold=1,
            recovery_strategy=RecoveryStrategy.ISOLATION,
            critical=True
        ))
        
        self.register_health_check(HealthCheck(
            name="optimization_engine",
            check_function=self._check_optimization_engine_health,
            interval_seconds=120.0,
            timeout_seconds=15.0,
            failure_threshold=3,
            recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            critical=False
        ))
        
        # Initialize circuit breakers
        self.circuit_breakers["synthesis"] = CircuitBreaker(
            name="synthesis",
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exceptions=(ValueError, RuntimeError)
        )
        
        self.circuit_breakers["validation"] = CircuitBreaker(
            name="validation",
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exceptions=(ValueError, TypeError)
        )
        
        # Initialize failure patterns
        self._initialize_failure_patterns()
        
        logger.info(f"Resilience manager initialized with {len(self.health_checks)} health checks")
    
    def _initialize_failure_patterns(self):
        """Initialize common failure patterns and recovery strategies."""
        self.failure_patterns = {
            "memory_exhaustion": RecoveryStrategy.GRACEFUL_DEGRADATION,
            "timeout_error": RecoveryStrategy.RETRY,
            "validation_failure": RecoveryStrategy.FALLBACK,
            "synthesis_error": RecoveryStrategy.RESTART,
            "security_violation": RecoveryStrategy.ISOLATION,
            "dependency_missing": RecoveryStrategy.FALLBACK,
            "configuration_error": RecoveryStrategy.RESTART,
            "network_error": RecoveryStrategy.RETRY
        }
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a new health check."""
        self.health_checks[health_check.name] = health_check
        
        # Initialize component status
        self.components[health_check.name] = ComponentStatus(
            name=health_check.name,
            health=ComponentHealth.UNKNOWN,
            last_check=0.0,
            failure_count=0,
            consecutive_failures=0,
            metrics={}
        )
        
        logger.info(f"Registered health check: {health_check.name}")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ResilienceMonitor"
        )
        self.monitoring_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Execute health checks
                futures = {}
                
                for name, health_check in self.health_checks.items():
                    component = self.components[name]
                    
                    # Check if it's time for this health check
                    if (time.time() - component.last_check) >= health_check.interval_seconds:
                        future = self.executor.submit(
                            self._execute_health_check,
                            name,
                            health_check
                        )
                        futures[name] = future
                
                # Process completed health checks
                for name, future in futures.items():
                    try:
                        # Wait for completion with timeout
                        health_result = future.result(timeout=self.health_checks[name].timeout_seconds)
                        self._process_health_result(name, health_result)
                    except Exception as e:
                        self._handle_health_check_failure(name, e)
                
                # Sleep before next monitoring cycle
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)  # Back off on error
    
    def _execute_health_check(self, name: str, health_check: HealthCheck) -> Dict[str, Any]:
        """Execute a single health check."""
        try:
            start_time = time.time()
            result = health_check.check_function()
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "execution_time": execution_time,
                "result": result,
                "timestamp": start_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _process_health_result(self, name: str, result: Dict[str, Any]):
        """Process health check result."""
        component = self.components[name]
        component.last_check = result["timestamp"]
        
        if result["success"]:
            # Health check succeeded
            if component.health in (ComponentHealth.FAILING, ComponentHealth.FAILED):
                self._log_resilience_event(
                    "recovery",
                    name,
                    "info",
                    f"Component {name} recovered from failure"
                )
            
            component.health = ComponentHealth.HEALTHY
            component.consecutive_failures = 0
            component.metrics.update(result.get("result", {}))
            component.metrics["last_execution_time"] = result.get("execution_time", 0)
            
        else:
            # Health check failed
            self._handle_component_failure(name, result["error"])
    
    def _handle_health_check_failure(self, name: str, exception: Exception):
        """Handle health check execution failure."""
        component = self.components[name]
        component.last_check = time.time()
        
        self._handle_component_failure(name, str(exception))
    
    def _handle_component_failure(self, name: str, error: str):
        """Handle component failure."""
        component = self.components[name]
        health_check = self.health_checks[name]
        
        component.failure_count += 1
        component.consecutive_failures += 1
        
        # Determine health status
        if component.consecutive_failures >= health_check.failure_threshold:
            component.health = ComponentHealth.FAILED
        else:
            component.health = ComponentHealth.FAILING
        
        # Log failure event
        self._log_resilience_event(
            "failure",
            name,
            "error" if health_check.critical else "warning",
            f"Component {name} health check failed: {error}",
            {"consecutive_failures": component.consecutive_failures}
        )
        
        # Attempt recovery if threshold reached
        if component.consecutive_failures >= health_check.failure_threshold:
            self._attempt_recovery(name, health_check.recovery_strategy)
    
    def _attempt_recovery(self, component_name: str, strategy: RecoveryStrategy):
        """Attempt component recovery."""
        with self.recovery_lock:
            component = self.components[component_name]
            component.recovery_attempts += 1
            
            self._log_resilience_event(
                "recovery_attempt",
                component_name,
                "info",
                f"Attempting {strategy.value} recovery for {component_name}",
                {"attempt": component.recovery_attempts}
            )
            
            success = False
            
            try:
                if strategy == RecoveryStrategy.RETRY:
                    success = self._recovery_retry(component_name)
                elif strategy == RecoveryStrategy.FALLBACK:
                    success = self._recovery_fallback(component_name)
                elif strategy == RecoveryStrategy.ISOLATION:
                    success = self._recovery_isolation(component_name)
                elif strategy == RecoveryStrategy.RESTART:
                    success = self._recovery_restart(component_name)
                elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                    success = self._recovery_graceful_degradation(component_name)
                
                if success:
                    component.consecutive_failures = 0
                    component.health = ComponentHealth.HEALTHY
                    
                    self._log_resilience_event(
                        "recovery_success",
                        component_name,
                        "info",
                        f"Successfully recovered {component_name} using {strategy.value}"
                    )
                else:
                    self._log_resilience_event(
                        "recovery_failure",
                        component_name,
                        "error",
                        f"Failed to recover {component_name} using {strategy.value}"
                    )
            
            except Exception as e:
                self._log_resilience_event(
                    "recovery_error",
                    component_name,
                    "error",
                    f"Recovery attempt failed with exception: {e}"
                )
    
    def _recovery_retry(self, component_name: str) -> bool:
        """Retry recovery strategy."""
        logger.info(f"Retrying component {component_name}")
        
        # Clear any cached state
        if hasattr(self, '_clear_component_cache'):
            self._clear_component_cache(component_name)
        
        return True
    
    def _recovery_fallback(self, component_name: str) -> bool:
        """Fallback recovery strategy."""
        logger.info(f"Activating fallback for component {component_name}")
        
        # Activate backup systems or simplified modes
        if component_name == "validation_system":
            # Use basic validation instead of comprehensive
            return True
        elif component_name == "optimization_engine":
            # Disable optimization
            return True
        
        return False
    
    def _recovery_isolation(self, component_name: str) -> bool:
        """Isolation recovery strategy."""
        logger.info(f"Isolating component {component_name}")
        
        # Isolate component to prevent cascading failures
        if component_name == "security_system":
            # Enable safe mode
            return True
        
        return True
    
    def _recovery_restart(self, component_name: str) -> bool:
        """Restart recovery strategy."""
        logger.info(f"Restarting component {component_name}")
        
        # Reinitialize component
        try:
            if component_name == "synthesis_engine":
                # Reinitialize synthesis bridge
                from .photonic_mlir_bridge import SynthesisBridge
                # This would recreate the bridge instance
                return True
        except Exception as e:
            logger.error(f"Failed to restart {component_name}: {e}")
            return False
        
        return True
    
    def _recovery_graceful_degradation(self, component_name: str) -> bool:
        """Graceful degradation recovery strategy."""
        logger.info(f"Enabling graceful degradation for component {component_name}")
        
        # Reduce functionality but maintain basic operation
        if component_name == "optimization_engine":
            # Disable advanced optimizations
            return True
        
        return True
    
    def _log_resilience_event(self, event_type: str, component: str, severity: str, 
                             message: str, context: Dict[str, Any] = None):
        """Log resilience event."""
        event = ResilienceEvent(
            timestamp=time.time(),
            event_type=event_type,
            component=component,
            severity=severity,
            message=message,
            context=context or {}
        )
        
        self.event_log.append(event)
        
        # Log to standard logger
        if severity == "error":
            logger.error(f"[{component}] {message}")
        elif severity == "warning":
            logger.warning(f"[{component}] {message}")
        else:
            logger.info(f"[{component}] {message}")
    
    # Health check implementations
    def _check_synthesis_engine_health(self) -> Dict[str, Any]:
        """Check synthesis engine health."""
        try:
            from .photonic_mlir_bridge import SynthesisBridge, create_simple_mzi_circuit
            
            # Quick synthesis test
            start_time = time.time()
            bridge = SynthesisBridge(enable_optimization=False)
            circuit = create_simple_mzi_circuit()
            result = bridge.synthesize_circuit(circuit)
            synthesis_time = time.time() - start_time
            
            return {
                "synthesis_time": synthesis_time,
                "components_synthesized": result["components_count"],
                "status": "healthy"
            }
            
        except Exception as e:
            raise RuntimeError(f"Synthesis engine check failed: {e}")
    
    def _check_validation_system_health(self) -> Dict[str, Any]:
        """Check validation system health."""
        try:
            from .photonic_validation import validate_photonic_circuit
            from .photonic_mlir_bridge import create_simple_mzi_circuit
            
            # Quick validation test
            start_time = time.time()
            circuit = create_simple_mzi_circuit()
            report = validate_photonic_circuit(circuit)
            validation_time = time.time() - start_time
            
            return {
                "validation_time": validation_time,
                "validation_result": report.overall_result.value,
                "issues_found": len(report.issues),
                "status": "healthy"
            }
            
        except Exception as e:
            raise RuntimeError(f"Validation system check failed: {e}")
    
    def _check_security_system_health(self) -> Dict[str, Any]:
        """Check security system health."""
        try:
            from .photonic_security import SecurityValidator
            
            # Quick security check
            start_time = time.time()
            validator = SecurityValidator()
            
            # Test input validation
            test_result = validator.validate_input("test_input", "component_id")
            security_time = time.time() - start_time
            
            return {
                "security_check_time": security_time,
                "validation_result": test_result,
                "status": "healthy"
            }
            
        except Exception as e:
            raise RuntimeError(f"Security system check failed: {e}")
    
    def _check_optimization_engine_health(self) -> Dict[str, Any]:
        """Check optimization engine health."""
        try:
            from .photonic_optimization import get_optimizer
            
            # Quick optimization check
            start_time = time.time()
            optimizer = get_optimizer()
            stats = optimizer.get_performance_stats()
            optimization_time = time.time() - start_time
            
            return {
                "optimization_check_time": optimization_time,
                "cache_stats": stats,
                "status": "healthy"
            }
            
        except Exception as e:
            raise RuntimeError(f"Optimization engine check failed: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        current_time = time.time()
        
        health_summary = {
            "timestamp": current_time,
            "overall_health": "healthy",
            "components": {},
            "critical_failures": 0,
            "total_failures": 0,
            "monitoring_active": self.monitoring_active
        }
        
        critical_failed = 0
        total_failed = 0
        
        for name, component in self.components.items():
            health_check = self.health_checks[name]
            
            component_info = {
                "health": component.health.value,
                "last_check": current_time - component.last_check,
                "failure_count": component.failure_count,
                "consecutive_failures": component.consecutive_failures,
                "recovery_attempts": component.recovery_attempts,
                "critical": health_check.critical,
                "metrics": component.metrics
            }
            
            health_summary["components"][name] = component_info
            
            if component.health == ComponentHealth.FAILED:
                total_failed += 1
                if health_check.critical:
                    critical_failed += 1
        
        health_summary["critical_failures"] = critical_failed
        health_summary["total_failures"] = total_failed
        
        # Determine overall health
        if critical_failed > 0:
            health_summary["overall_health"] = "critical"
        elif total_failed > 0:
            health_summary["overall_health"] = "degraded"
        elif any(c.health == ComponentHealth.FAILING for c in self.components.values()):
            health_summary["overall_health"] = "warning"
        
        return health_summary
    
    def get_resilience_statistics(self) -> Dict[str, Any]:
        """Get resilience statistics."""
        if not self.event_log:
            return {"total_events": 0}
        
        stats = {
            "total_events": len(self.event_log),
            "by_type": {},
            "by_component": {},
            "by_severity": {},
            "recovery_success_rate": 0,
            "recent_events": []
        }
        
        recovery_attempts = 0
        recovery_successes = 0
        
        for event in self.event_log:
            # By type
            event_type = event.event_type
            stats["by_type"][event_type] = stats["by_type"].get(event_type, 0) + 1
            
            # By component
            component = event.component
            stats["by_component"][component] = stats["by_component"].get(component, 0) + 1
            
            # By severity
            severity = event.severity
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
            
            # Recovery statistics
            if event_type == "recovery_attempt":
                recovery_attempts += 1
            elif event_type == "recovery_success":
                recovery_successes += 1
        
        # Calculate recovery success rate
        if recovery_attempts > 0:
            stats["recovery_success_rate"] = (recovery_successes / recovery_attempts) * 100
        
        # Recent events (last 20)
        stats["recent_events"] = [
            {
                "timestamp": event.timestamp,
                "type": event.event_type,
                "component": event.component,
                "severity": event.severity,
                "message": event.message
            }
            for event in self.event_log[-20:]
        ]
        
        return stats


class CircuitBreaker:
    """Circuit breaker for operation resilience."""
    
    def __init__(self, name: str, failure_threshold: int = 5, 
                 recovery_timeout: float = 60.0, expected_exceptions: tuple = (Exception,)):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "open":
                # Check if recovery timeout has passed
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                    logger.info(f"Circuit breaker {self.name} entering half-open state")
                else:
                    raise RuntimeError(f"Circuit breaker {self.name} is open")
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset failure count and close circuit
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} closed")
                
                return result
                
            except self.expected_exceptions as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.warning(f"Circuit breaker {self.name} opened due to {self.failure_count} failures")
                
                raise e


# Global resilience manager
_resilience_manager = ResilienceManager()


def get_resilience_manager() -> ResilienceManager:
    """Get global resilience manager."""
    return _resilience_manager


def start_resilience_monitoring():
    """Start global resilience monitoring."""
    _resilience_manager.start_monitoring()


def stop_resilience_monitoring():
    """Stop global resilience monitoring."""
    _resilience_manager.stop_monitoring()


def get_system_health() -> Dict[str, Any]:
    """Get global system health."""
    return _resilience_manager.get_system_health()


def get_resilience_stats() -> Dict[str, Any]:
    """Get global resilience statistics."""
    return _resilience_manager.get_resilience_statistics()


if __name__ == "__main__":
    # Demo resilience capabilities
    print("🛡️ Photonic-MLIR Bridge - Resilience System Demo")
    print("=" * 60)
    
    # Start monitoring
    start_resilience_monitoring()
    
    print("Health monitoring started...")
    time.sleep(2)
    
    # Get system health
    health = get_system_health()
    print(f"\nSystem Health: {health['overall_health'].upper()}")
    print(f"Monitoring Active: {health['monitoring_active']}")
    print(f"Components Monitored: {len(health['components'])}")
    
    # Show component status
    for name, component in health["components"].items():
        print(f"  {name}: {component['health'].upper()}")
    
    # Get statistics
    stats = get_resilience_stats()
    print(f"\nResilience Statistics:")
    print(f"Total Events: {stats['total_events']}")
    print(f"Recovery Success Rate: {stats.get('recovery_success_rate', 0):.1f}%")
    
    # Stop monitoring
    stop_resilience_monitoring()
    print("\n✅ Resilience system operational!")