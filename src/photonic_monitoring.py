"""
Photonic-MLIR Bridge Monitoring and Observability Module

Provides comprehensive monitoring, metrics collection, health checks,
and observability features for the photonic circuit synthesis system.
"""

import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class HealthStatus(Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Metric:
    """Represents a single metric."""
    name: str
    metric_type: MetricType
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
            "description": self.description
        }


@dataclass
class HealthCheck:
    """Represents a health check result."""
    name: str
    status: HealthStatus
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health check to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "details": self.details
        }


class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()
        self._start_time = time.time()
    
    def counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None, 
                description: str = "") -> None:
        """Increment a counter metric."""
        with self._lock:
            labels = labels or {}
            metric_key = self._make_key(name, labels)
            
            if metric_key in self.metrics:
                self.metrics[metric_key].value += value
                self.metrics[metric_key].timestamp = time.time()
            else:
                self.metrics[metric_key] = Metric(
                    name=name,
                    metric_type=MetricType.COUNTER,
                    value=value,
                    labels=labels,
                    description=description
                )
            
            # Store in history
            self.metric_history[metric_key].append({
                "timestamp": time.time(),
                "value": self.metrics[metric_key].value
            })
    
    def gauge(self, name: str, value: float, labels: Dict[str, str] = None,
              description: str = "") -> None:
        """Set a gauge metric value."""
        with self._lock:
            labels = labels or {}
            metric_key = self._make_key(name, labels)
            
            self.metrics[metric_key] = Metric(
                name=name,
                metric_type=MetricType.GAUGE,
                value=value,
                labels=labels,
                description=description
            )
            
            # Store in history
            self.metric_history[metric_key].append({
                "timestamp": time.time(),
                "value": value
            })
    
    def histogram(self, name: str, value: float, labels: Dict[str, str] = None,
                  description: str = "") -> None:
        """Record a value in a histogram metric."""
        with self._lock:
            labels = labels or {}
            metric_key = self._make_key(name, labels)
            
            # Store individual measurements for histogram calculation
            self.metric_history[metric_key].append({
                "timestamp": time.time(),
                "value": value
            })
            
            # Calculate histogram statistics
            values = [entry["value"] for entry in self.metric_history[metric_key]]
            if values:
                histogram_stats = {
                    "count": len(values),
                    "sum": sum(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "p95": self._percentile(values, 95),
                    "p99": self._percentile(values, 99)
                }
                
                self.metrics[metric_key] = Metric(
                    name=name,
                    metric_type=MetricType.HISTOGRAM,
                    value=histogram_stats["mean"],
                    labels={**labels, **{f"hist_{k}": str(v) for k, v in histogram_stats.items()}},
                    description=description
                )
    
    def timer(self, name: str, labels: Dict[str, str] = None, description: str = ""):
        """Context manager for timing operations."""
        return TimerContext(self, name, labels or {}, description)
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all current metrics."""
        with self._lock:
            return [metric.to_dict() for metric in self.metrics.values()]
    
    def get_metric_history(self, name: str, labels: Dict[str, str] = None) -> List[Dict]:
        """Get historical data for a metric."""
        metric_key = self._make_key(name, labels or {})
        return list(self.metric_history[metric_key])
    
    def get_uptime_seconds(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self._start_time
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()
            self.metric_history.clear()
    
    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, 
                 labels: Dict[str, str], description: str):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.description = description
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.histogram(
                f"{self.name}_duration_seconds",
                duration,
                self.labels,
                f"{self.description} - Duration in seconds"
            )


class HealthChecker:
    """Manages health checks for system components."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self.last_results: Dict[str, HealthCheck] = {}
        self._lock = threading.RLock()
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheck]) -> None:
        """Register a health check function."""
        with self._lock:
            self.health_checks[name] = check_func
            logger.info(f"Registered health check: {name}")
    
    def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        if name not in self.health_checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found"
            )
        
        start_time = time.time()
        try:
            result = self.health_checks[name]()
            result.duration_ms = (time.time() - start_time) * 1000
            
            with self._lock:
                self.last_results[name] = result
            
            return result
        except Exception as e:
            error_result = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
            
            with self._lock:
                self.last_results[name] = error_result
            
            logger.error(f"Health check '{name}' failed: {e}")
            return error_result
    
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        for name in self.health_checks:
            results[name] = self.run_check(name)
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.last_results.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health check summary."""
        with self._lock:
            return {
                "overall_status": self.get_overall_status().value,
                "checks": {name: check.to_dict() for name, check in self.last_results.items()},
                "total_checks": len(self.health_checks),
                "last_updated": max([check.timestamp for check in self.last_results.values()]) 
                              if self.last_results else 0
            }


class PhotonicMonitor:
    """Main monitoring class for photonic synthesis system."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.health = HealthChecker()
        self._setup_default_health_checks()
        self._setup_system_metrics()
        logger.info("Photonic monitor initialized")
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        self.health.register_check("system", self._check_system_health)
        self.health.register_check("memory", self._check_memory_health)
        self.health.register_check("synthesis", self._check_synthesis_health)
    
    def _setup_system_metrics(self):
        """Setup system-level metrics."""
        self.metrics.gauge("system_start_time", self.metrics._start_time,
                          description="System start timestamp")
    
    def _check_system_health(self) -> HealthCheck:
        """Check overall system health."""
        try:
            uptime = self.metrics.get_uptime_seconds()
            
            if uptime < 60:  # Less than 1 minute
                status = HealthStatus.DEGRADED
                message = "System recently started"
            else:
                status = HealthStatus.HEALTHY
                message = f"System running for {uptime:.1f} seconds"
            
            return HealthCheck(
                name="system",
                status=status,
                message=message,
                details={"uptime_seconds": uptime}
            )
        except Exception as e:
            return HealthCheck(
                name="system",
                status=HealthStatus.UNHEALTHY,
                message=f"System check failed: {e}"
            )
    
    def _check_memory_health(self) -> HealthCheck:
        """Check memory usage health."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High memory usage: {memory.percent:.1f}%"
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Elevated memory usage: {memory.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory.percent:.1f}%"
            
            return HealthCheck(
                name="memory",
                status=status,
                message=message,
                details={
                    "percent": memory.percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                }
            )
        except ImportError:
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="psutil not available for memory monitoring"
            )
        except Exception as e:
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {e}"
            )
    
    def _check_synthesis_health(self) -> HealthCheck:
        """Check synthesis system health."""
        try:
            # Check if we can create a simple circuit
            from .photonic_mlir_bridge import PhotonicCircuitBuilder, SynthesisBridge
            
            start_time = time.time()
            builder = PhotonicCircuitBuilder("health_check")
            wg = builder.add_waveguide(1.0)
            circuit = builder.build()
            
            bridge = SynthesisBridge()
            result = bridge.synthesize_circuit(circuit)
            
            synthesis_time = time.time() - start_time
            
            if synthesis_time > 5.0:  # More than 5 seconds for simple circuit
                status = HealthStatus.DEGRADED
                message = f"Slow synthesis performance: {synthesis_time:.2f}s"
            else:
                status = HealthStatus.HEALTHY
                message = f"Synthesis working normally: {synthesis_time:.3f}s"
            
            return HealthCheck(
                name="synthesis",
                status=status,
                message=message,
                details={
                    "synthesis_time_seconds": synthesis_time,
                    "components_synthesized": result["components_count"],
                    "mlir_ir_size": len(result["mlir_ir"])
                }
            )
        except Exception as e:
            return HealthCheck(
                name="synthesis",
                status=HealthStatus.UNHEALTHY,
                message=f"Synthesis check failed: {e}"
            )
    
    def record_synthesis_operation(self, component_count: int, connection_count: int,
                                 synthesis_time: float, success: bool = True):
        """Record metrics for a synthesis operation."""
        # Update counters
        self.metrics.counter("synthesis_operations_total", 
                           labels={"status": "success" if success else "error"})
        
        # Update gauges
        self.metrics.gauge("synthesis_last_component_count", component_count)
        self.metrics.gauge("synthesis_last_connection_count", connection_count)
        
        # Update histograms
        self.metrics.histogram("synthesis_duration_seconds", synthesis_time,
                             labels={"component_range": self._get_component_range(component_count)})
        self.metrics.histogram("synthesis_component_count", component_count)
        self.metrics.histogram("synthesis_connection_count", connection_count)
        
        # Calculate throughput
        if synthesis_time > 0:
            throughput = component_count / synthesis_time
            self.metrics.histogram("synthesis_throughput_components_per_second", throughput)
    
    def record_validation_operation(self, component_count: int, validation_time: float,
                                  success: bool = True):
        """Record metrics for a validation operation."""
        self.metrics.counter("validation_operations_total",
                           labels={"status": "success" if success else "error"})
        self.metrics.histogram("validation_duration_seconds", validation_time)
        self.metrics.histogram("validation_component_count", component_count)
    
    def record_security_event(self, threat_type: str, severity: str = "medium"):
        """Record a security event."""
        self.metrics.counter("security_events_total",
                           labels={"threat_type": threat_type, "severity": severity})
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        return {
            "timestamp": time.time(),
            "uptime_seconds": self.metrics.get_uptime_seconds(),
            "metrics": self.metrics.get_metrics(),
            "health": self.health.get_health_summary(),
            "system_info": {
                "version": "1.0.0",
                "component": "photonic-mlir-bridge"
            }
        }
    
    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        lines.append("# HELP photonic_uptime_seconds System uptime in seconds")
        lines.append("# TYPE photonic_uptime_seconds gauge")
        lines.append(f"photonic_uptime_seconds {self.metrics.get_uptime_seconds()}")
        
        for metric in self.metrics.get_metrics():
            metric_name = f"photonic_{metric['name']}"
            
            # Add help comment
            if metric['description']:
                lines.append(f"# HELP {metric_name} {metric['description']}")
            
            # Add type comment
            lines.append(f"# TYPE {metric_name} {metric['type']}")
            
            # Add metric line
            if metric['labels']:
                label_str = ",".join(f'{k}="{v}"' for k, v in metric['labels'].items())
                lines.append(f"{metric_name}{{{label_str}}} {metric['value']}")
            else:
                lines.append(f"{metric_name} {metric['value']}")
        
        return "\n".join(lines)
    
    def save_metrics_to_file(self, filepath: Path):
        """Save current metrics to JSON file."""
        dashboard_data = self.get_monitoring_dashboard()
        with open(filepath, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
    
    def _get_component_range(self, count: int) -> str:
        """Get component count range for labeling."""
        if count <= 10:
            return "small"
        elif count <= 100:
            return "medium"
        elif count <= 1000:
            return "large"
        else:
            return "xlarge"


# Global monitor instance
_global_monitor = PhotonicMonitor()


def get_monitor() -> PhotonicMonitor:
    """Get the global monitor instance."""
    return _global_monitor


def record_synthesis_metrics(component_count: int, connection_count: int, 
                           synthesis_time: float, success: bool = True):
    """Convenience function to record synthesis metrics."""
    _global_monitor.record_synthesis_operation(
        component_count, connection_count, synthesis_time, success
    )


def record_validation_metrics(component_count: int, validation_time: float,
                            success: bool = True):
    """Convenience function to record validation metrics."""
    _global_monitor.record_validation_operation(component_count, validation_time, success)


def record_security_event(threat_type: str, severity: str = "medium"):
    """Convenience function to record security events."""
    _global_monitor.record_security_event(threat_type, severity)