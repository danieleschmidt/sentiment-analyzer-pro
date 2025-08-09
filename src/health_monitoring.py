"""Advanced health monitoring with proactive alerting and self-healing."""

import asyncio
import logging
import psutil
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import json
import subprocess
import socket
import requests
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

class ComponentType(Enum):
    """Types of components to monitor."""
    DATABASE = "database"
    API_ENDPOINT = "api_endpoint"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM_RESOURCE = "system_resource"
    MODEL_SERVICE = "model_service"
    CACHE = "cache"
    QUEUE = "queue"

@dataclass
class HealthMetric:
    """Health metric data point."""
    timestamp: datetime
    component: str
    component_type: ComponentType
    metric_name: str
    value: float
    unit: str
    status: HealthStatus
    details: Optional[Dict[str, Any]] = None

@dataclass
class HealthThreshold:
    """Health monitoring thresholds."""
    warning_threshold: float
    critical_threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    sustained_duration: int = 60  # seconds

class HealthChecker:
    """Individual health checker for specific components."""
    
    def __init__(
        self,
        name: str,
        component_type: ComponentType,
        check_function: Callable[[], Dict[str, Any]],
        interval: int = 30,
        timeout: int = 10,
        thresholds: Optional[Dict[str, HealthThreshold]] = None
    ):
        self.name = name
        self.component_type = component_type
        self.check_function = check_function
        self.interval = interval
        self.timeout = timeout
        self.thresholds = thresholds or {}
        
        self.last_check: Optional[datetime] = None
        self.last_status = HealthStatus.HEALTHY
        self.consecutive_failures = 0
        self.metrics_history: List[HealthMetric] = []
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start health checking."""
        if self.is_running:
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._run_checks, daemon=True)
        self._thread.start()
        logger.info(f"Started health checker for {self.name}")
    
    def stop(self):
        """Stop health checking."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info(f"Stopped health checker for {self.name}")
    
    def _run_checks(self):
        """Run periodic health checks."""
        while self.is_running:
            try:
                self._perform_check()
            except Exception as e:
                logger.error(f"Health check failed for {self.name}: {e}")
                self._record_failure()
            
            time.sleep(self.interval)
    
    def _perform_check(self):
        """Perform single health check."""
        start_time = time.time()
        
        try:
            # Execute check with timeout
            result = self._execute_with_timeout(
                self.check_function, 
                self.timeout
            )
            
            check_duration = time.time() - start_time
            result['check_duration'] = check_duration
            
            # Process results
            status = self._evaluate_status(result)
            self._record_success(result, status)
            
        except Exception as e:
            logger.error(f"Health check error for {self.name}: {e}")
            self._record_failure(str(e))
    
    def _execute_with_timeout(self, func: Callable, timeout: int) -> Dict[str, Any]:
        """Execute function with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Health check timeout for {self.name}")
        
        # Set timeout alarm (Unix only)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            result = func()
            signal.alarm(0)  # Cancel alarm
            return result
        except AttributeError:
            # Windows doesn't support signal.SIGALRM
            return func()
    
    def _evaluate_status(self, result: Dict[str, Any]) -> HealthStatus:
        """Evaluate health status from check results."""
        overall_status = HealthStatus.HEALTHY
        
        for metric_name, value in result.items():
            if metric_name in self.thresholds and isinstance(value, (int, float)):
                threshold = self.thresholds[metric_name]
                
                if threshold.comparison == 'gt':
                    if value > threshold.critical_threshold:
                        overall_status = HealthStatus.CRITICAL
                    elif value > threshold.warning_threshold:
                        overall_status = max(overall_status, HealthStatus.WARNING, key=lambda x: x.value)
                elif threshold.comparison == 'lt':
                    if value < threshold.critical_threshold:
                        overall_status = HealthStatus.CRITICAL
                    elif value < threshold.warning_threshold:
                        overall_status = max(overall_status, HealthStatus.WARNING, key=lambda x: x.value)
        
        return overall_status
    
    def _record_success(self, result: Dict[str, Any], status: HealthStatus):
        """Record successful health check."""
        self.last_check = datetime.now()
        self.last_status = status
        self.consecutive_failures = 0
        
        # Create metrics
        for metric_name, value in result.items():
            if isinstance(value, (int, float)):
                metric = HealthMetric(
                    timestamp=self.last_check,
                    component=self.name,
                    component_type=self.component_type,
                    metric_name=metric_name,
                    value=value,
                    unit=self._get_metric_unit(metric_name),
                    status=status,
                    details=result
                )
                self.metrics_history.append(metric)
        
        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def _record_failure(self, error_message: str = ""):
        """Record failed health check."""
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= 3:
            self.last_status = HealthStatus.CRITICAL
        elif self.consecutive_failures >= 2:
            self.last_status = HealthStatus.DEGRADED
        else:
            self.last_status = HealthStatus.WARNING
        
        logger.warning(f"Health check failure #{self.consecutive_failures} for {self.name}: {error_message}")
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for metric."""
        unit_mapping = {
            'response_time': 'ms',
            'cpu_percent': '%',
            'memory_percent': '%',
            'disk_percent': '%',
            'check_duration': 's',
            'error_rate': '%',
            'throughput': 'req/s'
        }
        return unit_mapping.get(metric_name, 'unit')
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "name": self.name,
            "type": self.component_type.value,
            "status": self.last_status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "consecutive_failures": self.consecutive_failures,
            "is_running": self.is_running
        }

class SystemHealthMonitor:
    """System-level health monitoring."""
    
    @staticmethod
    def check_system_resources() -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024)
            }
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def check_network_connectivity() -> Dict[str, Any]:
        """Check network connectivity."""
        try:
            # Test DNS resolution
            socket.gethostbyname('google.com')
            
            # Test HTTP connectivity
            start_time = time.time()
            response = requests.get('https://httpbin.org/status/200', timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            return {
                "dns_resolution": True,
                "http_connectivity": response.status_code == 200,
                "response_time": response_time
            }
        except Exception as e:
            return {
                "dns_resolution": False,
                "http_connectivity": False,
                "error": str(e)
            }
    
    @staticmethod
    def check_disk_space() -> Dict[str, Any]:
        """Check available disk space."""
        try:
            disk_usage = psutil.disk_usage('/')
            return {
                "total_gb": disk_usage.total / (1024**3),
                "used_gb": disk_usage.used / (1024**3),
                "free_gb": disk_usage.free / (1024**3),
                "percent_used": (disk_usage.used / disk_usage.total) * 100
            }
        except Exception as e:
            return {"error": str(e)}

class AlertManager:
    """Alert management system."""
    
    def __init__(self):
        self.alert_channels: List[Callable[[Dict[str, Any]], None]] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.suppressed_alerts: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    def add_alert_channel(self, channel: Callable[[Dict[str, Any]], None]):
        """Add alert channel (e.g., email, Slack, webhook)."""
        self.alert_channels.append(channel)
    
    def send_alert(
        self,
        severity: str,
        title: str,
        message: str,
        component: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Send alert through all channels."""
        alert_id = f"{component}_{severity}_{int(time.time())}"
        
        # Check if alert is suppressed
        suppression_key = f"{component}_{severity}"
        with self._lock:
            if suppression_key in self.suppressed_alerts:
                if datetime.now() < self.suppressed_alerts[suppression_key]:
                    logger.debug(f"Alert suppressed: {suppression_key}")
                    return
                else:
                    del self.suppressed_alerts[suppression_key]
        
        alert_data = {
            "alert_id": alert_id,
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "title": title,
            "message": message,
            "component": component,
            "details": details or {}
        }
        
        # Send to all channels
        for channel in self.alert_channels:
            try:
                channel(alert_data)
            except Exception as e:
                logger.error(f"Failed to send alert through channel: {e}")
        
        # Record alert
        with self._lock:
            self.alert_history.append(alert_data)
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
        
        # Suppress similar alerts for 5 minutes
        with self._lock:
            self.suppressed_alerts[suppression_key] = datetime.now() + timedelta(minutes=5)
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        with self._lock:
            return self.alert_history[-limit:]

class SelfHealingSystem:
    """Self-healing system for automatic issue resolution."""
    
    def __init__(self):
        self.healing_actions: Dict[str, List[Callable]] = defaultdict(list)
        self.healing_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def register_healing_action(
        self,
        condition: str,
        action: Callable[[], bool],
        description: str
    ):
        """Register self-healing action."""
        action_wrapper = {
            'function': action,
            'description': description
        }
        self.healing_actions[condition].append(action_wrapper)
    
    def attempt_healing(
        self,
        component: str,
        issue: str,
        context: Dict[str, Any]
    ) -> bool:
        """Attempt to heal detected issue."""
        healing_key = f"{component}_{issue}"
        actions = self.healing_actions.get(healing_key, [])
        
        if not actions:
            logger.info(f"No healing actions available for {healing_key}")
            return False
        
        success = False
        for action in actions:
            try:
                logger.info(f"Attempting healing action: {action['description']}")
                if action['function']():
                    success = True
                    logger.info(f"Healing action successful: {action['description']}")
                    break
                else:
                    logger.warning(f"Healing action failed: {action['description']}")
            except Exception as e:
                logger.error(f"Healing action error: {e}")
        
        # Record healing attempt
        healing_record = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "issue": issue,
            "success": success,
            "context": context
        }
        
        with self._lock:
            self.healing_history.append(healing_record)
            if len(self.healing_history) > 500:
                self.healing_history = self.healing_history[-500:]
        
        return success

class HealthMonitoringSystem:
    """Main health monitoring system."""
    
    def __init__(self):
        self.health_checkers: Dict[str, HealthChecker] = {}
        self.alert_manager = AlertManager()
        self.self_healing = SelfHealingSystem()
        self.is_running = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def add_health_checker(self, checker: HealthChecker):
        """Add health checker."""
        self.health_checkers[checker.name] = checker
        logger.info(f"Added health checker: {checker.name}")
    
    def remove_health_checker(self, name: str):
        """Remove health checker."""
        if name in self.health_checkers:
            self.health_checkers[name].stop()
            del self.health_checkers[name]
            logger.info(f"Removed health checker: {name}")
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start all health checkers
        for checker in self.health_checkers.values():
            checker.start()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Health monitoring system started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_running = False
        
        # Stop all health checkers
        for checker in self.health_checkers.values():
            checker.stop()
        
        # Stop monitoring thread
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)
        
        logger.info("Health monitoring system stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                self._evaluate_overall_health()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _evaluate_overall_health(self):
        """Evaluate overall system health."""
        unhealthy_components = []
        degraded_components = []
        warning_components = []
        
        for checker in self.health_checkers.values():
            status = checker.last_status
            
            if status == HealthStatus.CRITICAL:
                unhealthy_components.append(checker.name)
                
                # Send alert
                self.alert_manager.send_alert(
                    severity="critical",
                    title=f"Component Critical: {checker.name}",
                    message=f"Health checker for {checker.name} reports critical status",
                    component=checker.name,
                    details=checker.get_current_status()
                )
                
                # Attempt self-healing
                self.self_healing.attempt_healing(
                    checker.name,
                    "critical_status",
                    checker.get_current_status()
                )
                
            elif status == HealthStatus.DEGRADED:
                degraded_components.append(checker.name)
            elif status == HealthStatus.WARNING:
                warning_components.append(checker.name)
        
        # Log overall status
        if unhealthy_components:
            logger.critical(f"Unhealthy components: {unhealthy_components}")
        elif degraded_components:
            logger.error(f"Degraded components: {degraded_components}")
        elif warning_components:
            logger.warning(f"Components with warnings: {warning_components}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        component_statuses = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, checker in self.health_checkers.items():
            status = checker.get_current_status()
            component_statuses[name] = status
            
            # Determine overall status (worst case)
            if checker.last_status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
            elif checker.last_status == HealthStatus.DEGRADED and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.DEGRADED
            elif checker.last_status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.WARNING
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "components": component_statuses,
            "total_components": len(self.health_checkers),
            "running_checkers": sum(1 for c in self.health_checkers.values() if c.is_running)
        }

# Global health monitoring system
_global_health_monitor = HealthMonitoringSystem()

def get_health_monitor() -> HealthMonitoringSystem:
    """Get global health monitoring system."""
    return _global_health_monitor

def setup_default_health_checkers():
    """Setup default health checkers."""
    monitor = get_health_monitor()
    
    # System resources checker
    system_checker = HealthChecker(
        name="system_resources",
        component_type=ComponentType.SYSTEM_RESOURCE,
        check_function=SystemHealthMonitor.check_system_resources,
        interval=30,
        thresholds={
            'cpu_percent': HealthThreshold(70.0, 90.0, 'gt'),
            'memory_percent': HealthThreshold(80.0, 95.0, 'gt'),
            'disk_percent': HealthThreshold(85.0, 95.0, 'gt')
        }
    )
    monitor.add_health_checker(system_checker)
    
    # Network connectivity checker
    network_checker = HealthChecker(
        name="network_connectivity",
        component_type=ComponentType.EXTERNAL_SERVICE,
        check_function=SystemHealthMonitor.check_network_connectivity,
        interval=60
    )
    monitor.add_health_checker(network_checker)