
import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable[[], bool]
    description: str
    timeout: float = 5.0
    critical: bool = True

@dataclass
class HealthStatus:
    """Health status result."""
    name: str
    status: str  # "healthy", "unhealthy", "timeout"
    message: str
    timestamp: float
    duration: float
    critical: bool

class HealthMonitor:
    """Advanced health monitoring system."""
    
    def __init__(self):
        self.checks: List[HealthCheck] = []
        self.last_results: Dict[str, HealthStatus] = {}
        self.metrics = {
            "uptime": time.time(),
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "response_times": [],
        }
        self._lock = threading.Lock()
    
    def add_check(self, name: str, check_function: Callable[[], bool], 
                  description: str, timeout: float = 5.0, critical: bool = True):
        """Add a health check."""
        check = HealthCheck(name, check_function, description, timeout, critical)
        self.checks.append(check)
    
    def run_checks(self) -> Dict[str, HealthStatus]:
        """Run all health checks."""
        results = {}
        
        for check in self.checks:
            start_time = time.time()
            
            try:
                # Run check with timeout
                result = self._run_with_timeout(check.check_function, check.timeout)
                duration = time.time() - start_time
                
                if result:
                    status = HealthStatus(
                        name=check.name,
                        status="healthy",
                        message="Check passed",
                        timestamp=start_time,
                        duration=duration,
                        critical=check.critical
                    )
                else:
                    status = HealthStatus(
                        name=check.name,
                        status="unhealthy",
                        message="Check failed",
                        timestamp=start_time,
                        duration=duration,
                        critical=check.critical
                    )
            
            except Exception as e:
                duration = time.time() - start_time
                status = HealthStatus(
                    name=check.name,
                    status="unhealthy",
                    message=f"Check error: {str(e)}",
                    timestamp=start_time,
                    duration=duration,
                    critical=check.critical
                )
            
            results[check.name] = status
        
        with self._lock:
            self.last_results = results
        
        return results
    
    def _run_with_timeout(self, func: Callable, timeout: float) -> Any:
        """Run function with timeout."""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Check timed out after {timeout} seconds")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    def record_request(self, response_time: float, success: bool = True):
        """Record request metrics."""
        with self._lock:
            self.metrics["total_requests"] += 1
            
            if success:
                self.metrics["successful_requests"] += 1
            else:
                self.metrics["failed_requests"] += 1
            
            # Update response time metrics
            self.metrics["response_times"].append(response_time)
            
            # Keep only last 1000 response times
            if len(self.metrics["response_times"]) > 1000:
                self.metrics["response_times"] = self.metrics["response_times"][-1000:]
            
            # Calculate average
            if self.metrics["response_times"]:
                self.metrics["avg_response_time"] = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        with self._lock:
            overall_status = "healthy"
            critical_failures = []
            
            for name, result in self.last_results.items():
                if result.status != "healthy" and result.critical:
                    overall_status = "unhealthy"
                    critical_failures.append(name)
            
            return {
                "status": overall_status,
                "timestamp": time.time(),
                "uptime": time.time() - self.metrics["uptime"],
                "checks": {name: asdict(result) for name, result in self.last_results.items()},
                "metrics": self.metrics.copy(),
                "critical_failures": critical_failures
            }
    
    def save_health_report(self, filepath: str = "health_report.json"):
        """Save health status to file."""
        status = self.get_status()
        with open(filepath, 'w') as f:
            json.dump(status, f, indent=2)

def basic_memory_check() -> bool:
    """Basic memory usage check."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent < 90  # Less than 90% memory usage
    except ImportError:
        # If psutil not available, assume healthy
        return True

def basic_disk_check() -> bool:
    """Basic disk space check."""
    try:
        import psutil
        disk = psutil.disk_usage('/')
        return (disk.free / disk.total) > 0.1  # More than 10% free space
    except ImportError:
        # If psutil not available, assume healthy
        return True

def basic_service_check() -> bool:
    """Basic service availability check."""
    # Simple check - if we can execute this function, service is running
    return True

# Global health monitor with basic checks
health_monitor = HealthMonitor()
health_monitor.add_check("memory", basic_memory_check, "Memory usage check")
health_monitor.add_check("disk", basic_disk_check, "Disk space check")
health_monitor.add_check("service", basic_service_check, "Service availability check")
