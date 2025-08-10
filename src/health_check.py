"""
Core health check system for sentiment analyzer
Generation 1: Make It Work - Simple health monitoring
"""
import time
import psutil
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: str
    metrics: Dict[str, Any]
    timestamp: float

class HealthChecker:
    def __init__(self):
        self.checks: List[str] = []
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks"""
        self.checks = [
            "system_resources",
            "dependencies",
            "data_availability",
            "model_status"
        ]
    
    def check_system_resources(self) -> HealthCheckResult:
        """Check system CPU and memory usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            status = HealthStatus.HEALTHY
            if cpu_percent > 80 or memory.percent > 85:
                status = HealthStatus.WARNING
            if cpu_percent > 95 or memory.percent > 95:
                status = HealthStatus.CRITICAL
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=f"CPU: {cpu_percent}%, Memory: {memory.percent}%",
                metrics={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3)
                },
                timestamp=time.time()
            )
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Error checking resources: {str(e)}",
                metrics={},
                timestamp=time.time()
            )
    
    def check_dependencies(self) -> HealthCheckResult:
        """Check critical dependencies are available"""
        try:
            import numpy
            import pandas
            import sklearn
            import nltk
            
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.HEALTHY,
                message="All critical dependencies available",
                metrics={
                    "numpy_version": numpy.__version__,
                    "pandas_version": pandas.__version__,
                    "sklearn_version": sklearn.__version__,
                    "nltk_version": nltk.__version__
                },
                timestamp=time.time()
            )
        except ImportError as e:
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.CRITICAL,
                message=f"Missing dependency: {str(e)}",
                metrics={},
                timestamp=time.time()
            )
    
    def check_data_availability(self) -> HealthCheckResult:
        """Check if sample data is accessible"""
        try:
            import os
            data_path = "data/sample_reviews.csv"
            
            if os.path.exists(data_path):
                file_size = os.path.getsize(data_path)
                return HealthCheckResult(
                    name="data_availability",
                    status=HealthStatus.HEALTHY,
                    message=f"Sample data available ({file_size} bytes)",
                    metrics={"file_size": file_size, "data_path": data_path},
                    timestamp=time.time()
                )
            else:
                return HealthCheckResult(
                    name="data_availability",
                    status=HealthStatus.WARNING,
                    message="Sample data not found",
                    metrics={"data_path": data_path},
                    timestamp=time.time()
                )
        except Exception as e:
            return HealthCheckResult(
                name="data_availability",
                status=HealthStatus.CRITICAL,
                message=f"Error checking data: {str(e)}",
                metrics={},
                timestamp=time.time()
            )
    
    def check_model_status(self) -> HealthCheckResult:
        """Check if models can be built"""
        try:
            from src.models import build_nb_model
            from src.preprocessing import preprocess_text
            
            model = build_nb_model()
            test_text = preprocess_text("test")
            
            return HealthCheckResult(
                name="model_status",
                status=HealthStatus.HEALTHY,
                message="Model building and preprocessing functional",
                metrics={
                    "preprocessed_test": test_text,
                    "model_type": type(model).__name__
                },
                timestamp=time.time()
            )
        except Exception as e:
            return HealthCheckResult(
                name="model_status",
                status=HealthStatus.CRITICAL,
                message=f"Model check failed: {str(e)}",
                metrics={},
                timestamp=time.time()
            )
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        for check_name in self.checks:
            method_name = f"check_{check_name}"
            if hasattr(self, method_name):
                try:
                    result = getattr(self, method_name)()
                    results[check_name] = result
                    logger.info(f"Health check '{check_name}': {result.status.value}")
                except Exception as e:
                    results[check_name] = HealthCheckResult(
                        name=check_name,
                        status=HealthStatus.CRITICAL,
                        message=f"Check failed: {str(e)}",
                        metrics={},
                        timestamp=time.time()
                    )
                    logger.error(f"Health check '{check_name}' failed: {e}")
        
        return results
    
    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Get overall system health status"""
        if not results:
            return HealthStatus.CRITICAL
        
        statuses = [result.status for result in results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

def quick_health_check() -> Dict[str, Any]:
    """Perform a quick health check and return summary"""
    checker = HealthChecker()
    results = checker.run_all_checks()
    overall_status = checker.get_overall_status(results)
    
    return {
        "overall_status": overall_status.value,
        "checks": {name: {
            "status": result.status.value,
            "message": result.message,
            "metrics": result.metrics
        } for name, result in results.items()},
        "timestamp": time.time()
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    health_data = quick_health_check()
    print(f"Overall Status: {health_data['overall_status']}")
    for check_name, check_data in health_data['checks'].items():
        print(f"  {check_name}: {check_data['status']} - {check_data['message']}")