"""
Health check implementations for the sentiment analyzer application.

This module provides comprehensive health checks for all application dependencies
and components, supporting both simple status checks and detailed diagnostics.
"""

import time
import logging
import psutil
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json
import os


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    response_time_ms: Optional[float] = None


class HealthChecker:
    """Comprehensive health checking for the application."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.checks: List[HealthCheck] = []
    
    def check_application_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive health checks and return detailed status.
        
        Returns:
            Dict containing overall health status and individual check results
        """
        start_time = time.time()
        self.checks = []
        
        # Core application checks
        self._check_memory_usage()
        self._check_cpu_usage()
        self._check_disk_space()
        self._check_model_availability()
        
        # Dependency checks
        self._check_redis_connection()
        self._check_database_connection()
        
        # External service checks
        self._check_external_dependencies()
        
        # Calculate overall status
        overall_status = self._calculate_overall_status()
        
        # Calculate total check time
        total_time_ms = (time.time() - start_time) * 1000
        
        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "total_checks": len(self.checks),
            "check_duration_ms": round(total_time_ms, 2),
            "version": os.getenv("APP_VERSION", "unknown"),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "checks": {check.name: self._serialize_check(check) for check in self.checks},
            "summary": self._generate_summary()
        }
    
    def _check_memory_usage(self) -> None:
        """Check system memory usage."""
        start_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High memory usage: {usage_percent:.1f}%"
            elif usage_percent > 75:
                status = HealthStatus.DEGRADED
                message = f"Elevated memory usage: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {usage_percent:.1f}%"
            
            details = {
                "usage_percent": round(usage_percent, 1),
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2)
            }
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check memory: {str(e)}"
            details = {"error": str(e)}
        
        response_time_ms = (time.time() - start_time) * 1000
        self.checks.append(HealthCheck("memory", status, message, details, response_time_ms))
    
    def _check_cpu_usage(self) -> None:
        """Check CPU usage over a short interval."""
        start_time = time.time()
        
        try:
            # Get CPU usage over 1 second interval
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent > 75:
                status = HealthStatus.DEGRADED
                message = f"Elevated CPU usage: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            details = {
                "usage_percent": round(cpu_percent, 1),
                "cpu_count": cpu_count,
                "load_average": [round(load, 2) for load in load_avg]
            }
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check CPU: {str(e)}"
            details = {"error": str(e)}
        
        response_time_ms = (time.time() - start_time) * 1000
        self.checks.append(HealthCheck("cpu", status, message, details, response_time_ms))
    
    def _check_disk_space(self) -> None:
        """Check disk space usage."""
        start_time = time.time()
        
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk usage: {usage_percent:.1f}%"
            elif usage_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {usage_percent:.1f}%"
            
            details = {
                "usage_percent": round(usage_percent, 1),
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2)
            }
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check disk: {str(e)}"
            details = {"error": str(e)}
        
        response_time_ms = (time.time() - start_time) * 1000
        self.checks.append(HealthCheck("disk", status, message, details, response_time_ms))
    
    def _check_model_availability(self) -> None:
        """Check if the ML model is loaded and accessible."""
        start_time = time.time()
        
        try:
            model_path = os.getenv("MODEL_PATH", "model.joblib")
            
            if os.path.exists(model_path):
                # Try to load model info (without actually loading the full model)
                file_size = os.path.getsize(model_path)
                mod_time = os.path.getmtime(model_path)
                
                status = HealthStatus.HEALTHY
                message = f"Model file available: {model_path}"
                details = {
                    "model_path": model_path,
                    "file_size_mb": round(file_size / (1024**2), 2),
                    "last_modified": time.ctime(mod_time),
                    "exists": True
                }
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Model file not found: {model_path}"
                details = {
                    "model_path": model_path,
                    "exists": False
                }
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check model: {str(e)}"
            details = {"error": str(e)}
        
        response_time_ms = (time.time() - start_time) * 1000
        self.checks.append(HealthCheck("model", status, message, details, response_time_ms))
    
    def _check_redis_connection(self) -> None:
        """Check Redis connection if configured."""
        start_time = time.time()
        
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            # Redis not configured, skip check
            return
        
        try:
            import redis
            r = redis.from_url(redis_url, socket_timeout=5)
            
            # Test basic operations
            r.ping()
            info = r.info()
            
            status = HealthStatus.HEALTHY
            message = "Redis connection healthy"
            details = {
                "connected": True,
                "version": info.get("redis_version"),
                "used_memory_mb": round(info.get("used_memory", 0) / (1024**2), 2),
                "connected_clients": info.get("connected_clients", 0)
            }
            
        except ImportError:
            status = HealthStatus.DEGRADED
            message = "Redis client not available"
            details = {"error": "redis package not installed"}
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Redis connection failed: {str(e)}"
            details = {"error": str(e), "connected": False}
        
        response_time_ms = (time.time() - start_time) * 1000
        self.checks.append(HealthCheck("redis", status, message, details, response_time_ms))
    
    def _check_database_connection(self) -> None:
        """Check database connection if configured."""
        start_time = time.time()
        
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            # Database not configured, skip check
            return
        
        try:
            import psycopg2
            from urllib.parse import urlparse
            
            parsed = urlparse(db_url)
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:],  # Remove leading slash
                user=parsed.username,
                password=parsed.password,
                connect_timeout=5
            )
            
            # Test basic query
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            status = HealthStatus.HEALTHY
            message = "Database connection healthy"
            details = {
                "connected": True,
                "version": version.split()[0] if version else "unknown"
            }
            
        except ImportError:
            status = HealthStatus.DEGRADED
            message = "Database client not available"
            details = {"error": "psycopg2 package not installed"}
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Database connection failed: {str(e)}"
            details = {"error": str(e), "connected": False}
        
        response_time_ms = (time.time() - start_time) * 1000
        self.checks.append(HealthCheck("database", status, message, details, response_time_ms))
    
    def _check_external_dependencies(self) -> None:
        """Check external service dependencies."""
        start_time = time.time()
        
        # Example: Check if we can reach external APIs
        external_endpoints = os.getenv("EXTERNAL_HEALTH_CHECKS", "").split(",")
        external_endpoints = [url.strip() for url in external_endpoints if url.strip()]
        
        if not external_endpoints:
            return
        
        try:
            failed_endpoints = []
            successful_endpoints = []
            
            for endpoint in external_endpoints:
                try:
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code == 200:
                        successful_endpoints.append(endpoint)
                    else:
                        failed_endpoints.append(f"{endpoint} ({response.status_code})")
                except Exception as e:
                    failed_endpoints.append(f"{endpoint} ({str(e)})")
            
            if failed_endpoints:
                status = HealthStatus.DEGRADED if successful_endpoints else HealthStatus.UNHEALTHY
                message = f"Some external dependencies unavailable: {len(failed_endpoints)}/{len(external_endpoints)}"
            else:
                status = HealthStatus.HEALTHY
                message = f"All external dependencies healthy: {len(successful_endpoints)}"
            
            details = {
                "total_endpoints": len(external_endpoints),
                "successful": successful_endpoints,
                "failed": failed_endpoints
            }
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check external dependencies: {str(e)}"
            details = {"error": str(e)}
        
        response_time_ms = (time.time() - start_time) * 1000
        self.checks.append(HealthCheck("external_deps", status, message, details, response_time_ms))
    
    def _calculate_overall_status(self) -> HealthStatus:
        """Calculate overall health status based on individual checks."""
        if not self.checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in self.checks]
        
        # If any check is unhealthy, overall is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        
        # If any check is degraded, overall is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # If any check is unknown, overall is degraded
        if HealthStatus.UNKNOWN in statuses:
            return HealthStatus.DEGRADED
        
        # All checks are healthy
        return HealthStatus.HEALTHY
    
    def _generate_summary(self) -> Dict[str, int]:
        """Generate summary of check results."""
        summary = {
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "unknown": 0
        }
        
        for check in self.checks:
            summary[check.status.value] += 1
        
        return summary
    
    def _serialize_check(self, check: HealthCheck) -> Dict[str, Any]:
        """Serialize a health check for JSON response."""
        return {
            "status": check.status.value,
            "message": check.message,
            "details": check.details or {},
            "response_time_ms": round(check.response_time_ms or 0, 2)
        }


# Global health checker instance
health_checker = HealthChecker()


def get_health_status() -> Dict[str, Any]:
    """Get current application health status."""
    return health_checker.check_application_health()


def get_simple_health_status() -> Dict[str, str]:
    """Get simplified health status for basic checks."""
    full_status = get_health_status()
    return {
        "status": full_status["status"],
        "timestamp": str(full_status["timestamp"]),
        "version": full_status["version"]
    }