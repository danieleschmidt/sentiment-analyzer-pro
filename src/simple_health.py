
import time
import psutil
from typing import Dict, Any

class SimpleHealthMonitor:
    """Basic health monitoring for Generation 1."""
    
    def __init__(self):
        self.start_time = time.time()
        self.requests_count = 0
        
    def get_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        return {
            "status": "healthy",
            "uptime": time.time() - self.start_time,
            "memory_usage": psutil.virtual_memory().percent if hasattr(psutil, 'virtual_memory') else 0,
            "requests_served": self.requests_count,
            "timestamp": time.time()
        }
    
    def record_request(self):
        """Record a request for monitoring."""
        self.requests_count += 1

# Global health monitor instance
health_monitor = SimpleHealthMonitor()
