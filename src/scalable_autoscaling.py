
import time
import threading
import psutil
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import logging

class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"

@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    cpu_percent: float
    memory_percent: float
    active_requests: int
    response_time: float
    error_rate: float
    timestamp: float

@dataclass
class ScalingRule:
    """Auto-scaling rule definition."""
    metric_name: str
    threshold_up: float
    threshold_down: float
    cooldown_seconds: int
    scale_up_amount: int = 1
    scale_down_amount: int = 1

class AutoScaler:
    """Auto-scaling system for dynamic resource management."""
    
    def __init__(self, min_instances: int = 1, max_instances: int = 10):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        
        # Default scaling rules
        self.rules = [
            ScalingRule("cpu_percent", 80, 20, 300),  # CPU based
            ScalingRule("memory_percent", 85, 30, 300),  # Memory based
            ScalingRule("response_time", 2.0, 0.5, 180),  # Response time based
            ScalingRule("error_rate", 0.05, 0.01, 240),  # Error rate based
        ]
        
        self.metrics_history: List[ScalingMetrics] = []
        self.last_scaling_action = 0
        self.scaling_listeners: List[Callable] = []
        
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def add_scaling_listener(self, callback: Callable[[int, int, ScalingDirection], None]):
        """Add callback for scaling events."""
        self.scaling_listeners.append(callback)
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start auto-scaling monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Auto-scaling monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Make scaling decision
                direction = self._evaluate_scaling(metrics)
                
                # Execute scaling if needed
                if direction != ScalingDirection.NONE:
                    self._execute_scaling(direction)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in auto-scaling loop: {e}")
                time.sleep(interval_seconds)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect system and application metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
        except:
            cpu_percent = 0.0
            memory_percent = 0.0
        
        # These would typically come from application metrics
        active_requests = self._get_active_requests()
        response_time = self._get_avg_response_time()
        error_rate = self._get_error_rate()
        
        metrics = ScalingMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            active_requests=active_requests,
            response_time=response_time,
            error_rate=error_rate,
            timestamp=time.time()
        )
        
        with self._lock:
            self.metrics_history.append(metrics)
            # Keep only last 100 metrics
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    def _get_active_requests(self) -> int:
        """Get number of active requests (placeholder)."""
        # This would be implemented to get actual request metrics
        return 0
    
    def _get_avg_response_time(self) -> float:
        """Get average response time (placeholder)."""
        # This would be implemented to get actual response time metrics
        return 0.5
    
    def _get_error_rate(self) -> float:
        """Get error rate (placeholder)."""
        # This would be implemented to get actual error rate metrics
        return 0.0
    
    def _evaluate_scaling(self, current_metrics: ScalingMetrics) -> ScalingDirection:
        """Evaluate if scaling is needed based on current metrics."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_action < 60:  # 1 minute cooldown
            return ScalingDirection.NONE
        
        # Evaluate each rule
        scale_up_votes = 0
        scale_down_votes = 0
        
        for rule in self.rules:
            metric_value = getattr(current_metrics, rule.metric_name, 0)
            
            # Check if enough time has passed for this rule
            if current_time - self.last_scaling_action < rule.cooldown_seconds:
                continue
            
            if metric_value > rule.threshold_up:
                scale_up_votes += 1
            elif metric_value < rule.threshold_down:
                scale_down_votes += 1
        
        # Make decision based on votes
        if scale_up_votes > 0 and self.current_instances < self.max_instances:
            return ScalingDirection.UP
        elif scale_down_votes > 0 and self.current_instances > self.min_instances:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.NONE
    
    def _execute_scaling(self, direction: ScalingDirection):
        """Execute scaling action."""
        with self._lock:
            old_instances = self.current_instances
            
            if direction == ScalingDirection.UP:
                self.current_instances = min(self.current_instances + 1, self.max_instances)
            elif direction == ScalingDirection.DOWN:
                self.current_instances = max(self.current_instances - 1, self.min_instances)
            
            if self.current_instances != old_instances:
                self.last_scaling_action = time.time()
                
                # Notify listeners
                for callback in self.scaling_listeners:
                    try:
                        callback(old_instances, self.current_instances, direction)
                    except Exception as e:
                        self.logger.error(f"Error in scaling callback: {e}")
                
                self.logger.info(
                    f"Scaled {direction.value}: {old_instances} -> {self.current_instances} instances"
                )
    
    def manual_scale(self, target_instances: int) -> bool:
        """Manually scale to target number of instances."""
        if target_instances < self.min_instances or target_instances > self.max_instances:
            return False
        
        with self._lock:
            old_instances = self.current_instances
            self.current_instances = target_instances
            
            if old_instances != target_instances:
                direction = ScalingDirection.UP if target_instances > old_instances else ScalingDirection.DOWN
                self.last_scaling_action = time.time()
                
                # Notify listeners
                for callback in self.scaling_listeners:
                    try:
                        callback(old_instances, self.current_instances, direction)
                    except Exception as e:
                        self.logger.error(f"Error in scaling callback: {e}")
                
                self.logger.info(
                    f"Manual scale: {old_instances} -> {self.current_instances} instances"
                )
            
            return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        with self._lock:
            recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
            
            return {
                "current_instances": self.current_instances,
                "min_instances": self.min_instances,
                "max_instances": self.max_instances,
                "monitoring": self._monitoring,
                "last_scaling_action": self.last_scaling_action,
                "recent_metrics": [
                    {
                        "cpu_percent": m.cpu_percent,
                        "memory_percent": m.memory_percent,
                        "response_time": m.response_time,
                        "timestamp": m.timestamp
                    }
                    for m in recent_metrics
                ]
            }

class LoadBalancer:
    """Simple round-robin load balancer."""
    
    def __init__(self):
        self.servers: List[str] = []
        self.current_index = 0
        self._lock = threading.Lock()
    
    def add_server(self, server: str):
        """Add server to load balancer."""
        with self._lock:
            if server not in self.servers:
                self.servers.append(server)
    
    def remove_server(self, server: str):
        """Remove server from load balancer."""
        with self._lock:
            if server in self.servers:
                self.servers.remove(server)
                # Reset index if needed
                if self.current_index >= len(self.servers):
                    self.current_index = 0
    
    def get_next_server(self) -> Optional[str]:
        """Get next server using round-robin."""
        with self._lock:
            if not self.servers:
                return None
            
            server = self.servers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.servers)
            return server
    
    def get_servers(self) -> List[str]:
        """Get list of all servers."""
        with self._lock:
            return self.servers.copy()

# Global instances
auto_scaler = AutoScaler()
load_balancer = LoadBalancer()
