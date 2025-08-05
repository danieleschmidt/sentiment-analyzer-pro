"""Auto-scaling and load balancing utilities for production deployment."""

import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import deque, defaultdict
import logging


logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics used for auto-scaling decisions."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_rate: float = 0.0
    response_time_ms: float = 0.0
    queue_depth: int = 0
    error_rate: float = 0.0


@dataclass 
class ScalingConfig:
    """Configuration for auto-scaling behavior."""
    # Scale up thresholds
    max_cpu_threshold: float = 70.0  # CPU % 
    max_memory_threshold: float = 80.0  # Memory %
    max_response_time_ms: float = 200.0
    max_request_rate: float = 100.0  # requests/second
    max_queue_depth: int = 50
    max_error_rate: float = 5.0  # %
    
    # Scale down thresholds (lower to prevent flapping)
    min_cpu_threshold: float = 30.0
    min_memory_threshold: float = 40.0
    min_response_time_ms: float = 50.0
    min_request_rate: float = 20.0
    min_queue_depth: int = 5
    
    # Scaling behavior
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    min_instances: int = 1
    max_instances: int = 10
    
    # Monitoring
    metrics_window: int = 60  # seconds
    check_interval: int = 30  # seconds


class AutoScaler:
    """Auto-scaling manager for sentiment analysis service."""
    
    def __init__(self, config: ScalingConfig = None):
        self.config = config or ScalingConfig()
        self.metrics_history: deque = deque(maxlen=100)
        self.last_scale_up: float = 0
        self.last_scale_down: float = 0
        self.current_instances: int = self.config.min_instances
        self.is_monitoring: bool = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        # Callbacks for scaling actions
        self.scale_up_callback: Optional[Callable[[int], None]] = None
        self.scale_down_callback: Optional[Callable[[int], None]] = None
        
    def set_scale_callbacks(self, 
                           scale_up: Callable[[int], None],
                           scale_down: Callable[[int], None]):
        """Set callbacks for scaling actions."""
        self.scale_up_callback = scale_up
        self.scale_down_callback = scale_down
        
    def record_metrics(self, metrics: ScalingMetrics):
        """Record current system metrics."""
        with self.lock:
            metrics.timestamp = time.time()
            self.metrics_history.append(metrics)
            
    def get_current_metrics(self) -> Optional[ScalingMetrics]:
        """Get the most recent metrics."""
        with self.lock:
            if self.metrics_history:
                return self.metrics_history[-1]
        return None
        
    def get_average_metrics(self, window_seconds: int = None) -> Optional[ScalingMetrics]:
        """Get average metrics over a time window."""
        if window_seconds is None:
            window_seconds = self.config.metrics_window
            
        with self.lock:
            if not self.metrics_history:
                return None
                
            cutoff_time = time.time() - window_seconds
            recent_metrics = [m for m in self.metrics_history 
                            if hasattr(m, 'timestamp') and m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return None
                
            # Calculate averages
            return ScalingMetrics(
                cpu_usage=sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
                memory_usage=sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
                request_rate=sum(m.request_rate for m in recent_metrics) / len(recent_metrics),
                response_time_ms=sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics),
                queue_depth=sum(m.queue_depth for m in recent_metrics) / len(recent_metrics),
                error_rate=sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
            )
    
    def should_scale_up(self, metrics: ScalingMetrics) -> bool:
        """Determine if scaling up is needed."""
        now = time.time()
        
        # Check cooldown period
        if now - self.last_scale_up < self.config.scale_up_cooldown:
            return False
            
        # Check if already at max instances
        if self.current_instances >= self.config.max_instances:
            return False
            
        # Check thresholds
        reasons = []
        if metrics.cpu_usage > self.config.max_cpu_threshold:
            reasons.append(f"CPU: {metrics.cpu_usage:.1f}%")
        if metrics.memory_usage > self.config.max_memory_threshold:
            reasons.append(f"Memory: {metrics.memory_usage:.1f}%")
        if metrics.response_time_ms > self.config.max_response_time_ms:
            reasons.append(f"Response time: {metrics.response_time_ms:.1f}ms")
        if metrics.request_rate > self.config.max_request_rate:
            reasons.append(f"Request rate: {metrics.request_rate:.1f}/s")
        if metrics.queue_depth > self.config.max_queue_depth:
            reasons.append(f"Queue depth: {metrics.queue_depth}")
        if metrics.error_rate > self.config.max_error_rate:
            reasons.append(f"Error rate: {metrics.error_rate:.1f}%")
            
        if reasons:
            logger.info(f"Scale up triggered: {', '.join(reasons)}")
            return True
            
        return False
    
    def should_scale_down(self, metrics: ScalingMetrics) -> bool:
        """Determine if scaling down is needed."""
        now = time.time()
        
        # Check cooldown period
        if now - self.last_scale_down < self.config.scale_down_cooldown:
            return False
            
        # Check if already at min instances
        if self.current_instances <= self.config.min_instances:
            return False
            
        # All thresholds must be below minimum for scale down
        if (metrics.cpu_usage < self.config.min_cpu_threshold and
            metrics.memory_usage < self.config.min_memory_threshold and
            metrics.response_time_ms < self.config.min_response_time_ms and
            metrics.request_rate < self.config.min_request_rate and
            metrics.queue_depth < self.config.min_queue_depth):
            
            logger.info(f"Scale down triggered: low resource usage")
            return True
            
        return False
    
    def scale_up(self):
        """Scale up the number of instances."""
        with self.lock:
            if self.current_instances < self.config.max_instances:
                new_instances = min(self.current_instances + 1, self.config.max_instances)
                old_instances = self.current_instances
                self.current_instances = new_instances
                self.last_scale_up = time.time()
                
                logger.info(f"Scaling up from {old_instances} to {new_instances} instances")
                
                if self.scale_up_callback:
                    try:
                        self.scale_up_callback(new_instances)
                    except Exception as e:
                        logger.error(f"Scale up callback failed: {e}")
    
    def scale_down(self):
        """Scale down the number of instances."""
        with self.lock:
            if self.current_instances > self.config.min_instances:
                new_instances = max(self.current_instances - 1, self.config.min_instances)
                old_instances = self.current_instances
                self.current_instances = new_instances
                self.last_scale_down = time.time()
                
                logger.info(f"Scaling down from {old_instances} to {new_instances} instances")
                
                if self.scale_down_callback:
                    try:
                        self.scale_down_callback(new_instances)
                    except Exception as e:
                        logger.error(f"Scale down callback failed: {e}")
    
    def evaluate_scaling(self):
        """Evaluate if scaling action is needed."""
        avg_metrics = self.get_average_metrics()
        if not avg_metrics:
            return
            
        if self.should_scale_up(avg_metrics):
            self.scale_up()
        elif self.should_scale_down(avg_metrics):
            self.scale_down()
    
    def start_monitoring(self):
        """Start the auto-scaling monitoring loop."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop the auto-scaling monitoring loop."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Auto-scaling monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                self.evaluate_scaling()
                time.sleep(self.config.check_interval)
            except Exception as e:
                logger.error(f"Error in auto-scaling monitoring: {e}")
                time.sleep(self.config.check_interval)
    
    def get_status(self) -> Dict:
        """Get current auto-scaling status."""
        with self.lock:
            avg_metrics = self.get_average_metrics()
            return {
                "current_instances": self.current_instances,
                "min_instances": self.config.min_instances,
                "max_instances": self.config.max_instances,
                "is_monitoring": self.is_monitoring,
                "last_scale_up": self.last_scale_up,
                "last_scale_down": self.last_scale_down,
                "current_metrics": avg_metrics.__dict__ if avg_metrics else None,
                "metrics_history_size": len(self.metrics_history)
            }


# Global auto-scaler instance
_auto_scaler: Optional[AutoScaler] = None


def get_auto_scaler() -> AutoScaler:
    """Get or create the global auto-scaler instance."""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler()
    return _auto_scaler


def init_auto_scaling(config: ScalingConfig = None,
                     scale_up_callback: Callable[[int], None] = None,
                     scale_down_callback: Callable[[int], None] = None):
    """Initialize auto-scaling with configuration and callbacks."""
    global _auto_scaler
    _auto_scaler = AutoScaler(config)
    
    if scale_up_callback or scale_down_callback:
        _auto_scaler.set_scale_callbacks(scale_up_callback, scale_down_callback)
    
    return _auto_scaler