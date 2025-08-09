"""Advanced auto-scaling with predictive algorithms and resource optimization."""

import asyncio
import logging
import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil

logger = logging.getLogger(__name__)

class ScalingDirection(Enum):
    """Scaling directions."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"  # Scale based on current metrics
    PREDICTIVE = "predictive"  # Scale based on predicted load
    SCHEDULED = "scheduled"  # Scale based on schedule
    HYBRID = "hybrid"  # Combination of strategies

class ResourceType(Enum):
    """Types of resources to scale."""
    CPU_CORES = "cpu_cores"
    MEMORY_GB = "memory_gb"
    WORKER_PROCESSES = "worker_processes"
    THREAD_POOL_SIZE = "thread_pool_size"
    CONNECTION_POOL = "connection_pool"

@dataclass
class ScalingMetrics:
    """Current system metrics for scaling decisions."""
    timestamp: datetime
    cpu_usage: float  # Percentage
    memory_usage: float  # Percentage
    request_rate: float  # Requests per second
    response_time_ms: float  # Average response time
    queue_depth: int  # Number of queued requests
    error_rate: float  # Percentage of errors
    active_connections: int = 0
    throughput: float = 0.0  # Requests processed per second
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for analysis."""
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'request_rate': self.request_rate,
            'response_time_ms': self.response_time_ms,
            'queue_depth': self.queue_depth,
            'error_rate': self.error_rate,
            'active_connections': self.active_connections,
            'throughput': self.throughput
        }

@dataclass
class ScalingRule:
    """Auto-scaling rule definition."""
    name: str
    metric: str
    threshold_up: float
    threshold_down: float
    scale_up_by: int
    scale_down_by: int
    cooldown_seconds: int = 300
    min_instances: int = 1
    max_instances: int = 10
    evaluation_periods: int = 2
    comparison: str = "gt"  # gt, lt, eq
    enabled: bool = True
    last_action: Optional[datetime] = None

@dataclass
class PredictionModel:
    """Simple prediction model for load forecasting."""
    lookback_window: int = 60  # minutes
    seasonal_periods: List[int] = field(default_factory=lambda: [24, 168])  # hours, week
    alpha: float = 0.3  # Exponential smoothing factor
    beta: float = 0.1  # Trend factor
    gamma: float = 0.2  # Seasonal factor
    
    def __post_init__(self):
        self.level = 0.0
        self.trend = 0.0
        self.seasonal: Dict[int, float] = {}
        self.history: deque = deque(maxlen=self.lookback_window * 60)  # 1-minute intervals

class LoadPredictor:
    """Advanced load prediction using multiple algorithms."""
    
    def __init__(self, model: PredictionModel):
        self.model = model
        self.metric_history: Dict[str, deque] = {}
        self._lock = threading.Lock()
    
    def add_datapoint(self, metrics: ScalingMetrics):
        """Add new datapoint to prediction model."""
        with self._lock:
            timestamp = metrics.timestamp
            
            for metric_name, value in metrics.to_dict().items():
                if metric_name not in self.metric_history:
                    self.metric_history[metric_name] = deque(maxlen=self.model.lookback_window * 60)
                
                self.metric_history[metric_name].append({
                    'timestamp': timestamp,
                    'value': value
                })
    
    def predict_load(self, metric: str, horizon_minutes: int = 30) -> List[Tuple[datetime, float]]:
        """Predict future load using Holt-Winters exponential smoothing."""
        if metric not in self.metric_history:
            return []
        
        with self._lock:
            data = list(self.metric_history[metric])
        
        if len(data) < 10:  # Need minimum data points
            return []
        
        # Extract values and apply Holt-Winters
        values = [point['value'] for point in data]
        timestamps = [point['timestamp'] for point in data]
        
        predictions = self._holt_winters_predict(values, horizon_minutes)
        
        # Generate future timestamps
        last_timestamp = timestamps[-1]
        future_timestamps = [
            last_timestamp + timedelta(minutes=i+1)
            for i in range(horizon_minutes)
        ]
        
        return list(zip(future_timestamps, predictions))
    
    def _holt_winters_predict(self, values: List[float], horizon: int) -> List[float]:
        """Holt-Winters exponential smoothing prediction."""
        if len(values) < 2:
            return [values[-1] if values else 0.0] * horizon
        
        # Initialize components
        level = values[0]
        trend = values[1] - values[0] if len(values) > 1 else 0.0
        seasonal = {}
        seasonal_period = min(24, len(values) // 2)  # 24-hour seasonality
        
        # Initialize seasonal components
        if seasonal_period > 0:
            for i in range(seasonal_period):
                seasonal[i] = 1.0
        
        # Apply Holt-Winters updating
        for i, value in enumerate(values[1:], 1):
            if seasonal_period > 0:
                seasonal_idx = i % seasonal_period
                if seasonal_idx in seasonal:
                    deseasonalized = value / seasonal[seasonal_idx]
                else:
                    deseasonalized = value
            else:
                deseasonalized = value
            
            # Update level and trend
            new_level = (self.model.alpha * deseasonalized + 
                        (1 - self.model.alpha) * (level + trend))
            new_trend = (self.model.beta * (new_level - level) + 
                        (1 - self.model.beta) * trend)
            
            level = new_level
            trend = new_trend
            
            # Update seasonal component
            if seasonal_period > 0 and seasonal_idx in seasonal:
                seasonal[seasonal_idx] = (
                    self.model.gamma * (value / new_level) +
                    (1 - self.model.gamma) * seasonal[seasonal_idx]
                )
        
        # Generate predictions
        predictions = []
        for h in range(1, horizon + 1):
            trend_component = trend * h
            if seasonal_period > 0:
                seasonal_idx = (len(values) - 1 + h) % seasonal_period
                seasonal_component = seasonal.get(seasonal_idx, 1.0)
            else:
                seasonal_component = 1.0
            
            prediction = (level + trend_component) * seasonal_component
            predictions.append(max(0, prediction))  # Ensure non-negative
        
        return predictions
    
    def detect_anomalies(self, metric: str, window_size: int = 20) -> List[Dict[str, Any]]:
        """Detect anomalies in metric data using statistical methods."""
        if metric not in self.metric_history:
            return []
        
        with self._lock:
            data = list(self.metric_history[metric])
        
        if len(data) < window_size:
            return []
        
        anomalies = []
        values = [point['value'] for point in data]
        
        # Use sliding window for anomaly detection
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            current_value = values[i]
            
            # Statistical anomaly detection
            mean = statistics.mean(window)
            stdev = statistics.stdev(window) if len(window) > 1 else 0
            
            if stdev > 0:
                z_score = abs(current_value - mean) / stdev
                if z_score > 2.5:  # 2.5 sigma threshold
                    anomalies.append({
                        'timestamp': data[i]['timestamp'],
                        'value': current_value,
                        'expected': mean,
                        'z_score': z_score,
                        'severity': 'high' if z_score > 3.0 else 'medium'
                    })
        
        return anomalies

class ResourceManager:
    """Manages scalable resources with dynamic allocation."""
    
    def __init__(self):
        self.resources: Dict[ResourceType, Dict[str, Any]] = {}
        self.scaling_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def register_resource(
        self,
        resource_type: ResourceType,
        current_value: int,
        min_value: int,
        max_value: int,
        scale_function: Callable[[int], bool]
    ):
        """Register a scalable resource."""
        with self._lock:
            self.resources[resource_type] = {
                'current': current_value,
                'min': min_value,
                'max': max_value,
                'scale_function': scale_function,
                'last_scaled': None
            }
    
    def scale_resource(
        self,
        resource_type: ResourceType,
        direction: ScalingDirection,
        amount: int
    ) -> bool:
        """Scale resource up or down."""
        with self._lock:
            if resource_type not in self.resources:
                logger.error(f"Resource type {resource_type} not registered")
                return False
            
            resource = self.resources[resource_type]
            current = resource['current']
            
            if direction == ScalingDirection.UP:
                new_value = min(current + amount, resource['max'])
            elif direction == ScalingDirection.DOWN:
                new_value = max(current - amount, resource['min'])
            else:
                return True  # Stable, no change needed
            
            if new_value == current:
                logger.info(f"Resource {resource_type} already at limit: {current}")
                return False
            
            # Execute scaling function
            try:
                if resource['scale_function'](new_value):
                    old_value = resource['current']
                    resource['current'] = new_value
                    resource['last_scaled'] = datetime.now()
                    
                    # Record scaling event
                    scaling_event = {
                        'timestamp': datetime.now(),
                        'resource_type': resource_type.value,
                        'direction': direction.value,
                        'old_value': old_value,
                        'new_value': new_value,
                        'amount': amount
                    }
                    self.scaling_history.append(scaling_event)
                    
                    logger.info(f"Scaled {resource_type.value} from {old_value} to {new_value}")
                    return True
                else:
                    logger.error(f"Failed to scale {resource_type.value} to {new_value}")
                    return False
            except Exception as e:
                logger.error(f"Error scaling {resource_type.value}: {e}")
                return False
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current status of all resources."""
        with self._lock:
            status = {}
            for resource_type, resource in self.resources.items():
                status[resource_type.value] = {
                    'current': resource['current'],
                    'min': resource['min'],
                    'max': resource['max'],
                    'utilization': resource['current'] / resource['max'] * 100,
                    'last_scaled': resource['last_scaled'].isoformat() if resource['last_scaled'] else None
                }
            return status

class AdvancedAutoScaler:
    """Advanced auto-scaler with predictive capabilities."""
    
    def __init__(
        self,
        strategy: ScalingStrategy = ScalingStrategy.HYBRID,
        prediction_horizon: int = 30  # minutes
    ):
        self.strategy = strategy
        self.prediction_horizon = prediction_horizon
        
        self.rules: List[ScalingRule] = []
        self.resource_manager = ResourceManager()
        self.load_predictor = LoadPredictor(PredictionModel())
        self.metrics_history: deque = deque(maxlen=1000)
        self.is_running = False
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add auto-scaling rule."""
        with self._lock:
            self.rules.append(rule)
        logger.info(f"Added scaling rule: {rule.name}")
    
    def record_metrics(self, metrics: ScalingMetrics):
        """Record new metrics for analysis."""
        with self._lock:
            self.metrics_history.append(metrics)
        
        # Update prediction model
        self.load_predictor.add_datapoint(metrics)
    
    def start_auto_scaling(self):
        """Start auto-scaling monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self._executor.submit(self._scaling_loop)
        logger.info("Auto-scaling started")
    
    def stop_auto_scaling(self):
        """Stop auto-scaling monitoring."""
        self.is_running = False
        self._executor.shutdown(wait=True)
        logger.info("Auto-scaling stopped")
    
    def _scaling_loop(self):
        """Main auto-scaling loop."""
        while self.is_running:
            try:
                if self.metrics_history:
                    current_metrics = self.metrics_history[-1]
                    scaling_decisions = self._make_scaling_decisions(current_metrics)
                    
                    for decision in scaling_decisions:
                        self._execute_scaling_decision(decision)
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _make_scaling_decisions(self, metrics: ScalingMetrics) -> List[Dict[str, Any]]:
        """Make scaling decisions based on strategy."""
        decisions = []
        
        if self.strategy in [ScalingStrategy.REACTIVE, ScalingStrategy.HYBRID]:
            decisions.extend(self._reactive_scaling_decisions(metrics))
        
        if self.strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]:
            decisions.extend(self._predictive_scaling_decisions(metrics))
        
        return decisions
    
    def _reactive_scaling_decisions(self, metrics: ScalingMetrics) -> List[Dict[str, Any]]:
        """Make reactive scaling decisions based on current metrics."""
        decisions = []
        
        with self._lock:
            active_rules = [rule for rule in self.rules if rule.enabled]
        
        for rule in active_rules:
            # Check cooldown period
            if (rule.last_action and 
                (datetime.now() - rule.last_action).total_seconds() < rule.cooldown_seconds):
                continue
            
            metric_value = getattr(metrics, rule.metric, None)
            if metric_value is None:
                continue
            
            # Evaluate scaling conditions
            if rule.comparison == "gt":
                scale_up = metric_value > rule.threshold_up
                scale_down = metric_value < rule.threshold_down
            elif rule.comparison == "lt":
                scale_up = metric_value < rule.threshold_up
                scale_down = metric_value > rule.threshold_down
            else:  # eq
                scale_up = metric_value >= rule.threshold_up
                scale_down = metric_value <= rule.threshold_down
            
            if scale_up:
                decisions.append({
                    'rule': rule.name,
                    'direction': ScalingDirection.UP,
                    'amount': rule.scale_up_by,
                    'reason': f"{rule.metric} ({metric_value}) > {rule.threshold_up}",
                    'confidence': min(1.0, (metric_value - rule.threshold_up) / rule.threshold_up)
                })
                rule.last_action = datetime.now()
            elif scale_down:
                decisions.append({
                    'rule': rule.name,
                    'direction': ScalingDirection.DOWN,
                    'amount': rule.scale_down_by,
                    'reason': f"{rule.metric} ({metric_value}) < {rule.threshold_down}",
                    'confidence': min(1.0, (rule.threshold_down - metric_value) / rule.threshold_down)
                })
                rule.last_action = datetime.now()
        
        return decisions
    
    def _predictive_scaling_decisions(self, metrics: ScalingMetrics) -> List[Dict[str, Any]]:
        """Make predictive scaling decisions based on forecasted load."""
        decisions = []
        
        try:
            # Get predictions for key metrics
            cpu_predictions = self.load_predictor.predict_load('cpu_usage', self.prediction_horizon)
            memory_predictions = self.load_predictor.predict_load('memory_usage', self.prediction_horizon)
            request_rate_predictions = self.load_predictor.predict_load('request_rate', self.prediction_horizon)
            
            # Analyze predictions for scaling needs
            if cpu_predictions:
                max_predicted_cpu = max(pred[1] for pred in cpu_predictions)
                if max_predicted_cpu > 80.0:  # Predicted high CPU usage
                    decisions.append({
                        'rule': 'predictive_cpu',
                        'direction': ScalingDirection.UP,
                        'amount': 1,
                        'reason': f"Predicted CPU usage: {max_predicted_cpu:.1f}%",
                        'confidence': min(1.0, max_predicted_cpu / 100.0)
                    })
            
            if memory_predictions:
                max_predicted_memory = max(pred[1] for pred in memory_predictions)
                if max_predicted_memory > 85.0:  # Predicted high memory usage
                    decisions.append({
                        'rule': 'predictive_memory',
                        'direction': ScalingDirection.UP,
                        'amount': 1,
                        'reason': f"Predicted memory usage: {max_predicted_memory:.1f}%",
                        'confidence': min(1.0, max_predicted_memory / 100.0)
                    })
            
        except Exception as e:
            logger.error(f"Predictive scaling error: {e}")
        
        return decisions
    
    def _execute_scaling_decision(self, decision: Dict[str, Any]):
        """Execute scaling decision."""
        try:
            # For now, we'll scale worker processes as an example
            resource_type = ResourceType.WORKER_PROCESSES
            success = self.resource_manager.scale_resource(
                resource_type,
                decision['direction'],
                decision['amount']
            )
            
            if success:
                logger.info(f"Executed scaling decision: {decision['rule']} - {decision['reason']}")
            else:
                logger.warning(f"Failed to execute scaling decision: {decision['rule']}")
        except Exception as e:
            logger.error(f"Error executing scaling decision: {e}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and statistics."""
        with self._lock:
            active_rules = len([rule for rule in self.rules if rule.enabled])
        
        recent_metrics = list(self.metrics_history)[-10:] if self.metrics_history else []
        
        return {
            'is_running': self.is_running,
            'strategy': self.strategy.value,
            'active_rules': active_rules,
            'total_rules': len(self.rules),
            'resource_status': self.resource_manager.get_resource_status(),
            'recent_scaling_events': self.resource_manager.scaling_history[-10:],
            'recent_metrics_count': len(recent_metrics),
            'prediction_horizon_minutes': self.prediction_horizon
        }

# Global auto-scaler instance
_global_auto_scaler = AdvancedAutoScaler()

def get_advanced_auto_scaler() -> AdvancedAutoScaler:
    """Get global advanced auto-scaler."""
    return _global_auto_scaler

# Example scaling functions
def scale_worker_processes(new_count: int) -> bool:
    """Example function to scale worker processes."""
    try:
        # This would integrate with your process manager (e.g., Gunicorn, uWSGI)
        logger.info(f"Scaling worker processes to {new_count}")
        return True
    except Exception as e:
        logger.error(f"Failed to scale worker processes: {e}")
        return False

def setup_default_scaling_rules():
    """Setup default auto-scaling rules."""
    scaler = get_advanced_auto_scaler()
    
    # Register resources
    scaler.resource_manager.register_resource(
        ResourceType.WORKER_PROCESSES,
        current_value=2,
        min_value=1,
        max_value=10,
        scale_function=scale_worker_processes
    )
    
    # CPU-based scaling
    cpu_rule = ScalingRule(
        name="cpu_scaling",
        metric="cpu_usage",
        threshold_up=70.0,
        threshold_down=30.0,
        scale_up_by=1,
        scale_down_by=1,
        cooldown_seconds=300,
        min_instances=1,
        max_instances=10
    )
    scaler.add_scaling_rule(cpu_rule)
    
    # Memory-based scaling
    memory_rule = ScalingRule(
        name="memory_scaling",
        metric="memory_usage",
        threshold_up=80.0,
        threshold_down=40.0,
        scale_up_by=1,
        scale_down_by=1,
        cooldown_seconds=300,
        min_instances=1,
        max_instances=10
    )
    scaler.add_scaling_rule(memory_rule)
    
    # Request rate scaling
    request_rule = ScalingRule(
        name="request_rate_scaling",
        metric="request_rate",
        threshold_up=50.0,  # requests per second
        threshold_down=10.0,
        scale_up_by=2,
        scale_down_by=1,
        cooldown_seconds=180,
        min_instances=1,
        max_instances=8
    )
    scaler.add_scaling_rule(request_rule)