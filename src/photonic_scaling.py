"""
Photonic-MLIR Bridge - Advanced Scaling and Performance System

This module provides comprehensive scaling capabilities including auto-scaling,
load balancing, distributed processing, and adaptive resource management.
"""

import time
import logging
import threading
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import multiprocessing as mp
import json
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategies for different workload patterns."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"


class LoadBalancingMethod(Enum):
    """Load balancing methods."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_BASED = "resource_based"
    RESPONSE_TIME = "response_time"


class ResourceType(Enum):
    """Resource types for monitoring and allocation."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    SYNTHESIS_THROUGHPUT = "synthesis_throughput"


@dataclass
class WorkerNode:
    """Represents a worker node in the scaling system."""
    id: str
    capacity: int
    current_load: int
    cpu_usage: float
    memory_usage: float
    synthesis_rate: float
    last_health_check: float
    status: str
    executor: Optional[ThreadPoolExecutor] = None
    process_pool: Optional[ProcessPoolExecutor] = None


@dataclass
class ScalingMetrics:
    """Scaling system metrics."""
    timestamp: float
    total_workers: int
    active_workers: int
    total_capacity: int
    current_load: int
    average_cpu_usage: float
    average_memory_usage: float
    synthesis_throughput: float
    queue_size: int
    response_times: List[float]


@dataclass
class ScalingEvent:
    """Scaling event log entry."""
    timestamp: float
    event_type: str
    details: Dict[str, Any]
    metrics_before: Optional[ScalingMetrics]
    metrics_after: Optional[ScalingMetrics]


class PhotonicScalingManager:
    """Advanced scaling manager for photonic synthesis operations."""
    
    def __init__(self, 
                 initial_workers: int = None,
                 max_workers: int = None,
                 scaling_strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE):
        
        # Determine optimal worker counts based on system resources
        cpu_count = mp.cpu_count()
        
        self.initial_workers = initial_workers or max(2, cpu_count // 2)
        self.max_workers = max_workers or min(32, cpu_count * 2)
        self.min_workers = max(1, self.initial_workers // 2)
        
        self.scaling_strategy = scaling_strategy
        self.load_balancing_method = LoadBalancingMethod.RESOURCE_BASED
        
        # Worker management
        self.workers: Dict[str, WorkerNode] = {}
        self.worker_counter = 0
        self.worker_lock = threading.RLock()
        
        # Task queue and management
        self.task_queue = asyncio.Queue()
        self.completed_tasks = []
        self.failed_tasks = []
        
        # Metrics and monitoring
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_events: List[ScalingEvent] = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Scaling parameters
        self.scale_up_threshold = 0.8  # Scale up when load > 80%
        self.scale_down_threshold = 0.3  # Scale down when load < 30%
        self.scale_up_cooldown = 30.0  # Seconds
        self.scale_down_cooldown = 60.0  # Seconds
        self.last_scale_action = 0.0
        
        # Performance optimization
        self.circuit_cache = {}
        self.result_cache = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 1000
        
        # Load prediction
        self.load_history = []
        self.max_history_size = 100
        
        self._initialize_scaling_system()
    
    def _initialize_scaling_system(self):
        """Initialize the scaling system."""
        logger.info(f"Initializing scaling system with {self.initial_workers} workers")
        
        # Create initial worker pool
        for i in range(self.initial_workers):
            self._add_worker()
        
        # Start monitoring
        self.start_monitoring()
        
        logger.info(f"Scaling system initialized with {len(self.workers)} workers")
    
    def _add_worker(self) -> str:
        """Add a new worker node."""
        with self.worker_lock:
            worker_id = f"worker_{self.worker_counter}"
            self.worker_counter += 1
            
            # Create worker node
            worker = WorkerNode(
                id=worker_id,
                capacity=10,  # Base capacity
                current_load=0,
                cpu_usage=0.0,
                memory_usage=0.0,
                synthesis_rate=0.0,
                last_health_check=time.time(),
                status="active",
                executor=ThreadPoolExecutor(max_workers=4),
                process_pool=ProcessPoolExecutor(max_workers=2)
            )
            
            self.workers[worker_id] = worker
            
            # Log scaling event
            self._log_scaling_event("worker_added", {
                "worker_id": worker_id,
                "total_workers": len(self.workers)
            })
            
            logger.info(f"Added worker {worker_id}")
            return worker_id
    
    def _remove_worker(self, worker_id: str):
        """Remove a worker node."""
        with self.worker_lock:
            if worker_id not in self.workers:
                return
            
            worker = self.workers[worker_id]
            
            # Gracefully shutdown worker
            if worker.executor:
                worker.executor.shutdown(wait=True)
            if worker.process_pool:
                worker.process_pool.shutdown(wait=True)
            
            del self.workers[worker_id]
            
            # Log scaling event
            self._log_scaling_event("worker_removed", {
                "worker_id": worker_id,
                "total_workers": len(self.workers)
            })
            
            logger.info(f"Removed worker {worker_id}")
    
    def _select_worker(self) -> Optional[str]:
        """Select optimal worker based on load balancing method."""
        if not self.workers:
            return None
        
        active_workers = {wid: w for wid, w in self.workers.items() if w.status == "active"}
        if not active_workers:
            return None
        
        if self.load_balancing_method == LoadBalancingMethod.ROUND_ROBIN:
            return self._round_robin_selection(active_workers)
        elif self.load_balancing_method == LoadBalancingMethod.LEAST_CONNECTIONS:
            return self._least_connections_selection(active_workers)
        elif self.load_balancing_method == LoadBalancingMethod.RESOURCE_BASED:
            return self._resource_based_selection(active_workers)
        elif self.load_balancing_method == LoadBalancingMethod.RESPONSE_TIME:
            return self._response_time_selection(active_workers)
        else:
            # Default to least connections
            return self._least_connections_selection(active_workers)
    
    def _round_robin_selection(self, workers: Dict[str, WorkerNode]) -> str:
        """Round-robin worker selection."""
        worker_ids = list(workers.keys())
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        selected = worker_ids[self._round_robin_index % len(worker_ids)]
        self._round_robin_index += 1
        return selected
    
    def _least_connections_selection(self, workers: Dict[str, WorkerNode]) -> str:
        """Select worker with least current load."""
        return min(workers.keys(), key=lambda wid: workers[wid].current_load)
    
    def _resource_based_selection(self, workers: Dict[str, WorkerNode]) -> str:
        """Select worker based on resource utilization."""
        def resource_score(worker):
            # Lower score is better
            return (worker.cpu_usage * 0.4 + 
                   worker.memory_usage * 0.3 + 
                   (worker.current_load / worker.capacity) * 0.3)
        
        return min(workers.keys(), key=lambda wid: resource_score(workers[wid]))
    
    def _response_time_selection(self, workers: Dict[str, WorkerNode]) -> str:
        """Select worker based on historical response times."""
        # For now, use synthesis rate as proxy for response time
        return max(workers.keys(), key=lambda wid: workers[wid].synthesis_rate)
    
    async def submit_synthesis_task(self, circuit, synthesis_params: Dict[str, Any] = None) -> str:
        """Submit a synthesis task for processing."""
        task_id = f"task_{int(time.time() * 1000000)}"
        
        # Check cache first
        cache_key = self._generate_cache_key(circuit, synthesis_params)
        if cache_key in self.result_cache:
            logger.debug(f"Cache hit for task {task_id}")
            return self.result_cache[cache_key]
        
        # Select worker
        worker_id = self._select_worker()
        if not worker_id:
            raise RuntimeError("No available workers")
        
        # Submit task
        task_data = {
            "id": task_id,
            "circuit": circuit,
            "synthesis_params": synthesis_params or {},
            "timestamp": time.time(),
            "worker_id": worker_id
        }
        
        await self.task_queue.put(task_data)
        
        # Execute task
        result = await self._execute_synthesis_task(task_data)
        
        # Cache result
        with self.cache_lock:
            if len(self.result_cache) < self.max_cache_size:
                self.result_cache[cache_key] = result
        
        return result
    
    async def _execute_synthesis_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a synthesis task on selected worker."""
        worker_id = task_data["worker_id"]
        worker = self.workers.get(worker_id)
        
        if not worker or worker.status != "active":
            raise RuntimeError(f"Worker {worker_id} not available")
        
        # Update worker load
        worker.current_load += 1
        
        try:
            # Execute synthesis
            start_time = time.time()
            
            # Use the photonic bridge for synthesis
            from .photonic_mlir_bridge import SynthesisBridge
            bridge = SynthesisBridge(enable_optimization=True)
            
            result = bridge.synthesize_circuit(task_data["circuit"])
            
            execution_time = time.time() - start_time
            
            # Update worker metrics
            worker.synthesis_rate = 1.0 / execution_time if execution_time > 0 else 0
            
            # Add task metadata
            result.update({
                "task_id": task_data["id"],
                "worker_id": worker_id,
                "execution_time": execution_time,
                "timestamp": time.time()
            })
            
            self.completed_tasks.append(task_data["id"])
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task_data['id']} failed on worker {worker_id}: {e}")
            self.failed_tasks.append(task_data["id"])
            raise
        
        finally:
            # Update worker load
            worker.current_load = max(0, worker.current_load - 1)
    
    def _generate_cache_key(self, circuit, synthesis_params: Dict[str, Any]) -> str:
        """Generate cache key for circuit and parameters."""
        import hashlib
        
        circuit_str = f"{circuit.name}_{len(circuit.components)}_{len(circuit.connections)}"
        params_str = json.dumps(synthesis_params, sort_keys=True)
        
        key_data = f"{circuit_str}_{params_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def start_monitoring(self):
        """Start scaling system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ScalingMonitor"
        )
        self.monitoring_thread.start()
        
        logger.info("Scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop scaling system monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Scaling monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring and scaling loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep metrics history manageable
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                # Update load history for prediction
                self.load_history.append(metrics.current_load / max(metrics.total_capacity, 1))
                if len(self.load_history) > self.max_history_size:
                    self.load_history = self.load_history[-self.max_history_size:]
                
                # Update worker health
                self._update_worker_health()
                
                # Make scaling decisions
                if self.scaling_strategy in (ScalingStrategy.DYNAMIC, ScalingStrategy.ADAPTIVE):
                    self._evaluate_scaling_decision(metrics)
                
                # Predictive scaling
                if self.scaling_strategy == ScalingStrategy.PREDICTIVE:
                    self._predictive_scaling(metrics)
                
                # Adaptive optimization
                if self.scaling_strategy == ScalingStrategy.ADAPTIVE:
                    self._adaptive_optimization(metrics)
                
                time.sleep(5.0)  # Monitoring interval
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10.0)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        current_time = time.time()
        
        # Worker metrics
        total_workers = len(self.workers)
        active_workers = sum(1 for w in self.workers.values() if w.status == "active")
        total_capacity = sum(w.capacity for w in self.workers.values())
        current_load = sum(w.current_load for w in self.workers.values())
        
        # Resource metrics
        avg_cpu = sum(w.cpu_usage for w in self.workers.values()) / max(total_workers, 1)
        avg_memory = sum(w.memory_usage for w in self.workers.values()) / max(total_workers, 1)
        
        # Synthesis throughput
        synthesis_rates = [w.synthesis_rate for w in self.workers.values() if w.synthesis_rate > 0]
        avg_synthesis_throughput = sum(synthesis_rates) / max(len(synthesis_rates), 1)
        
        # Queue metrics
        queue_size = self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0
        
        # Response times (simplified)
        response_times = []
        
        return ScalingMetrics(
            timestamp=current_time,
            total_workers=total_workers,
            active_workers=active_workers,
            total_capacity=total_capacity,
            current_load=current_load,
            average_cpu_usage=avg_cpu,
            average_memory_usage=avg_memory,
            synthesis_throughput=avg_synthesis_throughput,
            queue_size=queue_size,
            response_times=response_times
        )
    
    def _update_worker_health(self):
        """Update worker health and resource usage."""
        current_time = time.time()
        
        for worker in self.workers.values():
            try:
                # Update system resource usage
                worker.cpu_usage = psutil.cpu_percent(interval=None)
                worker.memory_usage = psutil.virtual_memory().percent
                worker.last_health_check = current_time
                
                # Check worker status
                if current_time - worker.last_health_check > 300:  # 5 minutes
                    worker.status = "unhealthy"
                else:
                    worker.status = "active"
                    
            except Exception as e:
                logger.warning(f"Failed to update health for worker {worker.id}: {e}")
                worker.status = "unknown"
    
    def _evaluate_scaling_decision(self, metrics: ScalingMetrics):
        """Evaluate whether to scale up or down."""
        current_time = time.time()
        
        # Check cooldown periods
        if current_time - self.last_scale_action < self.scale_up_cooldown:
            return
        
        # Calculate load ratio
        load_ratio = metrics.current_load / max(metrics.total_capacity, 1)
        
        # Scale up decision
        if (load_ratio > self.scale_up_threshold and 
            metrics.total_workers < self.max_workers):
            
            self._scale_up(metrics)
            self.last_scale_action = current_time
            
        # Scale down decision
        elif (load_ratio < self.scale_down_threshold and 
              metrics.total_workers > self.min_workers and
              current_time - self.last_scale_action > self.scale_down_cooldown):
            
            self._scale_down(metrics)
            self.last_scale_action = current_time
    
    def _scale_up(self, metrics_before: ScalingMetrics):
        """Scale up the system."""
        # Determine how many workers to add
        workers_to_add = min(2, self.max_workers - metrics_before.total_workers)
        
        logger.info(f"Scaling up: adding {workers_to_add} workers")
        
        for _ in range(workers_to_add):
            self._add_worker()
        
        # Collect metrics after scaling
        metrics_after = self._collect_metrics()
        
        # Log scaling event
        self._log_scaling_event("scale_up", {
            "workers_added": workers_to_add,
            "reason": "high_load",
            "load_ratio": metrics_before.current_load / max(metrics_before.total_capacity, 1)
        }, metrics_before, metrics_after)
    
    def _scale_down(self, metrics_before: ScalingMetrics):
        """Scale down the system."""
        # Determine how many workers to remove
        workers_to_remove = min(1, metrics_before.total_workers - self.min_workers)
        
        if workers_to_remove <= 0:
            return
        
        logger.info(f"Scaling down: removing {workers_to_remove} workers")
        
        # Select workers to remove (prefer least loaded)
        workers_by_load = sorted(
            self.workers.items(), 
            key=lambda x: x[1].current_load
        )
        
        for i in range(workers_to_remove):
            worker_id = workers_by_load[i][0]
            self._remove_worker(worker_id)
        
        # Collect metrics after scaling
        metrics_after = self._collect_metrics()
        
        # Log scaling event
        self._log_scaling_event("scale_down", {
            "workers_removed": workers_to_remove,
            "reason": "low_load",
            "load_ratio": metrics_before.current_load / max(metrics_before.total_capacity, 1)
        }, metrics_before, metrics_after)
    
    def _predictive_scaling(self, current_metrics: ScalingMetrics):
        """Implement predictive scaling based on load patterns."""
        if len(self.load_history) < 10:
            return
        
        # Simple trend analysis
        recent_loads = self.load_history[-10:]
        load_trend = (recent_loads[-1] - recent_loads[0]) / len(recent_loads)
        
        # Predict future load
        predicted_load = recent_loads[-1] + (load_trend * 3)  # 3 intervals ahead
        
        # Preemptive scaling
        if predicted_load > self.scale_up_threshold and current_metrics.total_workers < self.max_workers:
            logger.info(f"Predictive scale up: trend={load_trend:.3f}, predicted={predicted_load:.3f}")
            self._scale_up(current_metrics)
        elif predicted_load < self.scale_down_threshold and current_metrics.total_workers > self.min_workers:
            logger.info(f"Predictive scale down: trend={load_trend:.3f}, predicted={predicted_load:.3f}")
            self._scale_down(current_metrics)
    
    def _adaptive_optimization(self, metrics: ScalingMetrics):
        """Implement adaptive optimization strategies."""
        # Adjust thresholds based on system behavior
        if len(self.metrics_history) > 20:
            recent_metrics = self.metrics_history[-20:]
            
            # Calculate average response time proxy
            avg_throughput = sum(m.synthesis_throughput for m in recent_metrics) / len(recent_metrics)
            
            # Adapt thresholds based on performance
            if avg_throughput < 5.0:  # Low throughput
                self.scale_up_threshold = max(0.6, self.scale_up_threshold - 0.05)
            elif avg_throughput > 20.0:  # High throughput
                self.scale_up_threshold = min(0.9, self.scale_up_threshold + 0.05)
        
        # Adaptive load balancing
        if metrics.average_cpu_usage > 80:
            self.load_balancing_method = LoadBalancingMethod.RESOURCE_BASED
        elif metrics.synthesis_throughput < 10:
            self.load_balancing_method = LoadBalancingMethod.RESPONSE_TIME
        else:
            self.load_balancing_method = LoadBalancingMethod.LEAST_CONNECTIONS
    
    def _log_scaling_event(self, event_type: str, details: Dict[str, Any],
                          metrics_before: ScalingMetrics = None,
                          metrics_after: ScalingMetrics = None):
        """Log scaling event."""
        event = ScalingEvent(
            timestamp=time.time(),
            event_type=event_type,
            details=details,
            metrics_before=metrics_before,
            metrics_after=metrics_after
        )
        
        self.scaling_events.append(event)
        logger.info(f"Scaling event: {event_type} - {details}")
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        if not self.metrics_history:
            return {"no_data": True}
        
        current_metrics = self.metrics_history[-1]
        
        stats = {
            "timestamp": current_metrics.timestamp,
            "current_state": {
                "total_workers": current_metrics.total_workers,
                "active_workers": current_metrics.active_workers,
                "load_ratio": current_metrics.current_load / max(current_metrics.total_capacity, 1),
                "synthesis_throughput": current_metrics.synthesis_throughput,
                "average_cpu_usage": current_metrics.average_cpu_usage,
                "average_memory_usage": current_metrics.average_memory_usage
            },
            "scaling_config": {
                "strategy": self.scaling_strategy.value,
                "load_balancing": self.load_balancing_method.value,
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "scale_up_threshold": self.scale_up_threshold,
                "scale_down_threshold": self.scale_down_threshold
            },
            "performance": {
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "cache_size": len(self.result_cache),
                "cache_hit_rate": 0  # Would be calculated from cache statistics
            },
            "scaling_events": len(self.scaling_events),
            "recent_events": [
                {
                    "timestamp": event.timestamp,
                    "type": event.event_type,
                    "details": event.details
                }
                for event in self.scaling_events[-10:]
            ]
        }
        
        return stats
    
    def optimize_cache(self):
        """Optimize cache performance."""
        with self.cache_lock:
            if len(self.result_cache) > self.max_cache_size:
                # Remove oldest entries (simple LRU approximation)
                items_to_remove = len(self.result_cache) - self.max_cache_size
                keys_to_remove = list(self.result_cache.keys())[:items_to_remove]
                
                for key in keys_to_remove:
                    del self.result_cache[key]
                
                logger.info(f"Cache optimized: removed {items_to_remove} entries")
    
    def shutdown(self):
        """Gracefully shutdown the scaling system."""
        logger.info("Shutting down scaling system...")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Shutdown all workers
        with self.worker_lock:
            for worker_id in list(self.workers.keys()):
                self._remove_worker(worker_id)
        
        logger.info("Scaling system shutdown complete")


# Global scaling manager
_scaling_manager = None


def get_scaling_manager() -> PhotonicScalingManager:
    """Get or create global scaling manager."""
    global _scaling_manager
    if _scaling_manager is None:
        _scaling_manager = PhotonicScalingManager()
    return _scaling_manager


def scale_synthesis_operation(circuit, synthesis_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Scale a synthesis operation using the global scaling manager."""
    import asyncio
    
    scaling_manager = get_scaling_manager()
    
    # Run async operation in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            scaling_manager.submit_synthesis_task(circuit, synthesis_params)
        )
        return result
    finally:
        loop.close()


def get_scaling_stats() -> Dict[str, Any]:
    """Get global scaling statistics."""
    global _scaling_manager
    if _scaling_manager is None:
        return {"scaling_manager": "not_initialized"}
    
    return _scaling_manager.get_scaling_statistics()


if __name__ == "__main__":
    # Demo scaling capabilities
    print("⚡ Photonic-MLIR Bridge - Scaling System Demo")
    print("=" * 60)
    
    # Create scaling manager
    scaling_manager = PhotonicScalingManager(
        initial_workers=2,
        max_workers=8,
        scaling_strategy=ScalingStrategy.ADAPTIVE
    )
    
    print(f"Scaling manager initialized with {len(scaling_manager.workers)} workers")
    
    # Wait for monitoring to collect some metrics
    time.sleep(3)
    
    # Get statistics
    stats = scaling_manager.get_scaling_statistics()
    
    print(f"\nScaling Statistics:")
    print(f"Strategy: {stats['scaling_config']['strategy']}")
    print(f"Load Balancing: {stats['scaling_config']['load_balancing']}")
    print(f"Workers: {stats['current_state']['total_workers']}")
    print(f"Throughput: {stats['current_state']['synthesis_throughput']:.2f}")
    
    # Shutdown
    scaling_manager.shutdown()
    print("\n✅ Scaling system operational!")