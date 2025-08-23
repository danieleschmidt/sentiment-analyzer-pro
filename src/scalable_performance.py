
import time
import psutil
import threading
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, asdict
from functools import wraps
import gc
import sys

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    function_name: str
    execution_time: float
    memory_before: float
    memory_after: float
    memory_delta: float
    cpu_percent: float
    timestamp: float
    call_count: int = 1

class PerformanceProfiler:
    """Advanced performance profiler."""
    
    def __init__(self):
        self.metrics: Dict[str, List[PerformanceMetrics]] = {}
        self.function_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.enabled = True
    
    def profile(self, func: Callable) -> Callable:
        """Decorator for profiling function performance."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
            
            # Pre-execution measurements
            start_time = time.time()
            memory_before = self._get_memory_usage()
            cpu_before = psutil.cpu_percent()
            
            # Execute function
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                # Still record metrics for failed calls
                self._record_metrics(func.__name__, start_time, memory_before, cpu_before, True)
                raise e
            
            # Post-execution measurements
            self._record_metrics(func.__name__, start_time, memory_before, cpu_before, False)
            
            return result
        
        return wrapper
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    def _record_metrics(self, func_name: str, start_time: float, 
                       memory_before: float, cpu_before: float, failed: bool):
        """Record performance metrics."""
        execution_time = time.time() - start_time
        memory_after = self._get_memory_usage()
        memory_delta = memory_after - memory_before
        cpu_percent = psutil.cpu_percent() - cpu_before
        
        metrics = PerformanceMetrics(
            function_name=func_name,
            execution_time=execution_time,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_delta=memory_delta,
            cpu_percent=cpu_percent,
            timestamp=time.time()
        )
        
        with self._lock:
            if func_name not in self.metrics:
                self.metrics[func_name] = []
                self.function_stats[func_name] = {
                    'total_calls': 0,
                    'failed_calls': 0,
                    'total_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'avg_time': 0.0,
                    'total_memory_delta': 0.0,
                    'avg_memory_delta': 0.0
                }
            
            self.metrics[func_name].append(metrics)
            
            # Update statistics
            stats = self.function_stats[func_name]
            stats['total_calls'] += 1
            if failed:
                stats['failed_calls'] += 1
            
            stats['total_time'] += execution_time
            stats['min_time'] = min(stats['min_time'], execution_time)
            stats['max_time'] = max(stats['max_time'], execution_time)
            stats['avg_time'] = stats['total_time'] / stats['total_calls']
            
            stats['total_memory_delta'] += memory_delta
            stats['avg_memory_delta'] = stats['total_memory_delta'] / stats['total_calls']
            
            # Keep only last 1000 metrics per function
            if len(self.metrics[func_name]) > 1000:
                self.metrics[func_name] = self.metrics[func_name][-1000:]
    
    def get_stats(self, func_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if func_name:
                return {
                    func_name: self.function_stats.get(func_name, {})
                }
            return self.function_stats.copy()
    
    def get_slowest_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest functions by average execution time."""
        with self._lock:
            sorted_funcs = sorted(
                self.function_stats.items(),
                key=lambda x: x[1].get('avg_time', 0),
                reverse=True
            )
            
            return [
                {'function': name, **stats}
                for name, stats in sorted_funcs[:limit]
            ]
    
    def get_memory_hogs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get functions with highest memory usage."""
        with self._lock:
            sorted_funcs = sorted(
                self.function_stats.items(),
                key=lambda x: x[1].get('avg_memory_delta', 0),
                reverse=True
            )
            
            return [
                {'function': name, **stats}
                for name, stats in sorted_funcs[:limit]
            ]
    
    def reset_stats(self, func_name: Optional[str] = None):
        """Reset performance statistics."""
        with self._lock:
            if func_name:
                self.metrics.pop(func_name, None)
                self.function_stats.pop(func_name, None)
            else:
                self.metrics.clear()
                self.function_stats.clear()

class ResourceOptimizer:
    """System resource optimizer."""
    
    def __init__(self):
        self.gc_threshold = 100  # MB
        self.last_gc = time.time()
        self.gc_interval = 300  # 5 minutes
        self._monitor_thread = None
        self._stop_monitoring = False
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_monitoring = False
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._stop_monitoring = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring:
            try:
                self._check_memory()
                self._check_garbage_collection()
                time.sleep(30)  # Check every 30 seconds
            except Exception:
                pass
    
    def _check_memory(self):
        """Check and optimize memory usage."""
        try:
            memory_percent = psutil.virtual_memory().percent
            
            if memory_percent > 85:  # High memory usage
                # Force garbage collection
                collected = gc.collect()
                
                # Try to free up memory by clearing caches
                if hasattr(sys, 'intern'):
                    # Clear string intern cache (Python implementation detail)
                    pass
                
        except Exception:
            pass
    
    def _check_garbage_collection(self):
        """Check if garbage collection is needed."""
        current_time = time.time()
        
        if current_time - self.last_gc > self.gc_interval:
            try:
                # Get current memory usage
                memory_usage = self._get_memory_usage()
                
                if memory_usage > self.gc_threshold:
                    collected = gc.collect()
                    self.last_gc = current_time
                    
            except Exception:
                pass
    
    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def optimize_now(self):
        """Immediately perform optimization."""
        collected = gc.collect()
        return collected

class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str, profiler: Optional[PerformanceProfiler] = None):
        self.name = name
        self.profiler = profiler
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if self.profiler:
            # Create a dummy function for profiler
            def timed_operation():
                pass
            timed_operation.__name__ = self.name
            
            # Record metrics
            self.profiler._record_metrics(self.name, self.start_time, 0, 0, exc_type is not None)
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

# Global instances
performance_profiler = PerformanceProfiler()
resource_optimizer = ResourceOptimizer()

# Convenience decorators
def profile(func: Callable) -> Callable:
    """Convenience decorator for profiling."""
    return performance_profiler.profile(func)

def timed(name: Optional[str] = None):
    """Context manager for timing operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = name or f"{func.__module__}.{func.__name__}"
            with PerformanceTimer(operation_name, performance_profiler):
                return func(*args, **kwargs)
        return wrapper
    return decorator
