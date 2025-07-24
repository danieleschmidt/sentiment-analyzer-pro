"""Performance profiling utilities for Sentiment Analyzer Pro."""

import time
import psutil
import functools
import threading
from typing import Dict, Any, Callable, Optional, List
from contextlib import contextmanager
from collections import defaultdict, deque
import json
import logging

from .logging_config import get_logger, log_performance_metric
from .metrics import metrics_collector

logger = get_logger(__name__)

# Global profiling data storage
_profile_data = {
    'function_calls': defaultdict(list),
    'memory_usage': deque(maxlen=1000),
    'cpu_usage': deque(maxlen=1000),
    'io_stats': deque(maxlen=100),
    'slow_queries': deque(maxlen=50)
}
_profile_lock = threading.Lock()

# Configuration
PROFILING_ENABLED = True
SLOW_THRESHOLD = 0.5  # seconds
MEMORY_THRESHOLD = 100 * 1024 * 1024  # 100MB


class PerformanceProfiler:
    """Comprehensive performance profiler for the application."""
    
    def __init__(self):
        """Initialize the profiler."""
        self.enabled = PROFILING_ENABLED
        self.start_time = time.time()
        try:
            self.process = psutil.Process()
        except Exception as exc:
            logger.warning(f"Failed to initialize psutil process: {exc}")
            self.process = None
        self._monitoring_active = False
        self._monitor_thread = None
    
    def enable(self):
        """Enable profiling."""
        self.enabled = True
        logger.info("Performance profiling enabled")
    
    def disable(self):
        """Disable profiling."""
        self.enabled = False
        logger.info("Performance profiling disabled")
    
    def start_monitoring(self, interval: float = 5.0):
        """Start background monitoring of system metrics."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_system_metrics,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started system monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Stopped system monitoring")
    
    def _monitor_system_metrics(self, interval: float):
        """Background thread for monitoring system metrics."""
        while self._monitoring_active:
            try:
                if not self.process:
                    time.sleep(interval)
                    continue
                    
                # Memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # CPU usage
                cpu_percent = self.process.cpu_percent()
                
                # I/O stats
                io_counters = self.process.io_counters()
                
                with _profile_lock:
                    timestamp = time.time()
                    _profile_data['memory_usage'].append({
                        'timestamp': timestamp,
                        'rss_mb': memory_mb,
                        'vms_mb': memory_info.vms / 1024 / 1024
                    })
                    
                    _profile_data['cpu_usage'].append({
                        'timestamp': timestamp,
                        'cpu_percent': cpu_percent
                    })
                    
                    _profile_data['io_stats'].append({
                        'timestamp': timestamp,
                        'read_bytes': io_counters.read_bytes,
                        'write_bytes': io_counters.write_bytes
                    })
                
                # Update metrics collector
                metrics_collector.set_memory_usage('webapp', int(memory_mb * 1024 * 1024))
                
                # Check for high memory usage
                if memory_mb > MEMORY_THRESHOLD / 1024 / 1024:
                    logger.warning(f"High memory usage detected: {memory_mb:.1f}MB")
                
            except Exception as exc:
                logger.error(f"Error in system monitoring: {exc}")
            
            time.sleep(interval)
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get a summary of profiling data."""
        with _profile_lock:
            # Calculate function call statistics
            function_stats = {}
            for func_name, calls in _profile_data['function_calls'].items():
                if calls:
                    durations = [call['duration'] for call in calls]
                    function_stats[func_name] = {
                        'call_count': len(calls),
                        'total_time': sum(durations),
                        'avg_time': sum(durations) / len(durations),
                        'min_time': min(durations),
                        'max_time': max(durations),
                        'slow_calls': len([d for d in durations if d > SLOW_THRESHOLD])
                    }
            
            # Recent memory and CPU data
            recent_memory = list(_profile_data['memory_usage'])[-20:]
            recent_cpu = list(_profile_data['cpu_usage'])[-20:]
            
            # Current system state
            try:
                if self.process:
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    current_cpu = self.process.cpu_percent()
                else:
                    current_memory = 0
                    current_cpu = 0
            except:
                current_memory = 0
                current_cpu = 0
            
            return {
                'uptime_seconds': time.time() - self.start_time,
                'function_stats': function_stats,
                'current_memory_mb': current_memory,
                'current_cpu_percent': current_cpu,
                'recent_memory': recent_memory,
                'recent_cpu': recent_cpu,
                'slow_queries': list(_profile_data['slow_queries']),
                'profiling_enabled': self.enabled,
                'monitoring_active': self._monitoring_active
            }
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator to profile function execution."""
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                start_memory = None
                
                try:
                    if self.process:
                        start_memory = self.process.memory_info().rss
                except:
                    pass
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Memory delta
                    memory_delta = 0
                    try:
                        if start_memory and self.process:
                            end_memory = self.process.memory_info().rss
                            memory_delta = end_memory - start_memory
                    except:
                        pass
                    
                    # Record profiling data
                    call_data = {
                        'timestamp': start_time,
                        'duration': duration,
                        'memory_delta_bytes': memory_delta,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    }
                    
                    with _profile_lock:
                        _profile_data['function_calls'][name].append(call_data)
                        
                        # Keep only recent calls per function
                        if len(_profile_data['function_calls'][name]) > 100:
                            _profile_data['function_calls'][name] = \
                                _profile_data['function_calls'][name][-100:]
                    
                    # Log slow operations
                    if duration > SLOW_THRESHOLD:
                        slow_query_data = {
                            'function': name,
                            'duration': duration,
                            'timestamp': start_time,
                            'memory_delta_mb': memory_delta / 1024 / 1024 if memory_delta else 0
                        }
                        
                        with _profile_lock:
                            _profile_data['slow_queries'].append(slow_query_data)
                        
                        logger.warning(f"Slow operation detected: {name} took {duration:.3f}s")
                        log_performance_metric(logger, name, duration, {
                            'threshold_exceeded': True,
                            'memory_delta_mb': memory_delta / 1024 / 1024 if memory_delta else 0
                        })
                    
                    return result
                    
                except Exception as exc:
                    duration = time.time() - start_time
                    logger.error(f"Exception in profiled function {name}: {exc}", extra={
                        'function': name,
                        'duration': duration,
                        'exception_type': type(exc).__name__
                    })
                    raise
            
            return wrapper
        return decorator


# Global profiler instance
profiler = PerformanceProfiler()


def profile_function(func_name: Optional[str] = None):
    """Decorator to profile function execution time and memory usage."""
    return profiler.profile_function(func_name)


@contextmanager
def profile_block(block_name: str):
    """Context manager to profile a block of code."""
    if not profiler.enabled:
        yield
        return
    
    start_time = time.time()
    start_memory = None
    
    try:
        start_memory = profiler.process.memory_info().rss
    except:
        pass
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        memory_delta = 0
        
        try:
            if start_memory:
                end_memory = profiler.process.memory_info().rss
                memory_delta = end_memory - start_memory
        except:
            pass
        
        # Log performance data
        log_performance_metric(logger, block_name, duration, {
            'memory_delta_mb': memory_delta / 1024 / 1024 if memory_delta else 0,
            'block_type': 'code_block'
        })
        
        if duration > SLOW_THRESHOLD:
            logger.warning(f"Slow code block: {block_name} took {duration:.3f}s")


class MemoryProfiler:
    """Specialized memory profiling utilities."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """Get detailed memory usage information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            return {
                'rss_bytes': memory_info.rss,
                'vms_bytes': memory_info.vms,
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': memory_percent,
                'available_mb': psutil.virtual_memory().available / 1024 / 1024,
                'total_mb': psutil.virtual_memory().total / 1024 / 1024
            }
        except Exception as exc:
            logger.error(f"Error getting memory usage: {exc}")
            return {}
    
    @staticmethod
    def check_memory_threshold(threshold_mb: float = 100.0) -> bool:
        """Check if memory usage exceeds threshold."""
        memory_info = MemoryProfiler.get_memory_usage()
        current_mb = memory_info.get('rss_mb', 0)
        return current_mb > threshold_mb
    
    @contextmanager
    def track_memory(self, operation_name: str):
        """Context manager to track memory usage for an operation."""
        start_memory = self.get_memory_usage()
        
        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            
            memory_delta = end_memory.get('rss_mb', 0) - start_memory.get('rss_mb', 0)
            logger.info(f"Memory usage for {operation_name}: {memory_delta:+.2f}MB", extra={
                'operation': operation_name,
                'memory_delta_mb': memory_delta,
                'start_memory_mb': start_memory.get('rss_mb', 0),
                'end_memory_mb': end_memory.get('rss_mb', 0)
            })


def get_performance_report() -> Dict[str, Any]:
    """Get a comprehensive performance report."""
    return {
        'profiler_summary': profiler.get_profile_summary(),
        'memory_info': MemoryProfiler.get_memory_usage(),
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'boot_time': psutil.boot_time(),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else []
        }
    }


def start_profiling(enable_monitoring: bool = True, monitor_interval: float = 5.0):
    """Start performance profiling with optional system monitoring."""
    profiler.enable()
    
    if enable_monitoring:
        profiler.start_monitoring(monitor_interval)
    
    logger.info("Performance profiling started")


def stop_profiling():
    """Stop performance profiling and monitoring."""
    profiler.disable()
    profiler.stop_monitoring()
    logger.info("Performance profiling stopped")


# Create global instances
memory_profiler = MemoryProfiler()


if __name__ == '__main__':
    # Demo/test the profiling functionality
    start_profiling()
    
    @profile_function('test_function')
    def test_function(n: int = 1000000):
        """Test function for profiling demo."""
        return sum(range(n))
    
    with profile_block('test_block'):
        result = test_function(500000)
        time.sleep(0.1)
    
    print("Performance Report:")
    print(json.dumps(get_performance_report(), indent=2, default=str))
    
    stop_profiling()