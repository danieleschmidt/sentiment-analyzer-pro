"""Performance optimization utilities for sentiment analyzer."""

import asyncio
import concurrent.futures
import threading
import time
from typing import List, Dict, Any, Optional, Callable
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Generic connection pool for managing resources."""
    
    def __init__(self, create_connection: Callable, max_connections: int = 10, 
                 timeout: float = 30.0):
        self._create_connection = create_connection
        self._max_connections = max_connections
        self._timeout = timeout
        self._pool: List[Any] = []
        self._in_use: set = set()
        self._lock = threading.RLock()
        
    def get_connection(self):
        """Get a connection from the pool."""
        with self._lock:
            # Try to get an existing connection
            if self._pool:
                conn = self._pool.pop()
                self._in_use.add(id(conn))
                return conn
            
            # Create new connection if under limit
            if len(self._in_use) < self._max_connections:
                conn = self._create_connection()
                self._in_use.add(id(conn))
                return conn
            
            # Pool exhausted
            raise RuntimeError("Connection pool exhausted")
    
    def return_connection(self, conn):
        """Return a connection to the pool."""
        with self._lock:
            conn_id = id(conn)
            if conn_id in self._in_use:
                self._in_use.remove(conn_id)
                self._pool.append(conn)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            return {
                'available': len(self._pool),
                'in_use': len(self._in_use),
                'total': len(self._pool) + len(self._in_use),
                'max_connections': self._max_connections
            }


class AsyncBatchProcessor:
    """Asynchronous batch processor for handling multiple requests."""
    
    def __init__(self, max_workers: int = 4, batch_timeout: float = 0.1):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._batch_timeout = batch_timeout
        self._pending_requests: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._processing = False
        
    async def process_batch(self, processor_func: Callable, items: List[Any]) -> List[Any]:
        """Process items in batch asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Split into chunks for parallel processing
        chunk_size = max(1, len(items) // self._executor._max_workers)
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Process chunks in parallel
        tasks = []
        for chunk in chunks:
            task = loop.run_in_executor(self._executor, processor_func, chunk)
            tasks.append(task)
        
        # Wait for all chunks to complete
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        flattened = []
        for result in results:
            if isinstance(result, list):
                flattened.extend(result)
            else:
                flattened.append(result)
        
        return flattened
    
    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def record_timing(self, operation: str, duration: float):
        """Record timing for an operation."""
        with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = []
            self._metrics[operation].append(duration)
            
            # Keep only last 1000 measurements to prevent memory bloat
            if len(self._metrics[operation]) > 1000:
                self._metrics[operation] = self._metrics[operation][-1000:]
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        with self._lock:
            if operation not in self._metrics or not self._metrics[operation]:
                return {}
            
            timings = self._metrics[operation]
            return {
                'count': len(timings),
                'avg': sum(timings) / len(timings),
                'min': min(timings),
                'max': max(timings),
                'p95': sorted(timings)[int(len(timings) * 0.95)] if len(timings) > 20 else max(timings),
                'p99': sorted(timings)[int(len(timings) * 0.99)] if len(timings) > 100 else max(timings)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        with self._lock:
            return {op: self.get_stats(op) for op in self._metrics.keys()}


# Global performance monitor
performance_monitor = PerformanceMonitor()


def time_it(operation_name: str):
    """Decorator to time function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                performance_monitor.record_timing(operation_name, duration)
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for handling failures gracefully."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time = None
        self._state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function with circuit breaker protection."""
        with self._lock:
            if self._state == 'OPEN':
                if self._last_failure_time and \
                   time.time() - self._last_failure_time > self._recovery_timeout:
                    self._state = 'HALF_OPEN'
                else:
                    raise RuntimeError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset failure count
                if self._state == 'HALF_OPEN':
                    self._state = 'CLOSED'
                self._failure_count = 0
                return result
                
            except Exception as e:
                self._failure_count += 1
                self._last_failure_time = time.time()
                
                if self._failure_count >= self._failure_threshold:
                    self._state = 'OPEN'
                    logger.warning(f"Circuit breaker opened after {self._failure_count} failures")
                
                raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        with self._lock:
            return {
                'state': self._state,
                'failure_count': self._failure_count,
                'last_failure_time': self._last_failure_time
            }


class ResourceLimiter:
    """Limit resource usage (memory, CPU, etc.)."""
    
    def __init__(self, max_memory_mb: int = 1000, max_concurrent_requests: int = 100):
        self._max_memory_mb = max_memory_mb
        self._max_concurrent_requests = max_concurrent_requests
        self._current_requests = 0
        self._lock = threading.Semaphore(max_concurrent_requests)
    
    def __enter__(self):
        """Context manager entry."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Too many concurrent requests")
        self._current_requests += 1
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._current_requests -= 1
        self._lock.release()
    
    def get_stats(self) -> Dict[str, int]:
        """Get resource usage statistics."""
        return {
            'current_requests': self._current_requests,
            'max_concurrent_requests': self._max_concurrent_requests,
            'available_slots': self._max_concurrent_requests - self._current_requests
        }


# Global instances
resource_limiter = ResourceLimiter()
circuit_breaker = CircuitBreaker()
batch_processor = AsyncBatchProcessor()


def optimize_batch_prediction(texts: List[str], model, batch_size: int = 32) -> List[str]:
    """Optimize batch predictions by processing in chunks."""
    if len(texts) <= batch_size:
        return model.predict(texts)
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = model.predict(batch)
        if isinstance(batch_results, str):
            batch_results = [batch_results]
        results.extend(batch_results)
    
    return results


@time_it("text_preprocessing")
def preprocess_text_optimized(text: str, cache: Optional[Dict[str, str]] = None) -> str:
    """Optimized text preprocessing with caching."""
    if cache and text in cache:
        return cache[text]
    
    # Basic preprocessing - can be extended
    processed = text.strip().lower()
    
    if cache is not None:
        cache[text] = processed
    
    return processed


class AutoScaler:
    """Automatically scale resources based on load."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 10, scale_threshold: float = 0.8):
        self._min_workers = min_workers
        self._max_workers = max_workers
        self._scale_threshold = scale_threshold
        self._current_workers = min_workers
        self._last_scale_time = time.time()
        self._scale_cooldown = 30.0  # seconds
    
    def should_scale_up(self, current_load: float) -> bool:
        """Check if we should scale up workers."""
        return (current_load > self._scale_threshold and 
                self._current_workers < self._max_workers and
                time.time() - self._last_scale_time > self._scale_cooldown)
    
    def should_scale_down(self, current_load: float) -> bool:
        """Check if we should scale down workers."""
        return (current_load < (self._scale_threshold * 0.5) and 
                self._current_workers > self._min_workers and
                time.time() - self._last_scale_time > self._scale_cooldown)
    
    def scale(self, direction: str) -> int:
        """Scale workers up or down."""
        if direction == 'up' and self._current_workers < self._max_workers:
            self._current_workers += 1
        elif direction == 'down' and self._current_workers > self._min_workers:
            self._current_workers -= 1
        
        self._last_scale_time = time.time()
        return self._current_workers


# Global auto-scaler
auto_scaler = AutoScaler()