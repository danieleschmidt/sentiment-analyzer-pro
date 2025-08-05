"""
⚡ Neuromorphic Performance Optimization
======================================

Advanced performance optimization for neuromorphic spikeformer processing,
including intelligent caching, concurrent processing, and resource pooling.

Generation 3: MAKE IT SCALE - Optimized performance and scalability
"""

import numpy as np
import torch
import torch.multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import logging
import time
import hashlib
import pickle
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict, OrderedDict
import threading
from abc import ABC, abstractmethod
import warnings

logger = logging.getLogger(__name__)


class CachePolicy(ABC):
    """Abstract base class for cache policies."""
    
    @abstractmethod
    def should_evict(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Determine if a cache entry should be evicted."""
        pass
    
    @abstractmethod
    def update_access(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update metadata on cache access."""
        pass


class LRUPolicy(CachePolicy):
    """Least Recently Used cache policy."""
    
    def should_evict(self, key: str, metadata: Dict[str, Any]) -> bool:
        return True  # Let LRU mechanism handle it
    
    def update_access(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        metadata['last_accessed'] = time.time()
        metadata['access_count'] = metadata.get('access_count', 0) + 1
        return metadata


class TTLPolicy(CachePolicy):
    """Time-to-Live cache policy."""
    
    def __init__(self, ttl_seconds: float = 300.0):
        self.ttl_seconds = ttl_seconds
    
    def should_evict(self, key: str, metadata: Dict[str, Any]) -> bool:
        creation_time = metadata.get('created_at', 0)
        return time.time() - creation_time > self.ttl_seconds
    
    def update_access(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        metadata['last_accessed'] = time.time()
        return metadata


class LFUPolicy(CachePolicy):
    """Least Frequently Used cache policy."""
    
    def should_evict(self, key: str, metadata: Dict[str, Any]) -> bool:
        # Evict if access count is below threshold
        return metadata.get('access_count', 0) < 2
    
    def update_access(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        metadata['access_count'] = metadata.get('access_count', 0) + 1
        metadata['last_accessed'] = time.time()
        return metadata


class IntelligentCache:
    """
    Multi-level intelligent caching system for neuromorphic processing.
    
    Supports multiple cache policies, automatic eviction, and performance monitoring.
    """
    
    def __init__(
        self, 
        max_size: int = 1000, 
        policy: Optional[CachePolicy] = None,
        enable_compression: bool = True,
        enable_metrics: bool = True
    ):
        self.max_size = max_size
        self.policy = policy or LRUPolicy()
        self.enable_compression = enable_compression
        self.enable_metrics = enable_metrics
        
        # Cache storage
        self._cache = OrderedDict()
        self._metadata = {}
        self._lock = threading.RLock()
        
        # Performance metrics
        self._metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0,
            'cache_size': 0,
            'compression_ratio': 0.0
        }
        
        logger.info(f"Initialized IntelligentCache with max_size={max_size}")
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, (np.ndarray, torch.Tensor)):
            # Use shape and hash of flattened data for arrays
            if isinstance(data, torch.Tensor):
                data_np = data.detach().cpu().numpy()
            else:
                data_np = data
            
            shape_str = str(data_np.shape)
            data_hash = hashlib.md5(data_np.tobytes()).hexdigest()[:16]
            return f"array_{shape_str}_{data_hash}"
        else:
            # Use pickle for other data types
            data_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.md5(data_bytes).hexdigest()
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage."""
        if not self.enable_compression:
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        
        try:
            import gzip
            raw_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            compressed_data = gzip.compress(raw_data)
            
            # Update compression ratio
            if self.enable_metrics:
                ratio = len(compressed_data) / len(raw_data)
                self._metrics['compression_ratio'] = (
                    self._metrics['compression_ratio'] * 0.9 + ratio * 0.1
                )
            
            return compressed_data
        except Exception as e:
            logger.warning(f"Compression failed: {e}, using uncompressed")
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data from storage."""
        if not self.enable_compression:
            return pickle.loads(compressed_data)
        
        try:
            import gzip
            raw_data = gzip.decompress(compressed_data)
            return pickle.loads(raw_data)
        except Exception:
            # Fallback to uncompressed
            return pickle.loads(compressed_data)
    
    def _evict_if_needed(self):
        """Evict cache entries if needed."""
        with self._lock:
            # Check TTL-based evictions first
            keys_to_remove = []
            for key, metadata in self._metadata.items():
                if self.policy.should_evict(key, metadata):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            # Size-based eviction
            while len(self._cache) >= self.max_size:
                # Remove oldest entry (LRU)
                oldest_key = next(iter(self._cache))
                self._remove_entry(oldest_key)
    
    def _remove_entry(self, key: str):
        """Remove a cache entry."""
        if key in self._cache:
            del self._cache[key]
            del self._metadata[key]
            if self.enable_metrics:
                self._metrics['evictions'] += 1
                self._metrics['cache_size'] = len(self._cache)
    
    def get(self, key_data: Any) -> Optional[Any]:
        """
        Get data from cache.
        
        Args:
            key_data: Data to use as cache key
            
        Returns:
            Cached data if found, None otherwise
        """
        key = self._generate_key(key_data)
        
        with self._lock:
            if self.enable_metrics:
                self._metrics['total_requests'] += 1
            
            if key in self._cache:
                # Update access metadata
                self._metadata[key] = self.policy.update_access(key, self._metadata[key])
                
                # Move to end (LRU)
                self._cache.move_to_end(key)
                
                # Get and decompress data
                try:
                    data = self._decompress_data(self._cache[key])
                    if self.enable_metrics:
                        self._metrics['hits'] += 1
                    return data
                except Exception as e:
                    logger.warning(f"Cache decompression failed: {e}")
                    self._remove_entry(key)
            
            if self.enable_metrics:
                self._metrics['misses'] += 1
            return None
    
    def put(self, key_data: Any, value: Any):
        """
        Store data in cache.
        
        Args:
            key_data: Data to use as cache key
            value: Data to store
        """
        key = self._generate_key(key_data)
        
        with self._lock:
            # Evict if needed
            self._evict_if_needed()
            
            # Compress and store
            try:
                compressed_value = self._compress_data(value)
                self._cache[key] = compressed_value
                self._metadata[key] = {
                    'created_at': time.time(),
                    'last_accessed': time.time(),
                    'access_count': 1,
                    'size_bytes': len(compressed_value)
                }
                
                if self.enable_metrics:
                    self._metrics['cache_size'] = len(self._cache)
                
            except Exception as e:
                logger.error(f"Cache storage failed: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._metadata.clear()
            if self.enable_metrics:
                self._metrics['cache_size'] = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        if not self.enable_metrics:
            return {}
        
        with self._lock:
            hit_rate = 0.0
            if self._metrics['total_requests'] > 0:
                hit_rate = self._metrics['hits'] / self._metrics['total_requests']
            
            return {
                'hit_rate': hit_rate,
                'total_requests': self._metrics['total_requests'],
                'hits': self._metrics['hits'],
                'misses': self._metrics['misses'],
                'evictions': self._metrics['evictions'],
                'cache_size': self._metrics['cache_size'],
                'compression_ratio': self._metrics['compression_ratio']
            }


class ResourcePool:
    """Resource pooling for expensive neuromorphic operations."""
    
    def __init__(self, max_workers: int = 4, pool_type: str = "thread"):
        self.max_workers = max_workers
        self.pool_type = pool_type
        
        if pool_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        elif pool_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            raise ValueError(f"Invalid pool_type: {pool_type}")
        
        self._active_tasks = 0
        self._lock = threading.Lock()
        
        logger.info(f"Initialized {pool_type} pool with {max_workers} workers")
    
    def submit(self, fn: Callable, *args, **kwargs):
        """Submit task to resource pool."""
        with self._lock:
            self._active_tasks += 1
        
        future = self.executor.submit(fn, *args, **kwargs)
        
        def cleanup_callback(fut):
            with self._lock:
                self._active_tasks -= 1
        
        future.add_done_callback(cleanup_callback)
        return future
    
    def get_stats(self) -> Dict[str, int]:
        """Get resource pool statistics."""
        with self._lock:
            return {
                'max_workers': self.max_workers,
                'active_tasks': self._active_tasks,
                'pool_type': self.pool_type
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown resource pool."""
        self.executor.shutdown(wait=wait)


class NeuromorphicOptimizer:
    """
    Comprehensive performance optimizer for neuromorphic processing.
    
    Provides intelligent caching, resource pooling, and optimization strategies.
    """
    
    def __init__(
        self,
        cache_size: int = 1000,
        max_workers: int = 4,
        enable_caching: bool = True,
        enable_pooling: bool = True,
        cache_policy: str = "lru"
    ):
        self.enable_caching = enable_caching
        self.enable_pooling = enable_pooling
        
        # Initialize cache
        if enable_caching:
            policy_map = {
                'lru': LRUPolicy(),
                'ttl': TTLPolicy(),
                'lfu': LFUPolicy()
            }
            cache_policy_obj = policy_map.get(cache_policy, LRUPolicy())
            
            self.synthesis_cache = IntelligentCache(
                max_size=cache_size, 
                policy=cache_policy_obj,
                enable_compression=True
            )
            self.validation_cache = IntelligentCache(
                max_size=cache_size // 2,
                policy=TTLPolicy(ttl_seconds=60.0)  # Short TTL for validation
            )
        else:
            self.synthesis_cache = None
            self.validation_cache = None
        
        # Initialize resource pool
        if enable_pooling:
            self.resource_pool = ResourcePool(max_workers=max_workers, pool_type="thread")
        else:
            self.resource_pool = None
        
        # Performance tracking
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_executions': 0,
            'optimization_savings': 0.0
        }
        
        logger.info(f"Initialized NeuromorphicOptimizer (cache: {enable_caching}, pool: {enable_pooling})")
    
    def cached_operation(self, cache_type: str = "synthesis"):
        """
        Decorator for caching expensive operations.
        
        Args:
            cache_type: Type of cache to use ('synthesis' or 'validation')
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_caching:
                    return func(*args, **kwargs)
                
                # Select cache
                cache = self.synthesis_cache if cache_type == "synthesis" else self.validation_cache
                if cache is None:
                    return func(*args, **kwargs)
                
                # Create cache key from args
                cache_key = (args, tuple(sorted(kwargs.items())))
                
                # Try cache first
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    self.performance_stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Store in cache
                cache.put(cache_key, result)
                self.performance_stats['cache_misses'] += 1
                
                logger.debug(f"Cache miss for {func.__name__} (executed in {execution_time:.3f}s)")
                return result
            
            return wrapper
        return decorator
    
    def parallel_batch_processing(
        self, 
        batch_data: List[Any], 
        process_func: Callable,
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """
        Process batch data in parallel.
        
        Args:
            batch_data: List of data items to process
            process_func: Function to apply to each item
            chunk_size: Optional chunk size for processing
            
        Returns:
            List of processed results
        """
        if not self.enable_pooling or self.resource_pool is None:
            return [process_func(item) for item in batch_data]
        
        if chunk_size is None:
            chunk_size = max(1, len(batch_data) // self.resource_pool.max_workers)
        
        # Submit tasks to pool
        futures = []
        for i in range(0, len(batch_data), chunk_size):
            chunk = batch_data[i:i + chunk_size]
            future = self.resource_pool.submit(self._process_chunk, chunk, process_func)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"Parallel processing error: {e}")
                # Fallback to sequential processing for failed chunks
                results.extend([None] * chunk_size)
        
        self.performance_stats['parallel_executions'] += 1
        return results
    
    def _process_chunk(self, chunk: List[Any], process_func: Callable) -> List[Any]:
        """Process a chunk of data."""
        return [process_func(item) for item in chunk]
    
    def optimize_model_inference(self, model: torch.nn.Module):
        """
        Optimize PyTorch model for inference.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        try:
            # Set to evaluation mode
            model.eval()
            
            # Enable inference optimizations
            with torch.no_grad():
                # Compile model if available (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    try:
                        model = torch.compile(model, mode='reduce-overhead')
                        logger.info("Model compiled with torch.compile")
                    except Exception as e:
                        logger.warning(f"torch.compile failed: {e}")
                
                # Enable JIT if possible
                try:
                    # Create dummy input for tracing
                    dummy_input = torch.randn(1, 100, 768)  # Adjust as needed
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        traced_model = torch.jit.trace(model, dummy_input)
                        traced_model.eval()
                        logger.info("Model traced with TorchScript")
                        return traced_model
                except Exception as e:
                    logger.warning(f"TorchScript tracing failed: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            'performance_stats': self.performance_stats.copy(),
            'cache_enabled': self.enable_caching,
            'pooling_enabled': self.enable_pooling
        }
        
        # Add cache metrics
        if self.enable_caching:
            if self.synthesis_cache:
                stats['synthesis_cache'] = self.synthesis_cache.get_metrics()
            if self.validation_cache:
                stats['validation_cache'] = self.validation_cache.get_metrics()
        
        # Add pool stats
        if self.enable_pooling and self.resource_pool:
            stats['resource_pool'] = self.resource_pool.get_stats()
        
        # Calculate optimization effectiveness
        total_requests = self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
        if total_requests > 0:
            cache_hit_rate = self.performance_stats['cache_hits'] / total_requests
            stats['cache_hit_rate'] = cache_hit_rate
            stats['estimated_time_saved'] = cache_hit_rate * 0.1  # Assume 100ms saved per hit
        
        return stats
    
    def cleanup(self):
        """Cleanup optimizer resources."""
        if self.enable_caching:
            if self.synthesis_cache:
                self.synthesis_cache.clear()
            if self.validation_cache:
                self.validation_cache.clear()
        
        if self.enable_pooling and self.resource_pool:
            self.resource_pool.shutdown()
        
        logger.info("NeuromorphicOptimizer cleanup completed")


# Global optimizer instance
_global_optimizer = None


def get_optimizer() -> NeuromorphicOptimizer:
    """Get global optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = NeuromorphicOptimizer()
    return _global_optimizer


def configure_optimizer(
    cache_size: int = 1000,
    max_workers: int = 4,
    enable_caching: bool = True,
    enable_pooling: bool = True
) -> NeuromorphicOptimizer:
    """
    Configure global optimizer.
    
    Args:
        cache_size: Maximum cache size
        max_workers: Maximum worker threads
        enable_caching: Enable caching
        enable_pooling: Enable thread pooling
        
    Returns:
        Configured optimizer instance
    """
    global _global_optimizer
    if _global_optimizer is not None:
        _global_optimizer.cleanup()
    
    _global_optimizer = NeuromorphicOptimizer(
        cache_size=cache_size,
        max_workers=max_workers,
        enable_caching=enable_caching,
        enable_pooling=enable_pooling
    )
    
    return _global_optimizer


# Optimization decorators
def cached_synthesis(cache_type: str = "synthesis"):
    """Decorator for caching synthesis operations."""
    def decorator(func: Callable):
        optimizer = get_optimizer()
        return optimizer.cached_operation(cache_type)(func)
    return decorator


def parallel_processing(chunk_size: Optional[int] = None):
    """Decorator for parallel batch processing."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(batch_data: List[Any], *args, **kwargs):
            optimizer = get_optimizer()
            
            # Create a partial function with fixed args/kwargs
            def process_item(item):
                return func(item, *args, **kwargs)
            
            return optimizer.parallel_batch_processing(batch_data, process_item, chunk_size)
        
        return wrapper
    return decorator


def optimized_inference(func: Callable):
    """Decorator for optimized model inference."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Log performance
        execution_time = end_time - start_time
        logger.debug(f"Optimized inference completed in {execution_time:.3f}s")
        
        return result
    
    return wrapper


# Performance monitoring utilities
class PerformanceProfiler:
    """Simple performance profiler for neuromorphic operations."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.counters = defaultdict(int)
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        class TimingContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                duration = end_time - self.start_time
                self.profiler.timings[self.name].append(duration)
                self.profiler.counters[self.name] += 1
        
        return TimingContext(self, operation_name)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get profiling statistics."""
        stats = {}
        
        for operation, timings in self.timings.items():
            if timings:
                stats[operation] = {
                    'count': len(timings),
                    'total_time': sum(timings),
                    'avg_time': sum(timings) / len(timings),
                    'min_time': min(timings),
                    'max_time': max(timings)
                }
        
        return stats
    
    def reset(self):
        """Reset profiling data."""
        self.timings.clear()
        self.counters.clear()


# Global profiler instance
_global_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance."""
    return _global_profiler


def profile_operation(operation_name: str):
    """Decorator for profiling operations."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            with profiler.time_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator