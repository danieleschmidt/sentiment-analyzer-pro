"""
High-performance scalable engine for sentiment analysis
Generation 3: Make It Scale - Performance optimization and caching
"""
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import logging
import json
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
import queue
from functools import lru_cache, wraps
import weakref
import gc
import pickle
import gzip
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    LRU = "lru"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    NONE = "none"

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_usage_mb: float
    cache_hit: bool = False
    batch_size: int = 1
    error: Optional[str] = None

class SmartCache:
    """Advanced caching system with multiple strategies"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 ttl_seconds: int = 3600,
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 compression: bool = True):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.strategy = strategy
        self.compression = compression
        
        self.cache = {}
        self.access_times = {}
        self.hit_counts = {}
        self.creation_times = {}
        self.lock = threading.RLock()
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function signature"""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data if compression is enabled"""
        serialized = pickle.dumps(data)
        if self.compression:
            return gzip.compress(serialized)
        return serialized
    
    def _decompress_data(self, data: bytes) -> Any:
        """Decompress data if compression was used"""
        if self.compression:
            data = gzip.decompress(data)
        return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            current_time = time.time()
            
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            # Check TTL expiration
            if (self.strategy in [CacheStrategy.TTL, CacheStrategy.ADAPTIVE] and
                current_time - self.creation_times[key] > self.ttl_seconds):
                self._evict(key)
                self.miss_count += 1
                return None
            
            # Update access statistics
            self.access_times[key] = current_time
            self.hit_counts[key] = self.hit_counts.get(key, 0) + 1
            self.hit_count += 1
            
            try:
                return self._decompress_data(self.cache[key])
            except Exception as e:
                logger.warning(f"Cache decompression error: {e}")
                self._evict(key)
                self.miss_count += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache"""
        with self.lock:
            current_time = time.time()
            
            # If cache is full, evict based on strategy
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_by_strategy()
            
            try:
                compressed_value = self._compress_data(value)
                self.cache[key] = compressed_value
                self.access_times[key] = current_time
                self.creation_times[key] = current_time
                self.hit_counts[key] = 0
            except Exception as e:
                logger.warning(f"Cache compression error: {e}")
    
    def _evict(self, key: str) -> None:
        """Evict specific key"""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.creation_times[key]
            del self.hit_counts[key]
            self.eviction_count += 1
    
    def _evict_by_strategy(self) -> None:
        """Evict item based on caching strategy"""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            oldest_key = min(self.access_times.keys(), key=self.access_times.get)
            self._evict(oldest_key)
        
        elif self.strategy == CacheStrategy.TTL:
            # Evict oldest by creation time
            current_time = time.time()
            expired_keys = [
                key for key, creation_time in self.creation_times.items()
                if current_time - creation_time > self.ttl_seconds
            ]
            
            if expired_keys:
                self._evict(expired_keys[0])
            else:
                oldest_key = min(self.creation_times.keys(), key=self.creation_times.get)
                self._evict(oldest_key)
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive eviction based on access patterns
            current_time = time.time()
            
            # Score each item based on recency, frequency, and age
            scores = {}
            for key in self.cache.keys():
                recency = current_time - self.access_times[key]
                frequency = self.hit_counts[key]
                age = current_time - self.creation_times[key]
                
                # Lower score = more likely to evict
                score = frequency / (1 + recency + age * 0.1)
                scores[key] = score
            
            # Evict item with lowest score
            victim_key = min(scores.keys(), key=scores.get)
            self._evict(victim_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_ratio': hit_ratio,
            'eviction_count': self.eviction_count,
            'strategy': self.strategy.value
        }

def smart_cache(max_size: int = 1000, 
                ttl_seconds: int = 3600,
                strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                key_func: Optional[Callable] = None):
    """Decorator for smart caching"""
    
    cache_instance = SmartCache(max_size, ttl_seconds, strategy)
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_instance._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.put(cache_key, result)
            
            return result
        
        wrapper.cache_stats = cache_instance.get_stats
        wrapper.cache_clear = lambda: cache_instance.cache.clear()
        
        return wrapper
    return decorator

class BatchProcessor:
    """High-performance batch processing system"""
    
    def __init__(self, 
                 batch_size: int = 100,
                 max_workers: int = 4,
                 use_processes: bool = False,
                 timeout: float = 300.0):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.timeout = timeout
        
        # Performance tracking
        self.processed_count = 0
        self.error_count = 0
        self.total_time = 0.0
        
    def process_batch(self, 
                     items: List[Any],
                     process_func: Callable,
                     **kwargs) -> List[Any]:
        """Process items in optimized batches"""
        start_time = time.time()
        
        try:
            if len(items) <= self.batch_size:
                # Small batch - process directly
                results = [process_func(item, **kwargs) for item in items]
            else:
                # Large batch - use parallel processing
                results = self._process_parallel_batches(items, process_func, **kwargs)
            
            self.processed_count += len(items)
            self.total_time += time.time() - start_time
            
            return results
            
        except Exception as e:
            self.error_count += len(items)
            logger.error(f"Batch processing failed: {e}")
            raise
    
    def _process_parallel_batches(self, 
                                items: List[Any],
                                process_func: Callable,
                                **kwargs) -> List[Any]:
        """Process batches in parallel"""
        
        # Split into batches
        batches = [
            items[i:i + self.batch_size] 
            for i in range(0, len(items), self.batch_size)
        ]
        
        results = []
        
        if self.use_processes:
            # Use process pool for CPU-intensive tasks
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {
                    executor.submit(self._process_single_batch, batch, process_func, kwargs): batch
                    for batch in batches
                }
                
                for future in as_completed(future_to_batch, timeout=self.timeout):
                    batch_results = future.result()
                    results.extend(batch_results)
        else:
            # Use thread pool for I/O-bound tasks
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {
                    executor.submit(self._process_single_batch, batch, process_func, kwargs): batch
                    for batch in batches
                }
                
                for future in as_completed(future_to_batch, timeout=self.timeout):
                    batch_results = future.result()
                    results.extend(batch_results)
        
        return results
    
    @staticmethod
    def _process_single_batch(batch: List[Any], 
                            process_func: Callable,
                            kwargs: Dict[str, Any]) -> List[Any]:
        """Process a single batch"""
        return [process_func(item, **kwargs) for item in batch]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        avg_time_per_item = (self.total_time / self.processed_count 
                           if self.processed_count > 0 else 0)
        
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'total_time': self.total_time,
            'avg_time_per_item': avg_time_per_item,
            'items_per_second': self.processed_count / self.total_time if self.total_time > 0 else 0
        }

class MemoryOptimizer:
    """Memory usage optimization and monitoring"""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB"""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type == 'object':
                # Try to convert to category if it saves memory
                if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                    optimized_df[col] = optimized_df[col].astype('category')
            
            elif col_type in ['int64', 'int32']:
                # Downcast integers
                col_min = optimized_df[col].min()
                col_max = optimized_df[col].max()
                
                if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
            
            elif col_type in ['float64', 'float32']:
                # Downcast floats
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        return optimized_df
    
    @staticmethod
    def memory_efficient_chunked_processing(df: pd.DataFrame, 
                                          chunk_size: int = 10000,
                                          process_func: Callable = None) -> pd.DataFrame:
        """Process large DataFrame in memory-efficient chunks"""
        if process_func is None:
            return df
        
        results = []
        
        for chunk_start in range(0, len(df), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(df))
            chunk = df.iloc[chunk_start:chunk_end]
            
            # Process chunk
            processed_chunk = process_func(chunk)
            results.append(processed_chunk)
            
            # Force garbage collection to free memory
            del chunk
            gc.collect()
        
        return pd.concat(results, ignore_index=True)

class PerformanceMonitor:
    """Performance monitoring and profiling"""
    
    def __init__(self):
        self.metrics = []
        self.lock = threading.Lock()
    
    def track_operation(self, operation: str):
        """Context manager for tracking operation performance"""
        return PerformanceTracker(self, operation)
    
    def add_metric(self, metric: PerformanceMetrics):
        """Add performance metric"""
        with self.lock:
            self.metrics.append(metric)
            
            # Keep only recent metrics to prevent memory growth
            if len(self.metrics) > 10000:
                self.metrics = self.metrics[-5000:]
    
    def get_performance_summary(self, 
                              operation: Optional[str] = None,
                              last_n_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary"""
        with self.lock:
            cutoff_time = time.time() - (last_n_minutes * 60)
            
            relevant_metrics = [
                m for m in self.metrics 
                if m.start_time >= cutoff_time and (operation is None or m.operation == operation)
            ]
            
            if not relevant_metrics:
                return {}
            
            durations = [m.duration_ms for m in relevant_metrics]
            memory_usages = [m.memory_usage_mb for m in relevant_metrics if m.memory_usage_mb > 0]
            cache_hits = sum(1 for m in relevant_metrics if m.cache_hit)
            errors = sum(1 for m in relevant_metrics if m.error)
            
            return {
                'operation': operation,
                'total_operations': len(relevant_metrics),
                'avg_duration_ms': np.mean(durations),
                'min_duration_ms': np.min(durations),
                'max_duration_ms': np.max(durations),
                'p95_duration_ms': np.percentile(durations, 95),
                'avg_memory_mb': np.mean(memory_usages) if memory_usages else 0,
                'cache_hit_rate': cache_hits / len(relevant_metrics),
                'error_rate': errors / len(relevant_metrics),
                'operations_per_minute': len(relevant_metrics) / last_n_minutes
            }

class PerformanceTracker:
    """Context manager for performance tracking"""
    
    def __init__(self, monitor: PerformanceMonitor, operation: str):
        self.monitor = monitor
        self.operation = operation
        self.start_time = None
        self.start_memory = None
        self.cache_hit = False
        self.error = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = MemoryOptimizer.get_memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_memory = MemoryOptimizer.get_memory_usage()
        
        if exc_type is not None:
            self.error = str(exc_val)
        
        metric = PerformanceMetrics(
            operation=self.operation,
            start_time=self.start_time,
            end_time=end_time,
            duration_ms=(end_time - self.start_time) * 1000,
            memory_usage_mb=max(0, end_memory - self.start_memory),
            cache_hit=self.cache_hit,
            error=self.error
        )
        
        self.monitor.add_metric(metric)
    
    def mark_cache_hit(self):
        """Mark this operation as a cache hit"""
        self.cache_hit = True

class HighPerformanceSentimentAnalyzer:
    """High-performance sentiment analyzer with optimization"""
    
    def __init__(self, 
                 cache_size: int = 10000,
                 batch_size: int = 100,
                 max_workers: int = 4):
        
        self.performance_monitor = PerformanceMonitor()
        self.batch_processor = BatchProcessor(batch_size, max_workers)
        self.memory_optimizer = MemoryOptimizer()
        
        # Initialize with caching
        self._setup_cached_functions()
    
    def _setup_cached_functions(self):
        """Setup cached versions of core functions"""
        from .preprocessing import preprocess_text
        from .models import build_nb_model
        
        # Cache preprocessing results
        self.cached_preprocess = smart_cache(
            max_size=10000,
            ttl_seconds=3600,
            strategy=CacheStrategy.ADAPTIVE
        )(preprocess_text)
        
        # Cache model building (though this should be done once)
        self.cached_model_build = smart_cache(
            max_size=10,
            ttl_seconds=86400,  # 24 hours
            strategy=CacheStrategy.TTL
        )(build_nb_model)
    
    def predict_batch_optimized(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Optimized batch prediction"""
        with self.performance_monitor.track_operation("batch_prediction") as tracker:
            
            def predict_single(text: str) -> Dict[str, Any]:
                # Use cached preprocessing
                processed_text = self.cached_preprocess(text)
                
                # Simple sentiment logic (replace with actual model)
                positive_words = ['good', 'great', 'excellent', 'amazing', 'love']
                negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst']
                
                text_lower = processed_text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                if pos_count > neg_count:
                    sentiment = 'positive'
                    confidence = min(0.9, 0.5 + pos_count * 0.1)
                elif neg_count > pos_count:
                    sentiment = 'negative'
                    confidence = min(0.9, 0.5 + neg_count * 0.1)
                else:
                    sentiment = 'neutral'
                    confidence = 0.5
                
                return {
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'processed_text': processed_text
                }
            
            # Process in optimized batches
            results = self.batch_processor.process_batch(texts, predict_single)
            
            return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'monitor': self.performance_monitor.get_performance_summary(),
            'batch_processor': self.batch_processor.get_stats(),
            'preprocessing_cache': self.cached_preprocess.cache_stats(),
            'model_cache': self.cached_model_build.cache_stats(),
            'memory_usage_mb': self.memory_optimizer.get_memory_usage()
        }

if __name__ == "__main__":
    # Test high-performance system
    analyzer = HighPerformanceSentimentAnalyzer()
    
    # Test batch processing
    test_texts = [
        "This is amazing!",
        "I love this product",
        "This is terrible",
        "Not bad",
        "Excellent quality"
    ] * 20  # 100 texts for batch testing
    
    print("🚀 Testing high-performance batch processing...")
    start_time = time.time()
    
    results = analyzer.predict_batch_optimized(test_texts)
    
    end_time = time.time()
    
    print(f"Processed {len(results)} texts in {(end_time - start_time)*1000:.2f}ms")
    print(f"Average: {(end_time - start_time)*1000/len(results):.2f}ms per text")
    
    # Show first few results
    for i, result in enumerate(results[:3]):
        print(f"  {i+1}. '{result['text'][:30]}...' -> {result['sentiment']} ({result['confidence']:.2f})")
    
    # Show performance stats
    print("\n📊 Performance Statistics:")
    stats = analyzer.get_performance_stats()
    for category, data in stats.items():
        if isinstance(data, dict):
            print(f"  {category}:")
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.2f}")
                else:
                    print(f"    {key}: {value}")
    
    print("\n✅ Generation 3 performance optimization completed!")