#!/usr/bin/env python3
"""
Generation 3: Performance and Scaling Optimizations
Advanced caching, concurrent processing, load balancing, and auto-scaling
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
from typing import Dict, List, Any, Optional, Callable
import json
import hashlib
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import multiprocessing as mp
import psutil
import gc

@dataclass
class PerformanceMetrics:
    cpu_usage: float
    memory_usage: float
    request_rate: float
    avg_response_time: float
    cache_hit_rate: float
    active_connections: int
    queue_depth: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AdvancedCache:
    """High-performance multi-level caching system."""
    
    def __init__(self, max_memory_mb: int = 128, ttl_seconds: int = 3600):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.cache_sizes = {}
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
    
    def _generate_key(self, text: str, model_params: Dict[str, Any] = None) -> str:
        """Generate cache key with content and parameters."""
        content = text + json.dumps(model_params or {}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _evict_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time > self.ttl
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _evict_lru(self, target_size: int):
        """Evict least recently used entries to reach target size."""
        if not self.access_times:
            return
        
        current_size = sum(self.cache_sizes.values())
        if current_size <= target_size:
            return
        
        # Sort by access time (oldest first)
        sorted_entries = sorted(self.access_times.items(), key=lambda x: x[1])
        
        for key, _ in sorted_entries:
            self._remove_entry(key)
            current_size = sum(self.cache_sizes.values())
            if current_size <= target_size:
                break
    
    def _remove_entry(self, key: str):
        """Remove cache entry."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.cache_sizes.pop(key, None)
    
    def get(self, text: str, model_params: Dict[str, Any] = None) -> Optional[Any]:
        """Get cached result."""
        with self.lock:
            key = self._generate_key(text, model_params)
            
            if key in self.cache:
                self.access_times[key] = time.time()
                self.hits += 1
                return self.cache[key]
            
            self.misses += 1
            return None
    
    def put(self, text: str, result: Any, model_params: Dict[str, Any] = None):
        """Cache result with intelligent eviction."""
        with self.lock:
            self._evict_expired()
            
            key = self._generate_key(text, model_params)
            result_size = len(json.dumps(result).encode())
            
            # Check if we need to evict for memory
            current_memory = sum(self.cache_sizes.values())
            if current_memory + result_size > self.max_memory_bytes:
                target_size = self.max_memory_bytes - result_size
                self._evict_lru(target_size)
            
            self.cache[key] = result
            self.access_times[key] = time.time()
            self.cache_sizes[key] = result_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(total_requests, 1)
        memory_usage = sum(self.cache_sizes.values()) / self.max_memory_bytes
        
        return {
            "hit_rate": hit_rate,
            "hits": self.hits,
            "misses": self.misses,
            "total_entries": len(self.cache),
            "memory_usage_percent": memory_usage * 100,
            "memory_usage_mb": sum(self.cache_sizes.values()) / (1024 * 1024)
        }

class ConcurrentProcessor:
    """High-performance concurrent processing engine."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=mp.cpu_count() or 1)
        self.task_queue = Queue()
        self.result_cache = AdvancedCache()
        self.active_tasks = 0
        self.completed_tasks = 0
        self.lock = threading.Lock()
    
    async def process_batch_async(self, texts: List[str], model_func: Callable, 
                                 use_cache: bool = True) -> List[Dict[str, Any]]:
        """Process batch of texts asynchronously with caching."""
        results = []
        cache_tasks = []
        compute_tasks = []
        
        # Check cache first
        for i, text in enumerate(texts):
            if use_cache:
                cached_result = self.result_cache.get(text)
                if cached_result:
                    results.append((i, cached_result))
                    continue
            
            compute_tasks.append((i, text))
        
        # Process uncached items concurrently
        if compute_tasks:
            loop = asyncio.get_event_loop()
            
            async def process_item(index, text):
                with self.lock:
                    self.active_tasks += 1
                
                try:
                    # Run CPU-intensive work in thread pool
                    result = await loop.run_in_executor(
                        self.thread_pool, model_func, text
                    )
                    
                    if use_cache:
                        self.result_cache.put(text, result)
                    
                    return (index, result)
                finally:
                    with self.lock:
                        self.active_tasks -= 1
                        self.completed_tasks += 1
            
            # Process all uncached items concurrently
            compute_results = await asyncio.gather(*[
                process_item(index, text) for index, text in compute_tasks
            ])
            
            results.extend(compute_results)
        
        # Sort results by original index
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def process_batch_sync(self, texts: List[str], model_func: Callable,
                          use_cache: bool = True) -> List[Dict[str, Any]]:
        """Synchronous batch processing with threading."""
        results = [None] * len(texts)
        uncached_items = []
        
        # Check cache first
        for i, text in enumerate(texts):
            if use_cache:
                cached_result = self.result_cache.get(text)
                if cached_result:
                    results[i] = cached_result
                    continue
            
            uncached_items.append((i, text))
        
        # Process uncached items in parallel
        if uncached_items:
            def process_item(item):
                index, text = item
                with self.lock:
                    self.active_tasks += 1
                
                try:
                    result = model_func(text)
                    if use_cache:
                        self.result_cache.put(text, result)
                    return (index, result)
                finally:
                    with self.lock:
                        self.active_tasks -= 1
                        self.completed_tasks += 1
            
            # Use thread pool for I/O bound tasks
            computed_results = list(self.thread_pool.map(process_item, uncached_items))
            
            # Fill in computed results
            for index, result in computed_results:
                results[index] = result
        
        return results
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        cache_stats = self.result_cache.get_stats()
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            request_rate=self.completed_tasks / max(time.time() - getattr(self, 'start_time', time.time()), 1),
            avg_response_time=0.1,  # Placeholder - would track in real implementation
            cache_hit_rate=cache_stats['hit_rate'],
            active_connections=self.active_tasks,
            queue_depth=self.task_queue.qsize()
        )
    
    def cleanup(self):
        """Clean up resources."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 32):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.scaling_history = []
        self.last_scale_time = time.time()
        self.scale_cooldown = 30  # seconds
    
    def should_scale_up(self, metrics: PerformanceMetrics) -> bool:
        """Determine if we should scale up."""
        return (
            metrics.cpu_usage > 80 or
            metrics.queue_depth > 10 or
            metrics.avg_response_time > 2.0
        ) and self.current_workers < self.max_workers
    
    def should_scale_down(self, metrics: PerformanceMetrics) -> bool:
        """Determine if we should scale down."""
        return (
            metrics.cpu_usage < 30 and
            metrics.queue_depth < 2 and
            metrics.avg_response_time < 0.5
        ) and self.current_workers > self.min_workers
    
    def scale(self, processor: ConcurrentProcessor) -> Optional[str]:
        """Perform scaling decision."""
        current_time = time.time()
        if current_time - self.last_scale_time < self.scale_cooldown:
            return None
        
        metrics = processor.get_performance_metrics()
        action = None
        
        if self.should_scale_up(metrics):
            new_workers = min(self.current_workers * 2, self.max_workers)
            if new_workers > self.current_workers:
                self.current_workers = new_workers
                action = f"Scaled UP to {new_workers} workers"
                self.last_scale_time = current_time
        
        elif self.should_scale_down(metrics):
            new_workers = max(self.current_workers // 2, self.min_workers)
            if new_workers < self.current_workers:
                self.current_workers = new_workers
                action = f"Scaled DOWN to {new_workers} workers"
                self.last_scale_time = current_time
        
        if action:
            self.scaling_history.append({
                "timestamp": current_time,
                "action": action,
                "metrics": metrics.to_dict()
            })
        
        return action

class PerformanceOptimizer:
    """Comprehensive performance optimization engine."""
    
    def __init__(self):
        self.processor = ConcurrentProcessor()
        self.autoscaler = AutoScaler()
        self.start_time = time.time()
        self.processor.start_time = self.start_time
    
    @lru_cache(maxsize=1000)
    def _preprocess_text_cached(self, text: str) -> str:
        """Cached text preprocessing."""
        # Simulate preprocessing
        return text.strip().lower()
    
    def optimize_memory(self):
        """Perform memory optimization."""
        # Force garbage collection
        collected = gc.collect()
        
        # Clear LRU cache if memory usage is high
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 85:
            self._preprocess_text_cached.cache_clear()
        
        return {
            "garbage_collected": collected,
            "memory_usage_percent": memory_usage,
            "cache_cleared": memory_usage > 85
        }
    
    def benchmark_performance(self, test_texts: List[str], model_func: Callable) -> Dict[str, Any]:
        """Comprehensive performance benchmarking."""
        print("Starting Performance Benchmark...")
        
        # Test sequential processing
        start_time = time.time()
        sequential_results = [model_func(text) for text in test_texts]
        sequential_time = time.time() - start_time
        
        # Test concurrent processing
        start_time = time.time()
        concurrent_results = self.processor.process_batch_sync(test_texts, model_func)
        concurrent_time = time.time() - start_time
        
        # Test with caching (second run)
        start_time = time.time()
        cached_results = self.processor.process_batch_sync(test_texts, model_func)
        cached_time = time.time() - start_time
        
        # Memory optimization
        memory_stats = self.optimize_memory()
        
        # Auto-scaling test
        scaling_action = self.autoscaler.scale(self.processor)
        
        # Performance metrics
        metrics = self.processor.get_performance_metrics()
        cache_stats = self.processor.result_cache.get_stats()
        
        return {
            "sequential_time": sequential_time,
            "concurrent_time": concurrent_time,
            "cached_time": cached_time,
            "speedup_concurrent": sequential_time / max(concurrent_time, 0.001),
            "speedup_cached": sequential_time / max(cached_time, 0.001),
            "cache_stats": cache_stats,
            "memory_stats": memory_stats,
            "scaling_action": scaling_action,
            "performance_metrics": metrics.to_dict(),
            "scaling_history": self.autoscaler.scaling_history
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.processor.cleanup()

def mock_sentiment_model(text: str) -> Dict[str, Any]:
    """Mock sentiment analysis model for testing."""
    # Simulate processing time
    time.sleep(0.01)
    
    # Simple rule-based sentiment for testing
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible']
    
    text_lower = text.lower()
    if any(word in text_lower for word in positive_words):
        sentiment = 'positive'
        confidence = 0.9
    elif any(word in text_lower for word in negative_words):
        sentiment = 'negative'
        confidence = 0.9
    else:
        sentiment = 'neutral'
        confidence = 0.7
    
    return {
        "prediction": sentiment,
        "confidence": confidence,
        "processing_time": 0.01
    }

def test_scaling_optimizations():
    """Test all scaling and optimization features."""
    print("Testing Scaling and Performance Optimizations...")
    
    # Create test data
    test_texts = [
        "This is a great product!",
        "I love this service",
        "Terrible experience",
        "Not bad, could be better",
        "Amazing quality and fast delivery",
        "Worst purchase ever",
        "Pretty good overall",
        "Excellent customer support",
        "Poor quality for the price",
        "Highly recommended!"
    ] * 5  # 50 texts total
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer()
    
    try:
        # Run comprehensive benchmark
        results = optimizer.benchmark_performance(test_texts, mock_sentiment_model)
        
        print(f"\n📊 Performance Benchmark Results:")
        print(f"   Sequential Time: {results['sequential_time']:.3f}s")
        print(f"   Concurrent Time: {results['concurrent_time']:.3f}s")
        print(f"   Cached Time: {results['cached_time']:.3f}s")
        print(f"   Concurrent Speedup: {results['speedup_concurrent']:.2f}x")
        print(f"   Cache Speedup: {results['speedup_cached']:.2f}x")
        
        print(f"\n💾 Cache Performance:")
        cache_stats = results['cache_stats']
        print(f"   Hit Rate: {cache_stats['hit_rate']:.2%}")
        print(f"   Total Entries: {cache_stats['total_entries']}")
        print(f"   Memory Usage: {cache_stats['memory_usage_mb']:.2f} MB")
        
        print(f"\n🖥️ System Metrics:")
        metrics = results['performance_metrics']
        print(f"   CPU Usage: {metrics['cpu_usage']:.1f}%")
        print(f"   Memory Usage: {metrics['memory_usage']:.1f}%")
        print(f"   Request Rate: {metrics['request_rate']:.2f}/s")
        
        if results['scaling_action']:
            print(f"\n🔄 Auto-Scaling: {results['scaling_action']}")
        
        print(f"\n🧹 Memory Optimization:")
        mem_stats = results['memory_stats']
        print(f"   Garbage Collected: {mem_stats['garbage_collected']} objects")
        print(f"   Cache Cleared: {mem_stats['cache_cleared']}")
        
        print("\n✅ Scaling optimizations test completed successfully!")
        
    finally:
        optimizer.cleanup()

if __name__ == "__main__":
    test_scaling_optimizations()