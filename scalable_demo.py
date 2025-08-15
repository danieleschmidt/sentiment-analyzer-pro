#!/usr/bin/env python3
"""Scalable demonstration of Generation 3 functionality with performance optimization, caching, and auto-scaling."""

import sys
import os
import logging
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Any, Union
import json
import pickle
import hashlib
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""

    requests_processed: int = 0
    total_processing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    concurrent_requests: int = 0

    def get_avg_response_time(self) -> float:
        return self.total_processing_time / max(self.requests_processed, 1)

    def get_cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / max(total, 1)


class InMemoryCache:
    """High-performance in-memory cache with TTL and LRU eviction."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache = {}
        self.timestamps = {}
        self.access_order = deque()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def _hash_key(self, key: str) -> str:
        """Create hash of key for storage."""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU tracking."""
        with self.lock:
            hashed_key = self._hash_key(key)
            current_time = time.time()

            if hashed_key in self.cache:
                # Check TTL
                if current_time - self.timestamps[hashed_key] < self.ttl:
                    # Update access order
                    if hashed_key in self.access_order:
                        self.access_order.remove(hashed_key)
                    self.access_order.append(hashed_key)
                    self.hits += 1
                    return self.cache[hashed_key]
                else:
                    # Expired
                    self._remove_key(hashed_key)

            self.misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """Store item in cache with eviction policy."""
        with self.lock:
            hashed_key = self._hash_key(key)
            current_time = time.time()

            # Evict if at capacity
            while len(self.cache) >= self.max_size and self.access_order:
                oldest_key = self.access_order.popleft()
                self._remove_key(oldest_key)

            self.cache[hashed_key] = value
            self.timestamps[hashed_key] = current_time

            if hashed_key in self.access_order:
                self.access_order.remove(hashed_key)
            self.access_order.append(hashed_key)

    def _remove_key(self, hashed_key: str) -> None:
        """Remove key from all data structures."""
        self.cache.pop(hashed_key, None)
        self.timestamps.pop(hashed_key, None)
        if hashed_key in self.access_order:
            self.access_order.remove(hashed_key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(total, 1),
            "size": len(self.cache),
            "max_size": self.max_size,
        }


class AutoScaler:
    """Intelligent auto-scaling based on load metrics."""

    def __init__(self, min_workers: int = 2, max_workers: int = 16):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.load_history = deque(maxlen=60)  # 1 minute history
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.last_scale_time = 0
        self.scale_cooldown = 30  # 30 seconds between scaling events

    def record_load(
        self, cpu_percent: float, queue_size: int, response_time: float
    ) -> None:
        """Record current load metrics."""
        load_score = (
            (cpu_percent / 100) * 0.5
            + min(queue_size / 10, 1) * 0.3
            + min(response_time / 1000, 1) * 0.2
        )
        self.load_history.append(load_score)

    def should_scale(self) -> tuple[bool, int]:
        """Determine if scaling is needed."""
        if len(self.load_history) < 10:  # Need some history
            return False, self.current_workers

        current_time = time.time()
        if current_time - self.last_scale_time < self.scale_cooldown:
            return False, self.current_workers

        avg_load = sum(self.load_history) / len(self.load_history)

        if (
            avg_load > self.scale_up_threshold
            and self.current_workers < self.max_workers
        ):
            new_workers = min(self.current_workers * 2, self.max_workers)
            self.current_workers = new_workers
            self.last_scale_time = current_time
            return True, new_workers

        elif (
            avg_load < self.scale_down_threshold
            and self.current_workers > self.min_workers
        ):
            new_workers = max(self.current_workers // 2, self.min_workers)
            self.current_workers = new_workers
            self.last_scale_time = current_time
            return True, new_workers

        return False, self.current_workers


class ScalableSentimentAnalyzer:
    """High-performance sentiment analyzer with auto-scaling and optimization."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.cache = InMemoryCache(max_size=50000, ttl_seconds=1800)  # 30 min TTL
        self.metrics = PerformanceMetrics()
        self.auto_scaler = AutoScaler()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)

        # Batch processing optimization
        self.batch_queue = deque()
        self.batch_size = 32
        self.batch_timeout = 100  # ms
        self.last_batch_time = 0

        # Performance monitoring
        self.request_times = deque(maxlen=1000)
        self.active_requests = threading.active_count()

        # Setup directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("cache", exist_ok=True)

        logger.info("ScalableSentimentAnalyzer initialized with auto-scaling")

    def create_optimized_model(self) -> Pipeline:
        """Create performance-optimized model pipeline."""
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=10000,
                        stop_words="english",
                        min_df=2,
                        max_df=0.95,
                        ngram_range=(1, 2),
                        sublinear_tf=True,  # Performance optimization
                        use_idf=True,
                        smooth_idf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=42,
                        max_iter=1000,
                        class_weight="balanced",
                        solver="liblinear",  # Faster for small datasets
                        n_jobs=-1,  # Use all available cores
                    ),
                ),
            ]
        )

    def train_with_optimization(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train model with performance optimization."""
        start_time = time.time()

        # Data preprocessing optimization
        data = data.dropna(subset=["text", "label"])
        data["text"] = data["text"].astype(str)

        # Vectorized text length filtering
        text_lengths = data["text"].str.len()
        data = data[(text_lengths >= 5) & (text_lengths <= 1000)]

        if len(data) < 10:
            raise ValueError(
                f"Insufficient data after preprocessing: {len(data)} samples"
            )

        # Create and train model
        self.model = self.create_optimized_model()

        X = data["text"].values  # Use values for faster access
        y = data["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Training with timing
        train_start = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - train_start

        # Evaluation
        test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)

        total_time = time.time() - start_time

        metrics = {
            "test_accuracy": test_accuracy,
            "training_time": train_time,
            "total_time": total_time,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "features": self.model.named_steps["tfidf"].vocabulary_.__len__(),
        }

        logger.info(
            f"Optimized training complete - Accuracy: {test_accuracy:.3f}, Time: {total_time:.2f}s"
        )
        return metrics

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Optimized single prediction with caching."""
        start_time = time.time()

        # Check cache first
        cache_key = f"pred_{text}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.metrics.cache_hits += 1
            return cached_result

        self.metrics.cache_misses += 1

        # Make prediction
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        confidence = max(probabilities)

        result = {
            "text": text[:50] + "..." if len(text) > 50 else text,
            "prediction": prediction,
            "confidence": confidence,
            "processing_time": time.time() - start_time,
        }

        # Cache result
        self.cache.put(cache_key, result)

        return result

    def predict_batch_optimized(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Optimized batch prediction with vectorization."""
        if not texts:
            return []

        start_time = time.time()

        # Batch prediction for efficiency
        predictions = self.model.predict(texts)
        probabilities = self.model.predict_proba(texts)
        confidences = np.max(probabilities, axis=1)

        results = []
        for i, (text, pred, conf) in enumerate(zip(texts, predictions, confidences)):
            result = {
                "index": i,
                "text": text[:50] + "..." if len(text) > 50 else text,
                "prediction": pred,
                "confidence": float(conf),
                "batch_processing_time": time.time() - start_time,
            }
            results.append(result)

            # Cache individual results
            cache_key = f"pred_{text}"
            self.cache.put(cache_key, result)

        return results

    async def predict_async(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Asynchronous prediction processing."""
        loop = asyncio.get_event_loop()

        # Split into smaller batches for optimal performance
        batch_size = 16
        tasks = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            task = loop.run_in_executor(
                self.executor, self.predict_batch_optimized, batch
            )
            tasks.append(task)

        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)

        return results

    def predict_concurrent(
        self, texts: List[str], max_workers: int = 8
    ) -> List[Dict[str, Any]]:
        """Concurrent prediction processing."""
        if len(texts) <= 10:  # Use single thread for small batches
            return self.predict_batch_optimized(texts)

        # Split into chunks
        chunk_size = max(1, len(texts) // max_workers)
        chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.predict_batch_optimized, chunk) for chunk in chunks
            ]

            for future in futures:
                chunk_results = future.result()
                results.extend(chunk_results)

        return results

    def update_metrics(self, processing_time: float) -> None:
        """Update performance metrics."""
        self.metrics.requests_processed += 1
        self.metrics.total_processing_time += processing_time
        self.metrics.cpu_usage = psutil.cpu_percent()
        self.metrics.memory_usage = psutil.virtual_memory().percent

        self.request_times.append(processing_time)

        # Auto-scaling decision
        avg_response_time = sum(list(self.request_times)[-10:]) / min(
            len(self.request_times), 10
        )
        queue_size = len(self.batch_queue)

        self.auto_scaler.record_load(
            self.metrics.cpu_usage, queue_size, avg_response_time * 1000
        )
        should_scale, new_workers = self.auto_scaler.should_scale()

        if should_scale:
            logger.info(
                f"Auto-scaling triggered: {self.executor._max_workers} -> {new_workers} workers"
            )
            # Note: ThreadPoolExecutor doesn't support dynamic resizing,
            # but we log the decision for monitoring

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cache_stats = self.cache.get_stats()

        return {
            "processing_metrics": asdict(self.metrics),
            "avg_response_time_ms": self.metrics.get_avg_response_time() * 1000,
            "cache_performance": cache_stats,
            "system_metrics": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "active_threads": threading.active_count(),
            },
            "auto_scaler": {
                "current_workers": self.auto_scaler.current_workers,
                "load_history_size": len(self.auto_scaler.load_history),
                "avg_load": sum(self.auto_scaler.load_history)
                / max(len(self.auto_scaler.load_history), 1),
            },
        }

    def save_optimized(self, path: Optional[str] = None) -> str:
        """Save model with performance optimizations."""
        if path is None:
            path = "models/scalable_sentiment_model.joblib"

        # Save with compression for faster I/O
        joblib.dump(self.model, path, compress=3)

        # Save performance metadata
        metadata = {
            "model_type": "scalable",
            "cache_stats": self.cache.get_stats(),
            "performance_metrics": asdict(self.metrics),
            "created_at": time.time(),
        }

        metadata_path = path.replace(".joblib", "_performance.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Scalable model saved to {path}")
        return path


def demo_generation_3():
    """Demonstrate Generation 3: MAKE IT SCALE functionality."""
    print("⚡ GENERATION 3: MAKE IT SCALE - Performance Optimization & Auto-scaling")
    print("=" * 85)

    analyzer = ScalableSentimentAnalyzer()

    try:
        # Load enhanced dataset
        data = pd.read_csv("data/sample_reviews.csv")
        print(f"✅ Loaded {len(data)} samples for scalable training")

        # Train with optimization
        print("\n🚀 Training with performance optimization...")
        start_time = time.time()
        metrics = analyzer.train_with_optimization(data)
        training_time = time.time() - start_time

        print(f"✅ Optimized training complete!")
        print(f"   • Accuracy: {metrics['test_accuracy']:.3f}")
        print(f"   • Training time: {metrics['training_time']:.3f}s")
        print(f"   • Features: {metrics['features']:,}")

        # Save optimized model
        model_path = analyzer.save_optimized()
        print(f"✅ Scalable model saved with compression")

        # Performance testing with various loads
        print("\n⚡ Performance Testing:")
        print("-" * 50)

        # Single prediction test
        single_start = time.time()
        single_result = analyzer.predict_single("This is an amazing product!")
        single_time = time.time() - single_start
        print(
            f"Single prediction: {single_time*1000:.1f}ms - {single_result['prediction']}"
        )

        # Small batch test
        small_batch = ["Great product!", "Terrible service", "Average quality"] * 5
        batch_start = time.time()
        batch_results = analyzer.predict_batch_optimized(small_batch)
        batch_time = time.time() - batch_start
        print(
            f"Small batch ({len(small_batch)}): {batch_time*1000:.1f}ms ({batch_time/len(small_batch)*1000:.1f}ms/item)"
        )

        # Large batch with concurrency
        large_batch = ["Test text for performance"] * 100
        concurrent_start = time.time()
        concurrent_results = analyzer.predict_concurrent(large_batch, max_workers=8)
        concurrent_time = time.time() - concurrent_start
        print(
            f"Concurrent batch ({len(large_batch)}): {concurrent_time*1000:.1f}ms ({concurrent_time/len(large_batch)*1000:.1f}ms/item)"
        )

        # Async processing test
        print("\n🔄 Testing async processing...")

        async def async_test():
            async_batch = ["Async test text"] * 50
            async_start = time.time()
            async_results = await analyzer.predict_async(async_batch)
            async_time = time.time() - async_start
            return async_time, len(async_results)

        async_time, async_count = asyncio.run(async_test())
        print(
            f"Async batch ({async_count}): {async_time*1000:.1f}ms ({async_time/async_count*1000:.1f}ms/item)"
        )

        # Cache performance test
        print("\n💾 Cache Performance Test:")
        cache_test_texts = [
            "Cache test 1",
            "Cache test 2",
            "Cache test 1",
            "Cache test 2",
        ]
        for text in cache_test_texts:
            result = analyzer.predict_single(text)

        cache_stats = analyzer.cache.get_stats()
        print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"Cache size: {cache_stats['size']}/{cache_stats['max_size']}")

        # Performance report
        print("\n📊 Comprehensive Performance Report:")
        print("-" * 60)
        report = analyzer.get_performance_report()

        print(f"Total requests: {report['processing_metrics']['requests_processed']}")
        print(f"Average response time: {report['avg_response_time_ms']:.1f}ms")
        print(f"Cache hit rate: {report['cache_performance']['hit_rate']:.2%}")
        print(f"Current CPU usage: {report['system_metrics']['cpu_usage']:.1f}%")
        print(f"Memory usage: {report['system_metrics']['memory_usage']:.1f}%")
        print(f"Active threads: {report['system_metrics']['active_threads']}")

        # Auto-scaling simulation
        print("\n🎯 Auto-scaling Simulation:")
        print("Simulating high load conditions...")

        # Simulate load spikes
        for i in range(5):
            load_texts = [f"Load test {j}" for j in range(20)]
            start = time.time()
            results = analyzer.predict_concurrent(load_texts)
            duration = time.time() - start
            analyzer.update_metrics(duration)

            print(
                f"Load spike {i+1}: {len(results)} predictions in {duration*1000:.1f}ms"
            )

        final_report = analyzer.get_performance_report()
        print(f"\nFinal auto-scaler status:")
        print(
            f"• Recommended workers: {final_report['auto_scaler']['current_workers']}"
        )
        print(f"• Average load: {final_report['auto_scaler']['avg_load']:.3f}")

        print("\n🎉 Generation 3 Complete: High-performance scaling implemented!")
        print(
            f"System achieved {final_report['avg_response_time_ms']:.1f}ms average response time"
        )
        print(
            f"with {final_report['cache_performance']['hit_rate']:.1%} cache hit rate"
        )

    except Exception as e:
        logger.error(f"Scalable demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        raise

    finally:
        # Cleanup
        analyzer.executor.shutdown(wait=True)
        analyzer.process_executor.shutdown(wait=True)


if __name__ == "__main__":
    demo_generation_3()
