"""
Photonic-MLIR Bridge Performance Optimization Module

Provides advanced optimization features including caching, concurrent processing,
resource pooling, and intelligent circuit optimization algorithms.
"""

import time
import json
import hashlib
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache, wraps
import pickle
from pathlib import Path
import weakref

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels for synthesis."""
    O0 = 0  # No optimization
    O1 = 1  # Basic optimization
    O2 = 2  # Standard optimization
    O3 = 3  # Aggressive optimization


class CachePolicy(Enum):
    """Cache replacement policies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"


@dataclass
class CacheEntry:
    """Represents a cache entry."""
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.timestamp > self.ttl_seconds
    
    def access(self):
        """Record an access to this cache entry."""
        self.access_count += 1


class SmartCache:
    """Intelligent cache with multiple replacement policies."""
    
    def __init__(self, max_size: int = 1000, policy: CachePolicy = CachePolicy.LRU,
                 default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.policy = policy
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._access_order: List[str] = []  # For LRU
        
        # Start cleanup thread for TTL policy
        if policy == CachePolicy.TTL or default_ttl is not None:
            self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
            self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None
            
            # Update access tracking
            entry.access()
            
            if self.policy == CachePolicy.LRU:
                # Move to end of access order
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value into cache."""
        with self._lock:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
            
            # Create cache entry
            entry = CacheEntry(key=key, value=value, ttl_seconds=ttl)
            
            # If key exists, update it
            if key in self.cache:
                self.cache[key] = entry
                if self.policy == CachePolicy.LRU and key in self._access_order:
                    self._access_order.remove(key)
                    self._access_order.append(key)
                return
            
            # Check if we need to evict
            if len(self.cache) >= self.max_size:
                self._evict()
            
            # Add new entry
            self.cache[key] = entry
            if self.policy == CachePolicy.LRU:
                self._access_order.append(key)
    
    def _evict(self) -> None:
        """Evict entries based on policy."""
        if not self.cache:
            return
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used
            if self._access_order:
                key_to_remove = self._access_order.pop(0)
                if key_to_remove in self.cache:
                    del self.cache[key_to_remove]
        
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            min_access_count = min(entry.access_count for entry in self.cache.values())
            for key, entry in list(self.cache.items()):
                if entry.access_count == min_access_count:
                    del self.cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)
                    break
        
        elif self.policy == CachePolicy.TTL:
            # Remove expired entries first, then oldest
            now = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.ttl_seconds and now - entry.timestamp > entry.ttl_seconds
            ]
            
            if expired_keys:
                for key in expired_keys:
                    del self.cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)
            else:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
                del self.cache[oldest_key]
                if oldest_key in self._access_order:
                    self._access_order.remove(oldest_key)
    
    def _cleanup_expired(self) -> None:
        """Background thread to cleanup expired entries."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                with self._lock:
                    expired_keys = [
                        key for key, entry in list(self.cache.items())
                        if entry.is_expired()
                    ]
                    for key in expired_keys:
                        del self.cache[key]
                        if key in self._access_order:
                            self._access_order.remove(key)
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self._access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            if not self.cache:
                return {"size": 0, "hit_rate": 0.0}
            
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "policy": self.policy.value,
                "total_accesses": total_accesses,
                "avg_access_count": total_accesses / len(self.cache) if self.cache else 0
            }


class CachingOptimizer:
    """Optimizes synthesis operations using intelligent caching."""
    
    def __init__(self, cache_size: int = 1000, cache_ttl: float = 3600):
        self.synthesis_cache = SmartCache(cache_size, CachePolicy.LRU, cache_ttl)
        self.validation_cache = SmartCache(cache_size // 2, CachePolicy.TTL, cache_ttl // 2)
        self.component_cache = SmartCache(cache_size * 2, CachePolicy.LFU)
        
    def get_circuit_hash(self, circuit) -> str:
        """Generate hash for circuit for caching."""
        # Create a deterministic representation of the circuit
        circuit_data = {
            "name": circuit.name,
            "components": [
                {
                    "id": comp.id,
                    "type": comp.component_type.value,
                    "position": comp.position,
                    "parameters": comp.parameters,
                    "wavelength_band": comp.wavelength_band.value
                }
                for comp in sorted(circuit.components, key=lambda c: c.id)
            ],
            "connections": [
                {
                    "source": conn.source_component,
                    "target": conn.target_component,
                    "source_port": conn.source_port,
                    "target_port": conn.target_port,
                    "loss_db": conn.loss_db,
                    "delay_ps": conn.delay_ps
                }
                for conn in sorted(circuit.connections, key=lambda c: (c.source_component, c.target_component))
            ]
        }
        
        # Generate hash
        circuit_json = json.dumps(circuit_data, sort_keys=True)
        return hashlib.sha256(circuit_json.encode()).hexdigest()
    
    def cached_synthesis(self, synthesis_func: Callable, circuit, optimization_level: OptimizationLevel = OptimizationLevel.O2):
        """Perform cached synthesis operation."""
        # Generate cache key
        circuit_hash = self.get_circuit_hash(circuit)
        cache_key = f"{circuit_hash}_{optimization_level.value}"
        
        # Check cache first
        cached_result = self.synthesis_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for synthesis: {circuit.name}")
            return cached_result
        
        # Perform synthesis
        logger.debug(f"Cache miss for synthesis: {circuit.name}")
        start_time = time.time()
        result = synthesis_func(circuit)
        synthesis_time = time.time() - start_time
        
        # Add timing information
        result["cache_info"] = {
            "cache_hit": False,
            "synthesis_time": synthesis_time,
            "cache_key": cache_key
        }
        
        # Cache the result
        self.synthesis_cache.put(cache_key, result)
        
        return result
    
    def cached_validation(self, validation_func: Callable, circuit) -> bool:
        """Perform cached validation operation."""
        circuit_hash = self.get_circuit_hash(circuit)
        
        cached_result = self.validation_cache.get(circuit_hash)
        if cached_result is not None:
            logger.debug(f"Cache hit for validation: {circuit.name}")
            return cached_result
        
        # Perform validation
        logger.debug(f"Cache miss for validation: {circuit.name}")
        result = validation_func()
        
        # Cache the result
        self.validation_cache.put(circuit_hash, result, ttl=1800)  # 30 minutes
        
        return result
    
    def optimize_component_placement(self, circuit) -> Dict[str, Tuple[float, float]]:
        """Optimize component placement for better routing."""
        placement_key = f"placement_{self.get_circuit_hash(circuit)}"
        
        cached_placement = self.component_cache.get(placement_key)
        if cached_placement is not None:
            return cached_placement
        
        # Simple optimization: minimize total connection length
        optimized_positions = {}
        
        # Start with current positions
        for comp in circuit.components:
            optimized_positions[comp.id] = comp.position
        
        # Iterative improvement
        for iteration in range(10):  # Max 10 iterations
            improved = False
            
            for comp in circuit.components:
                # Find all connections for this component
                connected_components = []
                for conn in circuit.connections:
                    if conn.source_component == comp.id:
                        target_comp = next(c for c in circuit.components if c.id == conn.target_component)
                        connected_components.append(target_comp)
                    elif conn.target_component == comp.id:
                        source_comp = next(c for c in circuit.components if c.id == conn.source_component)
                        connected_components.append(source_comp)
                
                if connected_components:
                    # Calculate centroid of connected components
                    centroid_x = sum(optimized_positions[c.id][0] for c in connected_components) / len(connected_components)
                    centroid_y = sum(optimized_positions[c.id][1] for c in connected_components) / len(connected_components)
                    
                    # Move towards centroid (but not all the way)
                    current_pos = optimized_positions[comp.id]
                    new_x = current_pos[0] + 0.1 * (centroid_x - current_pos[0])
                    new_y = current_pos[1] + 0.1 * (centroid_y - current_pos[1])
                    
                    if abs(new_x - current_pos[0]) > 0.01 or abs(new_y - current_pos[1]) > 0.01:
                        optimized_positions[comp.id] = (new_x, new_y)
                        improved = True
            
            if not improved:
                break
        
        # Cache the result
        self.component_cache.put(placement_key, optimized_positions)
        
        return optimized_positions
    
    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            "synthesis_cache": self.synthesis_cache.get_stats(),
            "validation_cache": self.validation_cache.get_stats(),
            "component_cache": self.component_cache.get_stats()
        }


class ConcurrentProcessor:
    """Handles concurrent processing of synthesis operations."""
    
    def __init__(self, max_workers: int = None, use_processes: bool = False):
        self.max_workers = max_workers or min(32, (threading.active_count() or 1) + 4)
        self.use_processes = use_processes
        self._executor = None
        self._lock = threading.Lock()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._executor:
            self._executor.shutdown(wait=True)
    
    def _get_executor(self):
        """Get or create executor."""
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    if self.use_processes:
                        self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
                    else:
                        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor
    
    def process_circuits_parallel(self, circuits: List, synthesis_func: Callable,
                                optimization_level: OptimizationLevel = OptimizationLevel.O2) -> List[Dict[str, Any]]:
        """Process multiple circuits in parallel."""
        if not circuits:
            return []
        
        executor = self._get_executor()
        results = []
        
        # Submit all tasks
        future_to_circuit = {
            executor.submit(self._safe_synthesis, synthesis_func, circuit, optimization_level): circuit
            for circuit in circuits
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_circuit):
            circuit = future_to_circuit[future]
            try:
                result = future.result()
                result["circuit_name"] = circuit.name
                results.append(result)
                logger.debug(f"Completed synthesis for circuit: {circuit.name}")
            except Exception as e:
                logger.error(f"Failed to synthesize circuit {circuit.name}: {e}")
                results.append({
                    "circuit_name": circuit.name,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def _safe_synthesis(self, synthesis_func: Callable, circuit, optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """Safely perform synthesis with error handling."""
        try:
            start_time = time.time()
            result = synthesis_func(circuit)
            processing_time = time.time() - start_time
            
            result["processing_time"] = processing_time
            result["optimization_level"] = optimization_level.value
            result["success"] = True
            
            return result
        except Exception as e:
            logger.error(f"Synthesis failed for circuit {circuit.name}: {e}")
            return {
                "error": str(e),
                "success": False,
                "optimization_level": optimization_level.value
            }
    
    def batch_optimize_circuits(self, circuits: List, optimizer_func: Callable) -> List[Any]:
        """Optimize multiple circuits in parallel."""
        if not circuits:
            return []
        
        executor = self._get_executor()
        
        # Submit optimization tasks
        futures = [executor.submit(optimizer_func, circuit) for circuit in circuits]
        
        # Collect results
        results = []
        for future, circuit in zip(futures, circuits):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Optimization failed for circuit {circuit.name}: {e}")
                results.append(None)
        
        return results


class ResourcePool:
    """Manages pools of reusable resources."""
    
    def __init__(self, resource_factory: Callable, max_size: int = 10):
        self.resource_factory = resource_factory
        self.max_size = max_size
        self.pool: List[Any] = []
        self.in_use: Set[Any] = set()
        self._lock = threading.Lock()
    
    def acquire(self) -> Any:
        """Acquire a resource from the pool."""
        with self._lock:
            if self.pool:
                resource = self.pool.pop()
                self.in_use.add(resource)
                return resource
            
            # Create new resource if under limit
            if len(self.in_use) < self.max_size:
                resource = self.resource_factory()
                self.in_use.add(resource)
                return resource
            
            # Pool exhausted
            raise RuntimeError("Resource pool exhausted")
    
    def release(self, resource: Any) -> None:
        """Release a resource back to the pool."""
        with self._lock:
            if resource in self.in_use:
                self.in_use.remove(resource)
                if len(self.pool) < self.max_size:
                    self.pool.append(resource)
    
    def size(self) -> Tuple[int, int]:
        """Get pool size (available, in_use)."""
        with self._lock:
            return len(self.pool), len(self.in_use)


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, cache_size: int = 1000, max_workers: int = None):
        self.caching_optimizer = CachingOptimizer(cache_size)
        self.concurrent_processor = ConcurrentProcessor(max_workers)
        self.resource_pools: Dict[str, ResourcePool] = {}
        
        # Performance metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "parallel_batches_processed": 0,
            "total_synthesis_time": 0.0,
            "total_circuits_processed": 0
        }
        self._metrics_lock = threading.Lock()
    
    def create_resource_pool(self, name: str, factory: Callable, max_size: int = 10):
        """Create a named resource pool."""
        self.resource_pools[name] = ResourcePool(factory, max_size)
    
    def optimized_synthesis(self, synthesis_func: Callable, circuit, 
                          optimization_level: OptimizationLevel = OptimizationLevel.O2):
        """Perform optimized synthesis with caching."""
        start_time = time.time()
        
        # Use cached synthesis
        result = self.caching_optimizer.cached_synthesis(
            synthesis_func, circuit, optimization_level
        )
        
        # Update metrics
        with self._metrics_lock:
            self.metrics["total_synthesis_time"] += time.time() - start_time
            self.metrics["total_circuits_processed"] += 1
            
            if result.get("cache_info", {}).get("cache_hit", False):
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1
        
        return result
    
    def batch_process_circuits(self, circuits: List, synthesis_func: Callable,
                             optimization_level: OptimizationLevel = OptimizationLevel.O2) -> List[Dict[str, Any]]:
        """Process multiple circuits with full optimization."""
        if not circuits:
            return []
        
        start_time = time.time()
        
        # Process in parallel
        results = self.concurrent_processor.process_circuits_parallel(
            circuits, synthesis_func, optimization_level
        )
        
        # Update metrics
        with self._metrics_lock:
            self.metrics["parallel_batches_processed"] += 1
            self.metrics["total_synthesis_time"] += time.time() - start_time
            self.metrics["total_circuits_processed"] += len(circuits)
        
        return results
    
    def optimize_circuit_layout(self, circuit):
        """Optimize circuit layout for better performance."""
        return self.caching_optimizer.optimize_component_placement(circuit)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._metrics_lock:
            cache_stats = self.caching_optimizer.get_cache_stats()
            
            total_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
            cache_hit_rate = (self.metrics["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
            
            avg_synthesis_time = (
                self.metrics["total_synthesis_time"] / self.metrics["total_circuits_processed"]
                if self.metrics["total_circuits_processed"] > 0 else 0
            )
            
            return {
                "cache_hit_rate_percent": cache_hit_rate,
                "avg_synthesis_time_seconds": avg_synthesis_time,
                "total_circuits_processed": self.metrics["total_circuits_processed"],
                "parallel_batches_processed": self.metrics["parallel_batches_processed"],
                "cache_stats": cache_stats,
                "resource_pool_sizes": {
                    name: pool.size() for name, pool in self.resource_pools.items()
                }
            }
    
    def clear_caches(self):
        """Clear all caches."""
        self.caching_optimizer.synthesis_cache.clear()
        self.caching_optimizer.validation_cache.clear()
        self.caching_optimizer.component_cache.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.concurrent_processor.__exit__(exc_type, exc_val, exc_tb)


# Global optimizer instance
_global_optimizer = PerformanceOptimizer()


def get_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer."""
    return _global_optimizer


def cached_synthesis(synthesis_func: Callable, circuit, 
                    optimization_level: OptimizationLevel = OptimizationLevel.O2):
    """Convenience function for cached synthesis."""
    return _global_optimizer.optimized_synthesis(synthesis_func, circuit, optimization_level)


def parallel_synthesis(circuits: List, synthesis_func: Callable,
                      optimization_level: OptimizationLevel = OptimizationLevel.O2) -> List[Dict[str, Any]]:
    """Convenience function for parallel synthesis."""
    return _global_optimizer.batch_process_circuits(circuits, synthesis_func, optimization_level)