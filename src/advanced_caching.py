"""Advanced caching system with adaptive algorithms and distributed support."""

import hashlib
import json
import logging
import pickle
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict, defaultdict
import weakref

logger = logging.getLogger(__name__)

class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    ADAPTIVE = "adaptive"  # Adaptive Replacement Cache
    TTL = "ttl"  # Time To Live
    HYBRID = "hybrid"  # Combination of policies

class CacheEvent(Enum):
    """Cache events for monitoring."""
    HIT = "hit"
    MISS = "miss"
    EVICTION = "eviction"
    EXPIRATION = "expiration"
    UPDATE = "update"

@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    memory_usage: int = 0
    avg_access_time: float = 0.0
    hit_rate: float = 0.0
    
    def calculate_hit_rate(self):
        """Calculate hit rate percentage."""
        total = self.hits + self.misses
        self.hit_rate = (self.hits / total * 100) if total > 0 else 0.0

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int
    size: int
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at.timestamp() > self.ttl
    
    def update_access(self):
        """Update access metadata."""
        self.accessed_at = datetime.now()
        self.access_count += 1

class CacheBackend(ABC):
    """Abstract cache backend interface."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value by key."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all entries."""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """Get all keys."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get cache size."""
        pass

class MemoryCache(CacheBackend):
    """Advanced in-memory cache with multiple eviction policies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory: int = 100 * 1024 * 1024,  # 100MB
        policy: CachePolicy = CachePolicy.ADAPTIVE,
        default_ttl: Optional[float] = None
    ):
        self.max_size = max_size
        self.max_memory = max_memory
        self.policy = policy
        self.default_ttl = default_ttl
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = OrderedDict()  # For LRU
        self._frequency = defaultdict(int)  # For LFU
        self._adaptive_list1 = OrderedDict()  # For ARC T1
        self._adaptive_list2 = OrderedDict()  # For ARC T2
        self._ghost_list1 = OrderedDict()  # For ARC B1
        self._ghost_list2 = OrderedDict()  # For ARC B2
        self._p = 0  # ARC parameter
        
        self.metrics = CacheMetrics()
        self._lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.Lock()
        self._event_handlers: Dict[CacheEvent, List[Callable]] = defaultdict(list)
    
    def add_event_handler(self, event: CacheEvent, handler: Callable):
        """Add event handler for cache events."""
        self._event_handlers[event].append(handler)
    
    def _emit_event(self, event: CacheEvent, key: str, data: Dict[str, Any] = None):
        """Emit cache event."""
        for handler in self._event_handlers[event]:
            try:
                handler(event, key, data or {})
            except Exception as e:
                logger.error(f"Cache event handler error: {e}")
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode('utf-8'))
    
    def _cleanup_expired(self):
        """Clean up expired entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key, CacheEvent.EXPIRATION)
    
    def _remove_entry(self, key: str, event: CacheEvent = CacheEvent.EVICTION):
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache[key]
            del self._cache[key]
            
            # Update access structures
            self._access_order.pop(key, None)
            self._frequency.pop(key, None)
            self._adaptive_list1.pop(key, None)
            self._adaptive_list2.pop(key, None)
            
            # Update metrics
            self.metrics.total_size -= 1
            self.metrics.memory_usage -= entry.size
            if event == CacheEvent.EVICTION:
                self.metrics.evictions += 1
            
            self._emit_event(event, key, {"size": entry.size})
    
    def _evict_lru(self):
        """Evict using LRU policy."""
        if self._access_order:
            oldest_key = next(iter(self._access_order))
            self._remove_entry(oldest_key)
    
    def _evict_lfu(self):
        """Evict using LFU policy."""
        if self._frequency:
            min_freq = min(self._frequency.values())
            for key, freq in self._frequency.items():
                if freq == min_freq:
                    self._remove_entry(key)
                    break
    
    def _evict_adaptive(self):
        """Evict using Adaptive Replacement Cache (ARC) policy."""
        # Simplified ARC implementation
        if self._adaptive_list1:
            key = next(iter(self._adaptive_list1))
            self._adaptive_list1.pop(key)
            self._ghost_list1[key] = True
            self._remove_entry(key)
        elif self._adaptive_list2:
            key = next(iter(self._adaptive_list2))
            self._adaptive_list2.pop(key)
            self._ghost_list2[key] = True
            self._remove_entry(key)
    
    def _needs_eviction(self) -> bool:
        """Check if eviction is needed."""
        return (
            len(self._cache) >= self.max_size or
            self.metrics.memory_usage >= self.max_memory
        )
    
    def _evict(self):
        """Perform eviction based on policy."""
        while self._needs_eviction() and self._cache:
            if self.policy == CachePolicy.LRU:
                self._evict_lru()
            elif self.policy == CachePolicy.LFU:
                self._evict_lfu()
            elif self.policy == CachePolicy.ADAPTIVE:
                self._evict_adaptive()
            elif self.policy == CachePolicy.HYBRID:
                # Use LRU for large items, LFU for small items
                if self.metrics.memory_usage > self.max_memory * 0.8:
                    self._evict_lru()
                else:
                    self._evict_lfu()
            else:
                self._evict_lru()  # Default to LRU
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        start_time = time.time()
        
        with self._lock:
            self._cleanup_expired()
            
            if key in self._cache:
                entry = self._cache[key]
                entry.update_access()
                
                # Update access structures
                self._access_order.move_to_end(key)
                self._frequency[key] += 1
                
                # ARC list management
                if key in self._adaptive_list1:
                    self._adaptive_list1.pop(key)
                    self._adaptive_list2[key] = True
                elif key in self._adaptive_list2:
                    self._adaptive_list2.move_to_end(key)
                
                self.metrics.hits += 1
                access_time = time.time() - start_time
                self.metrics.avg_access_time = (
                    self.metrics.avg_access_time * 0.9 + access_time * 0.1
                )
                
                self._emit_event(CacheEvent.HIT, key, {
                    "access_time": access_time,
                    "access_count": entry.access_count
                })
                
                return entry.value
            else:
                self.metrics.misses += 1
                self._emit_event(CacheEvent.MISS, key)
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value with optional TTL."""
        if ttl is None:
            ttl = self.default_ttl
        
        size = self._calculate_size(value)
        
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key, CacheEvent.UPDATE)
            
            # Check if we need to evict
            self._evict()
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=1,
                size=size,
                ttl=ttl
            )
            
            self._cache[key] = entry
            self._access_order[key] = True
            self._frequency[key] = 1
            
            # ARC list management
            if key in self._ghost_list1:
                self._ghost_list1.pop(key)
                self._adaptive_list2[key] = True
                self._p = min(self._p + 1, self.max_size // 2)
            elif key in self._ghost_list2:
                self._ghost_list2.pop(key)
                self._adaptive_list2[key] = True
                self._p = max(self._p - 1, 0)
            else:
                self._adaptive_list1[key] = True
            
            # Update metrics
            self.metrics.total_size += 1
            self.metrics.memory_usage += size
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete value by key."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> bool:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._frequency.clear()
            self._adaptive_list1.clear()
            self._adaptive_list2.clear()
            self._ghost_list1.clear()
            self._ghost_list2.clear()
            
            self.metrics = CacheMetrics()
            return True
    
    def keys(self) -> List[str]:
        """Get all keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)
    
    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics."""
        with self._lock:
            self.metrics.calculate_hit_rate()
            return self.metrics

class DistributedCache:
    """Distributed cache with consistent hashing."""
    
    def __init__(self, nodes: List[str], replication_factor: int = 2):
        self.nodes = nodes
        self.replication_factor = min(replication_factor, len(nodes))
        self.hash_ring = self._build_hash_ring()
        self.local_cache = MemoryCache(max_size=500)  # Local L1 cache
    
    def _build_hash_ring(self) -> Dict[int, str]:
        """Build consistent hash ring."""
        ring = {}
        virtual_nodes = 100  # Virtual nodes per physical node
        
        for node in self.nodes:
            for i in range(virtual_nodes):
                key = f"{node}:{i}"
                hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
                ring[hash_value] = node
        
        return dict(sorted(ring.items()))
    
    def _get_nodes_for_key(self, key: str) -> List[str]:
        """Get nodes responsible for key."""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        # Find first node >= hash_value
        selected_nodes = []
        hash_values = list(self.hash_ring.keys())
        
        for i, ring_hash in enumerate(hash_values):
            if ring_hash >= hash_value:
                start_idx = i
                break
        else:
            start_idx = 0
        
        # Select nodes with replication
        for i in range(self.replication_factor):
            node_idx = (start_idx + i) % len(hash_values)
            node = self.hash_ring[hash_values[node_idx]]
            if node not in selected_nodes:
                selected_nodes.append(node)
        
        return selected_nodes
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache."""
        # Try L1 cache first
        value = self.local_cache.get(key)
        if value is not None:
            return value
        
        # Try distributed nodes
        nodes = self._get_nodes_for_key(key)
        for node in nodes:
            try:
                value = self._get_from_node(node, key)
                if value is not None:
                    # Store in L1 cache
                    self.local_cache.set(key, value, ttl=60)  # 1-minute L1 TTL
                    return value
            except Exception as e:
                logger.error(f"Failed to get from node {node}: {e}")
                continue
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in distributed cache."""
        nodes = self._get_nodes_for_key(key)
        success_count = 0
        
        for node in nodes:
            try:
                if self._set_to_node(node, key, value, ttl):
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to set to node {node}: {e}")
                continue
        
        # Update L1 cache
        self.local_cache.set(key, value, ttl=min(ttl or 3600, 60))
        
        # Require majority for success
        return success_count > len(nodes) // 2
    
    def _get_from_node(self, node: str, key: str) -> Optional[Any]:
        """Get value from specific node (to be implemented based on protocol)."""
        # This would be implemented with Redis, Memcached, or custom protocol
        # For now, return None as placeholder
        return None
    
    def _set_to_node(self, node: str, key: str, value: Any, ttl: Optional[float]) -> bool:
        """Set value to specific node (to be implemented based on protocol)."""
        # This would be implemented with Redis, Memcached, or custom protocol
        # For now, return True as placeholder
        return True

class CacheManager:
    """High-level cache management with multiple backends."""
    
    def __init__(self):
        self.backends: Dict[str, CacheBackend] = {}
        self.default_backend = "memory"
        self.cache_tags: Dict[str, List[str]] = {}
        self.tag_keys: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def add_backend(self, name: str, backend: CacheBackend, is_default: bool = False):
        """Add cache backend."""
        self.backends[name] = backend
        if is_default:
            self.default_backend = name
    
    def get_backend(self, name: Optional[str] = None) -> CacheBackend:
        """Get cache backend."""
        backend_name = name or self.default_backend
        if backend_name not in self.backends:
            raise ValueError(f"Backend '{backend_name}' not found")
        return self.backends[backend_name]
    
    def get(
        self,
        key: str,
        backend: Optional[str] = None,
        default: Any = None
    ) -> Any:
        """Get value from cache."""
        try:
            value = self.get_backend(backend).get(key)
            return value if value is not None else default
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        backend: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache with optional tags."""
        try:
            success = self.get_backend(backend).set(key, value, ttl)
            
            if success and tags:
                with self._lock:
                    self.cache_tags[key] = tags
                    for tag in tags:
                        if key not in self.tag_keys[tag]:
                            self.tag_keys[tag].append(key)
            
            return success
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str, backend: Optional[str] = None) -> bool:
        """Delete value from cache."""
        try:
            success = self.get_backend(backend).delete(key)
            
            if success:
                with self._lock:
                    tags = self.cache_tags.pop(key, [])
                    for tag in tags:
                        if key in self.tag_keys[tag]:
                            self.tag_keys[tag].remove(key)
            
            return success
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def delete_by_tag(self, tag: str, backend: Optional[str] = None) -> int:
        """Delete all entries with specific tag."""
        deleted_count = 0
        
        with self._lock:
            keys_to_delete = self.tag_keys.get(tag, []).copy()
        
        for key in keys_to_delete:
            if self.delete(key, backend):
                deleted_count += 1
        
        return deleted_count
    
    def clear(self, backend: Optional[str] = None) -> bool:
        """Clear all entries."""
        try:
            success = self.get_backend(backend).clear()
            if success:
                with self._lock:
                    self.cache_tags.clear()
                    self.tag_keys.clear()
            return success
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

def cache_result(
    ttl: Optional[float] = None,
    backend: Optional[str] = None,
    tags: Optional[List[str]] = None,
    cache_manager: Optional[CacheManager] = None
):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            cache_key = hashlib.md5(
                json.dumps(key_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Get cache manager
            mgr = cache_manager or _global_cache_manager
            
            # Try to get cached result
            result = mgr.get(cache_key, backend)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            mgr.set(cache_key, result, ttl, backend, tags)
            
            return result
        return wrapper
    return decorator

# Global cache manager
_global_cache_manager = CacheManager()
_global_cache_manager.add_backend(
    "memory",
    MemoryCache(max_size=1000, policy=CachePolicy.ADAPTIVE),
    is_default=True
)

def get_cache_manager() -> CacheManager:
    """Get global cache manager."""
    return _global_cache_manager