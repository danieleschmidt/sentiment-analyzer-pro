
import time
import threading
import hashlib
import json
import pickle
from typing import Any, Dict, Optional, Callable, Union
from functools import wraps
from pathlib import Path
import weakref

class CacheStats:
    """Cache statistics tracking."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size = 0
        self._lock = threading.Lock()
    
    def record_hit(self):
        with self._lock:
            self.hits += 1
    
    def record_miss(self):
        with self._lock:
            self.misses += 1
    
    def record_eviction(self):
        with self._lock:
            self.evictions += 1
    
    def get_hit_rate(self) -> float:
        with self._lock:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0

class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[float] = None):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.data = {}
        self.access_order = []
        self.timestamps = {}
        self.stats = CacheStats()
        self._lock = threading.RLock()
    
    def _is_expired(self, key: str) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.timestamps.get(key, 0) > self.ttl
    
    def _evict_expired(self):
        if self.ttl is None:
            return
        
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            self._remove(key)
    
    def _remove(self, key: str):
        if key in self.data:
            del self.data[key]
            if key in self.access_order:
                self.access_order.remove(key)
            if key in self.timestamps:
                del self.timestamps[key]
            self.stats.record_eviction()
    
    def _update_access(self, key: str):
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        self.timestamps[key] = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            self._evict_expired()
            
            if key not in self.data or self._is_expired(key):
                self.stats.record_miss()
                return None
            
            self._update_access(key)
            self.stats.record_hit()
            return self.data[key]
    
    def put(self, key: str, value: Any):
        with self._lock:
            self._evict_expired()
            
            # Remove if already exists
            if key in self.data:
                self._remove(key)
            
            # Evict oldest if at capacity
            while len(self.data) >= self.max_size:
                oldest_key = self.access_order[0]
                self._remove(oldest_key)
            
            # Add new item
            self.data[key] = value
            self._update_access(key)
            self.stats.size = len(self.data)
    
    def clear(self):
        with self._lock:
            self.data.clear()
            self.access_order.clear()
            self.timestamps.clear()
            self.stats.size = 0

class DistributedCache:
    """Distributed cache with persistence."""
    
    def __init__(self, cache_dir: str = "cache", max_memory_mb: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.memory_cache = LRUCache(max_size=1000, ttl_seconds=3600)  # 1 hour TTL
        self.stats = CacheStats()
        self._lock = threading.RLock()
    
    def _get_key_hash(self, key: str) -> str:
        """Get consistent hash for key."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def _get_disk_path(self, key: str) -> Path:
        """Get disk path for key."""
        key_hash = self._get_key_hash(key)
        return self.cache_dir / f"{key_hash}.cache"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Try JSON first (faster, human readable)
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                return json.dumps(value).encode('utf-8')
        except:
            pass
        
        # Fall back to pickle
        return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except:
            # Fall back to pickle
            return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            # Check memory cache first
            value = self.memory_cache.get(key)
            if value is not None:
                self.stats.record_hit()
                return value
            
            # Check disk cache
            disk_path = self._get_disk_path(key)
            if disk_path.exists():
                try:
                    with open(disk_path, 'rb') as f:
                        data = f.read()
                    value = self._deserialize_value(data)
                    
                    # Put back in memory cache
                    self.memory_cache.put(key, value)
                    self.stats.record_hit()
                    return value
                except Exception:
                    # Remove corrupted cache file
                    disk_path.unlink(missing_ok=True)
            
            self.stats.record_miss()
            return None
    
    def put(self, key: str, value: Any, persist: bool = True):
        with self._lock:
            # Store in memory cache
            self.memory_cache.put(key, value)
            
            # Optionally persist to disk
            if persist:
                try:
                    disk_path = self._get_disk_path(key)
                    data = self._serialize_value(value)
                    
                    with open(disk_path, 'wb') as f:
                        f.write(data)
                except Exception as e:
                    # Log error but don't fail
                    pass

def cached(cache_key_func: Optional[Callable] = None, ttl_seconds: Optional[float] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        # Create cache instance for this function
        cache = LRUCache(max_size=500, ttl_seconds=ttl_seconds)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    'func': func.__name__,
                    'args': str(args),
                    'kwargs': str(sorted(kwargs.items()))
                }
                cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Try cache first
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            return result
        
        # Expose cache for inspection
        wrapper._cache = cache
        return wrapper
    
    return decorator

# Global caches
global_memory_cache = LRUCache(max_size=2000, ttl_seconds=1800)  # 30 minutes
global_distributed_cache = DistributedCache()
