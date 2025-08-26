"""
QNP Production Framework
Enterprise-grade production system for Quantum-Neuromorphic-Photonic fusion.

Features:
- Comprehensive error handling and recovery
- Advanced monitoring and observability
- Security hardening for production deployment
- Performance optimization and caching
- Health checks and circuit breakers
- Compliance and audit logging
"""

import asyncio
import logging
import json
import time
import hashlib
import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import secrets
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum

# Import QNP components
from quantum_neuromorphic_photonic_fusion import QNPFusionEngine, create_qnp_fusion_engine

class HealthStatus(Enum):
    """System health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

class SecurityLevel(Enum):
    """Security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

@dataclass
class QNPMetrics:
    """Comprehensive QNP system metrics"""
    timestamp: datetime
    processing_time_ms: float
    quantum_coherence: float
    neuromorphic_activity: float
    photonic_efficiency: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_rate: float
    throughput_rps: float
    cache_hit_rate: float

@dataclass
class SecurityContext:
    """Security context for QNP operations"""
    user_id: str
    session_id: str
    classification: SecurityLevel
    access_token: Optional[str] = None
    client_ip: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_authorized(self, required_level: SecurityLevel) -> bool:
        """Check if context has required authorization level"""
        levels = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.RESTRICTED: 3
        }
        return levels.get(self.classification, 0) >= levels.get(required_level, 0)

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """Circuit breaker for QNP service protection"""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def record_success(self):
        """Record successful operation"""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
    
    def record_failure(self):
        """Record failed operation"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (time.time() - (self.last_failure_time or 0)) >= self.timeout_seconds

class QNPCache:
    """High-performance cache for QNP results"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, text: str, config: Optional[Dict] = None) -> str:
        """Generate cache key for text and configuration"""
        content = f"{text}:{json.dumps(config or {}, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, text: str, config: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached result"""
        key = self._generate_key(text, config)
        
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            # Check TTL
            cache_time = self._cache[key].get("_cached_at", 0)
            if time.time() - cache_time > self.ttl_seconds:
                del self._cache[key]
                del self._access_times[key]
                self.misses += 1
                return None
            
            # Update access time
            self._access_times[key] = time.time()
            self.hits += 1
            
            result = self._cache[key].copy()
            del result["_cached_at"]
            return result
    
    def put(self, text: str, result: Dict[str, Any], config: Optional[Dict] = None):
        """Store result in cache"""
        key = self._generate_key(text, config)
        
        with self._lock:
            # Evict if cache is full
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Store with timestamp
            cached_result = result.copy()
            cached_result["_cached_at"] = time.time()
            
            self._cache[key] = cached_result
            self._access_times[key] = time.time()
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds
        }
    
    def clear(self):
        """Clear all cached entries"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self.hits = 0
            self.misses = 0

class HealthChecker:
    """Comprehensive health monitoring for QNP system"""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], bool]] = {}
        self.health_history: deque = deque(maxlen=100)
        self.last_check_time = 0
        self.check_interval = 30  # seconds
    
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function"""
        self.checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        current_time = time.time()
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                is_healthy = check_func()
                results[name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "timestamp": current_time
                }
                if not is_healthy:
                    overall_healthy = False
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": current_time
                }
                overall_healthy = False
        
        health_status = HealthStatus.HEALTHY if overall_healthy else HealthStatus.UNHEALTHY
        
        health_report = {
            "overall_status": health_status.value,
            "checks": results,
            "timestamp": current_time
        }
        
        self.health_history.append(health_report)
        self.last_check_time = current_time
        
        return health_report
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary over time"""
        if not self.health_history:
            return {"status": "no_data"}
        
        recent_checks = list(self.health_history)[-10:]  # Last 10 checks
        healthy_count = sum(1 for check in recent_checks if check["overall_status"] == "healthy")
        
        return {
            "current_status": recent_checks[-1]["overall_status"],
            "recent_health_rate": healthy_count / len(recent_checks),
            "total_checks": len(self.health_history),
            "last_check_time": self.last_check_time
        }

class QNPAuditLogger:
    """Comprehensive audit logging for compliance"""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or Path("qnp_audit.jsonl")
        self.logger = logging.getLogger("qnp_audit")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_request(self, security_context: SecurityContext, text: str, 
                   result: Dict[str, Any], processing_time_ms: float):
        """Log sentiment analysis request"""
        audit_record = {
            "event_type": "sentiment_analysis",
            "timestamp": datetime.now().isoformat(),
            "user_id": security_context.user_id,
            "session_id": security_context.session_id,
            "classification": security_context.classification.value,
            "client_ip": security_context.client_ip,
            "text_length": len(text),
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "sentiment": max(["positive", "negative", "neutral"], 
                           key=lambda k: result.get(k, 0)),
            "confidence": result.get("confidence", 0),
            "processing_time_ms": processing_time_ms,
            "qnp_fusion_score": result.get("qnp_fusion_score", 0)
        }
        
        self.logger.info(json.dumps(audit_record))
    
    def log_error(self, security_context: SecurityContext, error: Exception, text: str):
        """Log error event"""
        error_record = {
            "event_type": "error",
            "timestamp": datetime.now().isoformat(),
            "user_id": security_context.user_id,
            "session_id": security_context.session_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "text_length": len(text),
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16]
        }
        
        self.logger.error(json.dumps(error_record))
    
    def log_security_event(self, event_type: str, security_context: SecurityContext, 
                          details: Dict[str, Any]):
        """Log security-related event"""
        security_record = {
            "event_type": f"security_{event_type}",
            "timestamp": datetime.now().isoformat(),
            "user_id": security_context.user_id,
            "session_id": security_context.session_id,
            "classification": security_context.classification.value,
            "client_ip": security_context.client_ip,
            **details
        }
        
        self.logger.warning(json.dumps(security_record))

class QNPProductionEngine:
    """Production-hardened QNP Fusion Engine"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize core QNP engine
        self.qnp_engine = create_qnp_fusion_engine(self.config.get("qnp_config"))
        
        # Initialize production components
        self.cache = QNPCache(
            max_size=self.config.get("cache_max_size", 10000),
            ttl_seconds=self.config.get("cache_ttl", 3600)
        )
        
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.get("circuit_breaker_threshold", 5),
            timeout_seconds=self.config.get("circuit_breaker_timeout", 60)
        )
        
        self.health_checker = HealthChecker()
        self.audit_logger = QNPAuditLogger(self.config.get("audit_log_file"))
        
        # Metrics tracking
        self.metrics_history: deque = deque(maxlen=1000)
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get("max_workers", 8)
        )
        
        # Initialize health checks
        self._setup_health_checks()
        
        # Rate limiting
        self.rate_limiter = defaultdict(lambda: {"count": 0, "window_start": time.time()})
        self.rate_limit_per_minute = self.config.get("rate_limit_per_minute", 1000)
    
    def _setup_health_checks(self):
        """Setup default health checks"""
        def check_qnp_engine():
            try:
                test_result = self.qnp_engine.fallback_analysis("health check")
                return "positive" in test_result
            except Exception:
                return False
        
        def check_cache():
            try:
                self.cache.get_stats()
                return True
            except Exception:
                return False
        
        def check_circuit_breaker():
            return self.circuit_breaker.state != CircuitBreakerState.OPEN
        
        def check_memory_usage():
            import psutil
            return psutil.virtual_memory().percent < 85
        
        self.health_checker.register_check("qnp_engine", check_qnp_engine)
        self.health_checker.register_check("cache", check_cache)
        self.health_checker.register_check("circuit_breaker", check_circuit_breaker)
        self.health_checker.register_check("memory", check_memory_usage)
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits"""
        current_time = time.time()
        user_stats = self.rate_limiter[user_id]
        
        # Reset window if needed
        if current_time - user_stats["window_start"] >= 60:  # 1 minute window
            user_stats["count"] = 0
            user_stats["window_start"] = current_time
        
        # Check limit
        if user_stats["count"] >= self.rate_limit_per_minute:
            return False
        
        user_stats["count"] += 1
        return True
    
    def _record_metrics(self, processing_time_ms: float, error: Optional[Exception] = None):
        """Record performance metrics"""
        try:
            import psutil
            memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
            cpu_usage = psutil.cpu_percent()
        except ImportError:
            memory_usage = 0
            cpu_usage = 0
        
        uptime_seconds = time.time() - self.start_time
        throughput = self.request_count / uptime_seconds if uptime_seconds > 0 else 0
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        
        cache_stats = self.cache.get_stats()
        
        metrics = QNPMetrics(
            timestamp=datetime.now(),
            processing_time_ms=processing_time_ms,
            quantum_coherence=0.8,  # Placeholder
            neuromorphic_activity=0.7,  # Placeholder
            photonic_efficiency=0.9,  # Placeholder
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            error_rate=error_rate,
            throughput_rps=throughput,
            cache_hit_rate=cache_stats["hit_rate"]
        )
        
        self.metrics_history.append(metrics)
    
    async def analyze_sentiment_secure(self, text: str, 
                                     security_context: SecurityContext) -> Dict[str, Any]:
        """Secure sentiment analysis with full production features"""
        start_time = time.time()
        
        try:
            # Security checks
            if not security_context.is_authorized(SecurityLevel.PUBLIC):
                raise PermissionError("Insufficient authorization")
            
            # Rate limiting
            if not self._check_rate_limit(security_context.user_id):
                raise Exception("Rate limit exceeded")
            
            # Circuit breaker check
            if not self.circuit_breaker.can_execute():
                raise Exception("Service temporarily unavailable")
            
            # Input validation
            if not text or not isinstance(text, str):
                raise ValueError("Invalid text input")
            
            if len(text) > self.config.get("max_text_length", 10000):
                raise ValueError("Text too long")
            
            # Check cache first
            cached_result = self.cache.get(text, self.config.get("qnp_config"))
            if cached_result:
                processing_time_ms = (time.time() - start_time) * 1000
                self.audit_logger.log_request(security_context, text, cached_result, processing_time_ms)
                self._record_metrics(processing_time_ms)
                self.request_count += 1
                return {**cached_result, "cached": True, "processing_time_ms": processing_time_ms}
            
            # Perform QNP analysis
            result = await self.qnp_engine.analyze_sentiment_async(text)
            
            processing_time_ms = (time.time() - start_time) * 1000
            result["processing_time_ms"] = processing_time_ms
            
            # Cache result
            self.cache.put(text, result, self.config.get("qnp_config"))
            
            # Record success
            self.circuit_breaker.record_success()
            self.request_count += 1
            
            # Audit logging
            self.audit_logger.log_request(security_context, text, result, processing_time_ms)
            
            # Record metrics
            self._record_metrics(processing_time_ms)
            
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Record failure
            self.circuit_breaker.record_failure()
            self.error_count += 1
            
            # Audit error
            self.audit_logger.log_error(security_context, e, text)
            
            # Record metrics
            self._record_metrics(processing_time_ms, e)
            
            # Return error response
            return {
                "error": True,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "processing_time_ms": processing_time_ms,
                "fallback_available": True
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        health_report = self.health_checker.run_health_checks()
        health_summary = self.health_checker.get_health_summary()
        
        return {
            "health_report": health_report,
            "health_summary": health_summary,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "cache_stats": self.cache.get_stats(),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
            "uptime_seconds": time.time() - self.start_time
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 metrics
        
        def avg(field):
            return sum(getattr(m, field) for m in recent_metrics) / len(recent_metrics)
        
        return {
            "avg_processing_time_ms": avg("processing_time_ms"),
            "avg_memory_usage_mb": avg("memory_usage_mb"),
            "avg_cpu_usage_percent": avg("cpu_usage_percent"),
            "avg_error_rate": avg("error_rate"),
            "avg_throughput_rps": avg("throughput_rps"),
            "avg_cache_hit_rate": avg("cache_hit_rate"),
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "metrics_count": len(self.metrics_history)
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        logging.info("Shutting down QNP Production Engine...")
        
        # Stop health checks
        # Clear cache
        self.cache.clear()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logging.info("QNP Production Engine shutdown complete")

# Factory function
def create_qnp_production_engine(config: Optional[Dict] = None) -> QNPProductionEngine:
    """Create production-ready QNP engine"""
    return QNPProductionEngine(config)

# Context manager for secure operations
@asynccontextmanager
async def qnp_secure_context(user_id: str, classification: SecurityLevel = SecurityLevel.PUBLIC):
    """Context manager for secure QNP operations"""
    security_context = SecurityContext(
        user_id=user_id,
        session_id=secrets.token_hex(16),
        classification=classification,
        client_ip="127.0.0.1"
    )
    
    try:
        yield security_context
    finally:
        # Cleanup if needed
        pass

if __name__ == "__main__":
    # Demo production system
    async def main():
        print("🛡️  QNP Production Framework Demo")
        
        engine = create_qnp_production_engine({
            "cache_max_size": 1000,
            "rate_limit_per_minute": 100
        })
        
        async with qnp_secure_context("demo_user", SecurityLevel.INTERNAL) as context:
            result = await engine.analyze_sentiment_secure(
                "This production system is amazing!", 
                context
            )
            print("Result:", json.dumps(result, indent=2))
        
        health_status = engine.get_health_status()
        print("Health:", json.dumps(health_status, indent=2))
        
        await engine.shutdown()
    
    asyncio.run(main())