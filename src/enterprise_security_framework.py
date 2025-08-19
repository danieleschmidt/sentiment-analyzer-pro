"""
Enterprise Security Framework for AGI Engine
Comprehensive security, audit, and compliance system.
"""

from __future__ import annotations

import time
import hashlib
import hmac
import secrets
import jwt
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set, Union
from enum import Enum
from collections import defaultdict, deque
import logging
import asyncio
from datetime import datetime, timedelta
import json
import re
import ipaddress
from functools import wraps, lru_cache
import os

from .logging_config import get_logger
from .metrics import metrics

logger = get_logger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """Threat assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AccessPattern(Enum):
    """Access pattern types for behavioral analysis."""
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    ANOMALOUS = "anomalous"
    MALICIOUS = "malicious"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str = field(default_factory=lambda: secrets.token_hex(16))
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""
    severity: ThreatLevel = ThreatLevel.LOW
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class AccessAttempt:
    """Access attempt tracking."""
    timestamp: float = field(default_factory=time.time)
    source_ip: str = ""
    user_agent: str = ""
    endpoint: str = ""
    success: bool = False
    failure_reason: Optional[str] = None
    user_id: Optional[str] = None


class SecurityAuditLogger:
    """Comprehensive security audit logging system."""
    
    def __init__(self, retention_days: int = 90):
        self.retention_days = retention_days
        self.events: deque = deque(maxlen=10000)
        self.event_counts = defaultdict(int)
        self.threat_timeline = deque(maxlen=1000)
        self._lock = threading.RLock()
    
    def log_event(self, 
                  event_type: str,
                  severity: ThreatLevel = ThreatLevel.LOW,
                  source_ip: Optional[str] = None,
                  user_id: Optional[str] = None,
                  description: str = "",
                  metadata: Optional[Dict[str, Any]] = None) -> SecurityEvent:
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.events.append(event)
            self.event_counts[event_type] += 1
            
            if severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self.threat_timeline.append(event)
        
        # Log to application logger
        logger.warning(f"Security event: {event_type}", extra={
            "event_id": event.event_id,
            "severity": severity.value,
            "source_ip": source_ip,
            "user_id": user_id,
            "description": description
        })
        
        return event
    
    def get_events(self, 
                  since: Optional[float] = None,
                  event_type: Optional[str] = None,
                  severity: Optional[ThreatLevel] = None) -> List[SecurityEvent]:
        """Retrieve security events with filtering."""
        with self._lock:
            events = list(self.events)
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        return events
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get threat landscape summary."""
        now = time.time()
        last_24h = now - 86400
        last_hour = now - 3600
        
        with self._lock:
            recent_events = [e for e in self.events if e.timestamp >= last_24h]
            hourly_events = [e for e in self.events if e.timestamp >= last_hour]
            
            threat_levels = defaultdict(int)
            for event in recent_events:
                threat_levels[event.severity.value] += 1
        
        return {
            "total_events_24h": len(recent_events),
            "events_last_hour": len(hourly_events),
            "threat_distribution": dict(threat_levels),
            "top_event_types": dict(sorted(self.event_counts.items(), 
                                         key=lambda x: x[1], reverse=True)[:10]),
            "critical_threats_active": len([e for e in self.threat_timeline 
                                          if e.severity == ThreatLevel.CRITICAL and not e.resolved])
        }


class BehavioralAnalyzer:
    """Advanced behavioral analysis and anomaly detection."""
    
    def __init__(self):
        self.access_patterns: Dict[str, List[AccessAttempt]] = defaultdict(list)
        self.baseline_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.anomaly_thresholds = {
            "request_rate": 100.0,  # requests per minute
            "failure_rate": 0.3,    # 30% failure rate
            "geographic_variance": 1000.0,  # km distance
            "time_variance": 4.0    # hours outside normal pattern
        }
        self._lock = threading.RLock()
    
    def analyze_access(self, access: AccessAttempt) -> AccessPattern:
        """Analyze access attempt for anomalies."""
        user_key = access.user_id or access.source_ip
        
        with self._lock:
            # Store access attempt
            self.access_patterns[user_key].append(access)
            
            # Keep only recent access attempts (last 24 hours)
            cutoff = time.time() - 86400
            self.access_patterns[user_key] = [
                a for a in self.access_patterns[user_key] 
                if a.timestamp >= cutoff
            ]
            
            user_attempts = self.access_patterns[user_key]
        
        # Analyze patterns
        if len(user_attempts) < 5:
            return AccessPattern.NORMAL  # Not enough data
        
        # Check request rate
        recent_attempts = [a for a in user_attempts if a.timestamp >= time.time() - 60]
        if len(recent_attempts) > self.anomaly_thresholds["request_rate"] / 60:
            return AccessPattern.MALICIOUS
        
        # Check failure rate
        failed_attempts = [a for a in user_attempts if not a.success]
        failure_rate = len(failed_attempts) / len(user_attempts)
        if failure_rate > self.anomaly_thresholds["failure_rate"]:
            return AccessPattern.SUSPICIOUS
        
        # Check time patterns
        access_hours = [datetime.fromtimestamp(a.timestamp).hour for a in user_attempts]
        hour_variance = max(access_hours) - min(access_hours)
        if hour_variance > self.anomaly_thresholds["time_variance"]:
            return AccessPattern.ANOMALOUS
        
        # Check endpoint diversity (potential reconnaissance)
        unique_endpoints = set(a.endpoint for a in user_attempts)
        if len(unique_endpoints) > 20:  # Accessing many different endpoints
            return AccessPattern.SUSPICIOUS
        
        return AccessPattern.NORMAL
    
    def update_baseline(self, user_id: str, metrics: Dict[str, float]) -> None:
        """Update baseline metrics for a user."""
        with self._lock:
            self.baseline_metrics[user_id].update(metrics)
    
    def detect_anomalies(self, user_id: str, current_metrics: Dict[str, float]) -> List[str]:
        """Detect anomalies compared to baseline."""
        anomalies = []
        baseline = self.baseline_metrics.get(user_id, {})
        
        for metric, value in current_metrics.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                deviation = abs(value - baseline_value) / (baseline_value + 1e-6)
                
                if deviation > 2.0:  # More than 200% deviation
                    anomalies.append(f"{metric}_anomaly")
        
        return anomalies


class AdvancedRateLimiter:
    """Advanced rate limiting with adaptive thresholds."""
    
    def __init__(self):
        self.buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "tokens": 100,
            "last_refill": time.time(),
            "burst_capacity": 100,
            "refill_rate": 10,  # tokens per second
            "violations": 0,
            "blocked_until": 0
        })
        self.global_limits = {
            "requests_per_second": 1000,
            "concurrent_users": 500,
            "bandwidth_mbps": 100
        }
        self._lock = threading.RLock()
    
    def check_limit(self, key: str, tokens_required: int = 1) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits."""
        now = time.time()
        
        with self._lock:
            bucket = self.buckets[key]
            
            # Check if currently blocked
            if bucket["blocked_until"] > now:
                return False, {
                    "reason": "temporarily_blocked",
                    "blocked_until": bucket["blocked_until"],
                    "violations": bucket["violations"]
                }
            
            # Refill bucket based on elapsed time
            elapsed = now - bucket["last_refill"]
            tokens_to_add = elapsed * bucket["refill_rate"]
            bucket["tokens"] = min(bucket["burst_capacity"], 
                                 bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = now
            
            # Check if enough tokens available
            if bucket["tokens"] >= tokens_required:
                bucket["tokens"] -= tokens_required
                return True, {"tokens_remaining": bucket["tokens"]}
            else:
                # Rate limit exceeded
                bucket["violations"] += 1
                
                # Exponential backoff for repeated violations
                if bucket["violations"] > 5:
                    block_duration = min(300, 2 ** (bucket["violations"] - 5))  # Max 5 minutes
                    bucket["blocked_until"] = now + block_duration
                
                return False, {
                    "reason": "rate_limit_exceeded",
                    "tokens_available": bucket["tokens"],
                    "tokens_required": tokens_required,
                    "violations": bucket["violations"]
                }
    
    def adaptive_adjust(self, key: str, success_rate: float, response_time: float) -> None:
        """Adaptively adjust rate limits based on performance."""
        with self._lock:
            bucket = self.buckets[key]
            
            # Adjust based on success rate and response time
            if success_rate > 0.95 and response_time < 0.5:
                # Performance is good, can increase limits
                bucket["refill_rate"] = min(50, bucket["refill_rate"] * 1.1)
                bucket["burst_capacity"] = min(500, bucket["burst_capacity"] * 1.05)
            elif success_rate < 0.8 or response_time > 2.0:
                # Performance is poor, decrease limits
                bucket["refill_rate"] = max(1, bucket["refill_rate"] * 0.9)
                bucket["burst_capacity"] = max(10, bucket["burst_capacity"] * 0.95)


class EncryptionManager:
    """Advanced encryption and key management."""
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or self._generate_master_key()
        self.derived_keys: Dict[str, bytes] = {}
        self.key_rotation_interval = 86400  # 24 hours
        self.last_rotation = time.time()
        self._lock = threading.RLock()
    
    def _generate_master_key(self) -> str:
        """Generate a new master key."""
        return secrets.token_hex(32)
    
    def derive_key(self, purpose: str, salt: Optional[bytes] = None) -> bytes:
        """Derive a key for specific purpose."""
        if salt is None:
            salt = purpose.encode('utf-8')
        
        with self._lock:
            if purpose not in self.derived_keys or self._should_rotate_key():
                # Use PBKDF2 for key derivation
                import hashlib
                derived = hashlib.pbkdf2_hmac(
                    'sha256',
                    self.master_key.encode('utf-8'),
                    salt,
                    100000,  # iterations
                    32  # key length
                )
                self.derived_keys[purpose] = derived
            
            return self.derived_keys[purpose]
    
    def _should_rotate_key(self) -> bool:
        """Check if keys should be rotated."""
        return time.time() - self.last_rotation > self.key_rotation_interval
    
    def encrypt_sensitive_data(self, data: str, purpose: str = "general") -> Dict[str, str]:
        """Encrypt sensitive data with authenticated encryption."""
        from cryptography.fernet import Fernet
        import base64
        
        key = self.derive_key(purpose)
        fernet_key = base64.urlsafe_b64encode(key)
        cipher = Fernet(fernet_key)
        
        encrypted = cipher.encrypt(data.encode('utf-8'))
        
        return {
            "encrypted_data": base64.b64encode(encrypted).decode('utf-8'),
            "purpose": purpose,
            "timestamp": str(time.time())
        }
    
    def decrypt_sensitive_data(self, encrypted_data: Dict[str, str]) -> str:
        """Decrypt sensitive data."""
        from cryptography.fernet import Fernet
        import base64
        
        key = self.derive_key(encrypted_data["purpose"])
        fernet_key = base64.urlsafe_b64encode(key)
        cipher = Fernet(fernet_key)
        
        encrypted_bytes = base64.b64decode(encrypted_data["encrypted_data"])
        decrypted = cipher.decrypt(encrypted_bytes)
        
        return decrypted.decode('utf-8')


class ComplianceManager:
    """Comprehensive compliance management for GDPR, CCPA, etc."""
    
    def __init__(self):
        self.data_processing_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.data_retention_policies = {
            "personal_data": 730,  # 2 years
            "analytics_data": 365,  # 1 year
            "audit_logs": 2555,    # 7 years
            "security_events": 1095  # 3 years
        }
        self._lock = threading.RLock()
    
    def record_data_processing(self, 
                             user_id: str,
                             data_type: str,
                             purpose: str,
                             legal_basis: str,
                             retention_period: Optional[int] = None) -> str:
        """Record data processing activity."""
        record_id = secrets.token_hex(16)
        
        record = {
            "record_id": record_id,
            "timestamp": time.time(),
            "data_type": data_type,
            "purpose": purpose,
            "legal_basis": legal_basis,
            "retention_period": retention_period or self.data_retention_policies.get(data_type, 365),
            "status": "active"
        }
        
        with self._lock:
            self.data_processing_records[user_id].append(record)
        
        logger.info("Data processing recorded", extra={
            "user_id": user_id,
            "record_id": record_id,
            "data_type": data_type,
            "purpose": purpose
        })
        
        return record_id
    
    def record_consent(self, 
                      user_id: str,
                      purpose: str,
                      granted: bool,
                      consent_method: str = "explicit") -> Dict[str, Any]:
        """Record user consent."""
        consent_record = {
            "user_id": user_id,
            "purpose": purpose,
            "granted": granted,
            "consent_method": consent_method,
            "timestamp": time.time(),
            "ip_address": None,  # Should be provided by caller
            "user_agent": None   # Should be provided by caller
        }
        
        with self._lock:
            consent_key = f"{user_id}_{purpose}"
            self.consent_records[consent_key] = consent_record
        
        return consent_record
    
    def check_data_retention(self) -> List[Dict[str, Any]]:
        """Check for data that should be deleted due to retention policies."""
        now = time.time()
        expired_data = []
        
        with self._lock:
            for user_id, records in self.data_processing_records.items():
                for record in records:
                    if record["status"] == "active":
                        age_days = (now - record["timestamp"]) / 86400
                        if age_days > record["retention_period"]:
                            expired_data.append({
                                "user_id": user_id,
                                "record_id": record["record_id"],
                                "data_type": record["data_type"],
                                "age_days": age_days,
                                "retention_period": record["retention_period"]
                            })
        
        return expired_data
    
    def generate_data_export(self, user_id: str) -> Dict[str, Any]:
        """Generate data export for GDPR compliance."""
        with self._lock:
            user_data = {
                "user_id": user_id,
                "export_timestamp": time.time(),
                "data_processing_records": self.data_processing_records.get(user_id, []),
                "consent_records": [
                    record for key, record in self.consent_records.items()
                    if record["user_id"] == user_id
                ]
            }
        
        return user_data


class SecurityOrchestrator:
    """Main security orchestration system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.audit_logger = SecurityAuditLogger()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.rate_limiter = AdvancedRateLimiter()
        self.encryption_manager = EncryptionManager()
        self.compliance_manager = ComplianceManager()
        
        # Security policies
        self.security_policies = {
            "max_login_attempts": 5,
            "session_timeout": 3600,  # 1 hour
            "password_min_length": 12,
            "require_mfa": False,
            "ip_whitelist": [],
            "blocked_countries": [],
            "suspicious_pattern_threshold": 3
        }
        
        # Active security measures
        self.active_threats: Dict[str, SecurityEvent] = {}
        self.blocked_ips: Set[str] = set()
        self.quarantined_users: Set[str] = set()
        
        logger.info("SecurityOrchestrator initialized", extra={
            "policies_loaded": len(self.security_policies),
            "encryption_enabled": True,
            "compliance_frameworks": ["GDPR", "CCPA", "SOX"],
            "behavioral_analysis": True
        })
    
    def validate_request(self, 
                        request_data: Dict[str, Any],
                        user_id: Optional[str] = None,
                        source_ip: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive request validation."""
        validation_result = {
            "allowed": True,
            "security_level": SecurityLevel.PUBLIC,
            "restrictions": [],
            "warnings": []
        }
        
        # Check IP blocking
        if source_ip and source_ip in self.blocked_ips:
            validation_result["allowed"] = False
            validation_result["restrictions"].append("ip_blocked")
            
            self.audit_logger.log_event(
                "blocked_ip_access",
                ThreatLevel.HIGH,
                source_ip=source_ip,
                description="Access attempt from blocked IP"
            )
            return validation_result
        
        # Check user quarantine
        if user_id and user_id in self.quarantined_users:
            validation_result["allowed"] = False
            validation_result["restrictions"].append("user_quarantined")
            return validation_result
        
        # Rate limiting
        rate_limit_key = user_id or source_ip or "anonymous"
        rate_allowed, rate_info = self.rate_limiter.check_limit(rate_limit_key)
        
        if not rate_allowed:
            validation_result["allowed"] = False
            validation_result["restrictions"].append("rate_limited")
            validation_result["rate_limit_info"] = rate_info
            
            self.audit_logger.log_event(
                "rate_limit_violation",
                ThreatLevel.MEDIUM,
                source_ip=source_ip,
                user_id=user_id,
                metadata=rate_info
            )
        
        # Behavioral analysis
        if source_ip:
            access_attempt = AccessAttempt(
                source_ip=source_ip,
                user_agent=request_data.get("user_agent", ""),
                endpoint=request_data.get("endpoint", ""),
                success=True,  # Will be updated later
                user_id=user_id
            )
            
            pattern = self.behavioral_analyzer.analyze_access(access_attempt)
            
            if pattern in [AccessPattern.SUSPICIOUS, AccessPattern.MALICIOUS]:
                validation_result["warnings"].append(f"suspicious_pattern_{pattern.value}")
                
                self.audit_logger.log_event(
                    "suspicious_behavior",
                    ThreatLevel.HIGH if pattern == AccessPattern.MALICIOUS else ThreatLevel.MEDIUM,
                    source_ip=source_ip,
                    user_id=user_id,
                    description=f"Detected {pattern.value} access pattern"
                )
        
        # Input validation
        input_validation = self._validate_input_security(request_data)
        if not input_validation["safe"]:
            validation_result["warnings"].extend(input_validation["threats"])
        
        return validation_result
    
    def _validate_input_security(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input for security threats."""
        threats = []
        safe = True
        
        # SQL injection patterns
        sql_patterns = [
            r"(\bUNION\b.*\bSELECT\b)|(\bSELECT\b.*\bFROM\b)",
            r"(\bINSERT\b.*\bINTO\b)|(\bUPDATE\b.*\bSET\b)",
            r"(\bDELETE\b.*\bFROM\b)|(\bDROP\b.*\bTABLE\b)",
            r"(\bEXEC\b)|(\bEXECUTE\b)",
            r"(\'.*\bOR\b.*\'=\')",
        ]
        
        # XSS patterns
        xss_patterns = [
            r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe\b",
            r"<object\b"
        ]
        
        # Command injection patterns
        cmd_patterns = [
            r"[;&|`]",
            r"\$\(",
            r"\\x[0-9a-fA-F]{2}",
            r"eval\s*\(",
            r"exec\s*\("
        ]
        
        def check_patterns(text: str, patterns: List[str], threat_type: str) -> None:
            nonlocal safe, threats
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    threats.append(f"{threat_type}_detected")
                    safe = False
                    break
        
        # Check all string values in the data
        for key, value in data.items():
            if isinstance(value, str):
                check_patterns(value, sql_patterns, "sql_injection")
                check_patterns(value, xss_patterns, "xss")
                check_patterns(value, cmd_patterns, "command_injection")
        
        return {"safe": safe, "threats": threats}
    
    def handle_security_incident(self, 
                                event_type: str,
                                severity: ThreatLevel,
                                source_ip: Optional[str] = None,
                                user_id: Optional[str] = None,
                                auto_respond: bool = True) -> SecurityEvent:
        """Handle security incidents with automatic response."""
        # Log the incident
        event = self.audit_logger.log_event(
            event_type, severity, source_ip, user_id,
            f"Security incident: {event_type}"
        )
        
        # Store as active threat
        self.active_threats[event.event_id] = event
        
        # Automatic response based on severity
        if auto_respond:
            if severity == ThreatLevel.CRITICAL:
                # Immediate blocking
                if source_ip:
                    self.blocked_ips.add(source_ip)
                if user_id:
                    self.quarantined_users.add(user_id)
                
                logger.critical(f"Critical security incident: {event_type}", extra={
                    "event_id": event.event_id,
                    "source_ip": source_ip,
                    "user_id": user_id,
                    "auto_response": "immediate_blocking"
                })
            
            elif severity == ThreatLevel.HIGH:
                # Temporary rate limiting
                if source_ip or user_id:
                    key = user_id or source_ip
                    # Reduce rate limits dramatically
                    bucket = self.rate_limiter.buckets[key]
                    bucket["refill_rate"] = 1
                    bucket["burst_capacity"] = 5
                    bucket["blocked_until"] = time.time() + 300  # 5 minutes
        
        return event
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        return {
            "threat_summary": self.audit_logger.get_threat_summary(),
            "active_threats": len(self.active_threats),
            "blocked_ips": len(self.blocked_ips),
            "quarantined_users": len(self.quarantined_users),
            "rate_limiting_stats": {
                "active_buckets": len(self.rate_limiter.buckets),
                "total_violations": sum(b["violations"] for b in self.rate_limiter.buckets.values())
            },
            "compliance_status": {
                "data_processing_records": sum(len(records) for records in 
                                             self.compliance_manager.data_processing_records.values()),
                "consent_records": len(self.compliance_manager.consent_records),
                "expired_data_count": len(self.compliance_manager.check_data_retention())
            },
            "encryption_status": {
                "derived_keys": len(self.encryption_manager.derived_keys),
                "last_key_rotation": self.encryption_manager.last_rotation
            },
            "timestamp": time.time()
        }


# Factory function
def create_security_framework(config: Optional[Dict[str, Any]] = None) -> SecurityOrchestrator:
    """Create and initialize security framework."""
    return SecurityOrchestrator(config)


# Export main classes
__all__ = [
    "SecurityOrchestrator",
    "SecurityLevel",
    "ThreatLevel", 
    "SecurityEvent",
    "create_security_framework"
]