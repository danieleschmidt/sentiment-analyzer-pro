"""Advanced security hardening with threat detection and response."""

import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict, deque
import ipaddress

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AttackType(Enum):
    """Types of security attacks."""
    BRUTE_FORCE = "brute_force"
    INJECTION = "injection"
    XSS = "xss"
    CSRF = "csrf"
    RATE_LIMIT_ABUSE = "rate_limit_abuse"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    MALICIOUS_PAYLOAD = "malicious_payload"
    CREDENTIAL_STUFFING = "credential_stuffing"

@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    timestamp: datetime
    threat_level: ThreatLevel
    attack_type: AttackType
    source_ip: str
    user_agent: Optional[str]
    request_path: str
    details: Dict[str, Any]
    blocked: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['threat_level'] = self.threat_level.value
        data['attack_type'] = self.attack_type.value
        return data

class IPReputation:
    """IP reputation management system."""
    
    def __init__(self):
        self.reputation_scores: Dict[str, float] = {}
        self.blocked_ips: Set[str] = set()
        self.trusted_ips: Set[str] = set()
        self.suspicious_ips: Dict[str, datetime] = {}
        self.ip_activity: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._lock = threading.Lock()
    
    def add_trusted_ip(self, ip: str):
        """Add IP to trusted list."""
        with self._lock:
            self.trusted_ips.add(ip)
            if ip in self.blocked_ips:
                self.blocked_ips.remove(ip)
    
    def block_ip(self, ip: str, duration_hours: int = 24):
        """Block IP for specified duration."""
        with self._lock:
            self.blocked_ips.add(ip)
            self.suspicious_ips[ip] = datetime.now() + timedelta(hours=duration_hours)
            logger.warning(f"IP {ip} blocked for {duration_hours} hours")
    
    def is_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        with self._lock:
            if ip in self.trusted_ips:
                return False
            
            if ip in self.blocked_ips:
                if ip in self.suspicious_ips:
                    if datetime.now() > self.suspicious_ips[ip]:
                        self.blocked_ips.remove(ip)
                        del self.suspicious_ips[ip]
                        return False
                return True
            
            return False
    
    def record_activity(self, ip: str, activity_type: str, success: bool):
        """Record IP activity."""
        with self._lock:
            self.ip_activity[ip].append({
                'timestamp': datetime.now(),
                'activity': activity_type,
                'success': success
            })
            
            # Update reputation score
            if success:
                self.reputation_scores[ip] = max(
                    self.reputation_scores.get(ip, 0.5) + 0.1, 1.0
                )
            else:
                self.reputation_scores[ip] = min(
                    self.reputation_scores.get(ip, 0.5) - 0.2, 0.0
                )
                
                # Auto-block if reputation drops too low
                if self.reputation_scores[ip] <= 0.1:
                    self.block_ip(ip, 1)
    
    def get_reputation(self, ip: str) -> float:
        """Get IP reputation score (0.0 to 1.0)."""
        return self.reputation_scores.get(ip, 0.5)
    
    def analyze_ip_patterns(self, ip: str) -> Dict[str, Any]:
        """Analyze patterns for IP."""
        with self._lock:
            if ip not in self.ip_activity:
                return {"pattern": "unknown", "risk_score": 0.5}
            
            activities = list(self.ip_activity[ip])
            if not activities:
                return {"pattern": "no_activity", "risk_score": 0.5}
            
            # Calculate metrics
            total_requests = len(activities)
            failed_requests = sum(1 for a in activities if not a['success'])
            failure_rate = failed_requests / total_requests if total_requests > 0 else 0
            
            # Time-based analysis
            recent_activities = [
                a for a in activities 
                if (datetime.now() - a['timestamp']).total_seconds() < 300  # Last 5 minutes
            ]
            
            request_rate = len(recent_activities) / 5.0  # requests per second
            
            # Pattern classification
            if failure_rate > 0.8 and total_requests > 10:
                pattern = "brute_force"
                risk_score = 0.9
            elif request_rate > 10:
                pattern = "rate_abuse"
                risk_score = 0.8
            elif failure_rate > 0.5:
                pattern = "suspicious"
                risk_score = 0.7
            else:
                pattern = "normal"
                risk_score = 0.3
            
            return {
                "pattern": pattern,
                "risk_score": risk_score,
                "total_requests": total_requests,
                "failure_rate": failure_rate,
                "request_rate": request_rate,
                "reputation": self.get_reputation(ip)
            }

class InputSanitizer:
    """Advanced input sanitization and validation."""
    
    INJECTION_PATTERNS = [
        r'(?i)(union\s+select|select\s+.*\s+from)',  # SQL Injection
        r'(?i)(script\s*>|javascript:|vbscript:)',   # XSS
        r'(?i)(exec\s*\(|eval\s*\(|system\s*\()',   # Code injection
        r'(?i)(\.\.\/|\.\.\\)',                      # Path traversal
        r'(?i)(drop\s+table|delete\s+from)',        # SQL manipulation
        r'<\s*script[^>]*>.*?<\s*/\s*script\s*>',   # Script tags
        r'(?i)(document\.cookie|document\.write)',   # DOM manipulation
    ]
    
    SUSPICIOUS_PATTERNS = [
        r'(?i)(password|passwd|pwd)=',               # Password exposure
        r'(?i)(admin|root|administrator)',           # Privileged accounts
        r'[^\x20-\x7E]',                            # Non-printable characters
        r'(?i)(base64|eval|decode)',                 # Encoding/decoding attempts
    ]
    
    @classmethod
    def sanitize_input(cls, value: Any) -> Tuple[Any, List[str]]:
        """Sanitize input and return warnings."""
        if not isinstance(value, str):
            return value, []
        
        warnings = []
        
        # Check for injection patterns
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, value):
                warnings.append(f"Potential injection detected: {pattern}")
        
        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, value):
                warnings.append(f"Suspicious pattern detected: {pattern}")
        
        # Basic sanitization
        sanitized = re.sub(r'[<>"\']', '', value)  # Remove dangerous characters
        sanitized = sanitized.strip()
        
        # Length validation
        if len(sanitized) > 10000:
            warnings.append("Input exceeds maximum length")
            sanitized = sanitized[:10000]
        
        return sanitized, warnings
    
    @classmethod
    def validate_json_payload(cls, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Validate and sanitize JSON payload."""
        sanitized = {}
        all_warnings = []
        
        for key, value in payload.items():
            # Sanitize key
            sanitized_key, key_warnings = cls.sanitize_input(key)
            all_warnings.extend([f"Key '{key}': {w}" for w in key_warnings])
            
            # Sanitize value
            if isinstance(value, dict):
                sanitized_value, value_warnings = cls.validate_json_payload(value)
                all_warnings.extend([f"Value '{key}': {w}" for w in value_warnings])
            elif isinstance(value, list):
                sanitized_value = []
                for i, item in enumerate(value):
                    sanitized_item, item_warnings = cls.sanitize_input(item)
                    sanitized_value.append(sanitized_item)
                    all_warnings.extend([f"List item {i} in '{key}': {w}" for w in item_warnings])
            else:
                sanitized_value, value_warnings = cls.sanitize_input(value)
                all_warnings.extend([f"Value '{key}': {w}" for w in value_warnings])
            
            sanitized[sanitized_key] = sanitized_value
        
        return sanitized, all_warnings

class ThreatDetector:
    """Real-time threat detection system."""
    
    def __init__(self):
        self.ip_reputation = IPReputation()
        self.security_events: List[SecurityEvent] = []
        self.alert_thresholds = {
            ThreatLevel.LOW: 10,
            ThreatLevel.MEDIUM: 5,
            ThreatLevel.HIGH: 2,
            ThreatLevel.CRITICAL: 1
        }
        self._lock = threading.Lock()
    
    def analyze_request(
        self,
        ip: str,
        user_agent: Optional[str],
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Tuple[ThreatLevel, List[str], bool]:
        """Analyze request for security threats."""
        threats = []
        max_threat_level = ThreatLevel.LOW
        should_block = False
        
        # Check IP reputation
        if self.ip_reputation.is_blocked(ip):
            threats.append("IP is blocked")
            max_threat_level = ThreatLevel.CRITICAL
            should_block = True
        
        # Analyze IP patterns
        ip_analysis = self.ip_reputation.analyze_ip_patterns(ip)
        if ip_analysis["risk_score"] > 0.8:
            threats.append(f"High-risk IP pattern: {ip_analysis['pattern']}")
            max_threat_level = ThreatLevel.HIGH
        
        # Analyze payload
        if payload:
            sanitized_payload, warnings = InputSanitizer.validate_json_payload(payload)
            if warnings:
                threats.extend(warnings)
                max_threat_level = ThreatLevel.MEDIUM
                
                # Check for critical injection patterns
                for warning in warnings:
                    if "injection" in warning.lower():
                        max_threat_level = ThreatLevel.HIGH
                        should_block = True
        
        # Analyze user agent
        if user_agent:
            suspicious_agents = [
                'sqlmap', 'nikto', 'nmap', 'burp', 'owasp',
                'dirbuster', 'gobuster', 'wfuzz'
            ]
            if any(agent in user_agent.lower() for agent in suspicious_agents):
                threats.append("Suspicious user agent detected")
                max_threat_level = ThreatLevel.HIGH
                should_block = True
        
        # Analyze request path
        suspicious_paths = [
            '/admin', '/wp-admin', '/.env', '/config',
            '/backup', '/phpmyadmin', '/.git'
        ]
        if any(path.startswith(sp) for sp in suspicious_paths):
            threats.append("Access to sensitive path attempted")
            max_threat_level = ThreatLevel.MEDIUM
        
        # Record security event if threats detected
        if threats:
            event = SecurityEvent(
                event_id=f"threat_{int(time.time())}_{secrets.token_hex(4)}",
                timestamp=datetime.now(),
                threat_level=max_threat_level,
                attack_type=self._classify_attack_type(threats),
                source_ip=ip,
                user_agent=user_agent,
                request_path=path,
                details={
                    "threats": threats,
                    "ip_analysis": ip_analysis,
                    "payload_size": len(str(payload)) if payload else 0
                },
                blocked=should_block
            )
            
            self.record_security_event(event)
        
        return max_threat_level, threats, should_block
    
    def _classify_attack_type(self, threats: List[str]) -> AttackType:
        """Classify attack type based on threat patterns."""
        threat_text = ' '.join(threats).lower()
        
        if 'injection' in threat_text:
            return AttackType.INJECTION
        elif 'script' in threat_text or 'xss' in threat_text:
            return AttackType.XSS
        elif 'brute_force' in threat_text:
            return AttackType.BRUTE_FORCE
        elif 'rate' in threat_text:
            return AttackType.RATE_LIMIT_ABUSE
        elif 'user agent' in threat_text:
            return AttackType.SUSPICIOUS_PATTERN
        else:
            return AttackType.MALICIOUS_PAYLOAD
    
    def record_security_event(self, event: SecurityEvent):
        """Record security event."""
        with self._lock:
            self.security_events.append(event)
            
            # Keep only last 10000 events
            if len(self.security_events) > 10000:
                self.security_events = self.security_events[-10000:]
        
        # Log event
        log_level = {
            ThreatLevel.LOW: logging.INFO,
            ThreatLevel.MEDIUM: logging.WARNING,
            ThreatLevel.HIGH: logging.ERROR,
            ThreatLevel.CRITICAL: logging.CRITICAL
        }[event.threat_level]
        
        logger.log(
            log_level,
            f"Security event [{event.event_id}]: {event.attack_type.value} from {event.source_ip}",
            extra={"security_event": event.to_dict()}
        )
        
        # Auto-block high/critical threats
        if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.ip_reputation.block_ip(event.source_ip, 24)
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security statistics."""
        with self._lock:
            if not self.security_events:
                return {"total_events": 0}
            
            # Calculate statistics
            threat_counts = defaultdict(int)
            attack_counts = defaultdict(int)
            blocked_count = 0
            
            for event in self.security_events:
                threat_counts[event.threat_level.value] += 1
                attack_counts[event.attack_type.value] += 1
                if event.blocked:
                    blocked_count += 1
            
            return {
                "total_events": len(self.security_events),
                "blocked_requests": blocked_count,
                "threat_level_breakdown": dict(threat_counts),
                "attack_type_breakdown": dict(attack_counts),
                "blocked_ips": len(self.ip_reputation.blocked_ips),
                "trusted_ips": len(self.ip_reputation.trusted_ips)
            }

class SecurityMiddleware:
    """Security middleware for request processing."""
    
    def __init__(self, threat_detector: ThreatDetector):
        self.threat_detector = threat_detector
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))
        self.csrf_tokens: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    def validate_rate_limit(
        self,
        identifier: str,
        max_requests: int = 100,
        window_seconds: int = 60
    ) -> bool:
        """Validate request rate limit."""
        now = time.time()
        
        with self._lock:
            # Clean old requests
            self.rate_limits[identifier] = deque([
                req_time for req_time in self.rate_limits[identifier]
                if now - req_time < window_seconds
            ], maxlen=100)
            
            # Check limit
            if len(self.rate_limits[identifier]) >= max_requests:
                return False
            
            # Add current request
            self.rate_limits[identifier].append(now)
            return True
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token."""
        token = secrets.token_urlsafe(32)
        with self._lock:
            self.csrf_tokens[token] = datetime.now() + timedelta(hours=1)
        return token
    
    def validate_csrf_token(self, token: str) -> bool:
        """Validate CSRF token."""
        with self._lock:
            if token not in self.csrf_tokens:
                return False
            
            if datetime.now() > self.csrf_tokens[token]:
                del self.csrf_tokens[token]
                return False
            
            return True
    
    def process_request(
        self,
        ip: str,
        user_agent: Optional[str],
        path: str,
        method: str,
        headers: Dict[str, str],
        payload: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Process request through security middleware."""
        # Rate limiting
        if not self.validate_rate_limit(ip):
            return False, {
                "error": "Rate limit exceeded",
                "retry_after": 60
            }
        
        # Threat detection
        threat_level, threats, should_block = self.threat_detector.analyze_request(
            ip, user_agent, path, payload, headers
        )
        
        if should_block:
            return False, {
                "error": "Request blocked due to security concerns",
                "threat_level": threat_level.value,
                "threats": threats
            }
        
        # Log successful validation
        self.threat_detector.ip_reputation.record_activity(ip, "request", True)
        
        return True, {
            "threat_level": threat_level.value,
            "warnings": threats if threats else []
        }

# Global security components
_global_threat_detector = ThreatDetector()
_global_security_middleware = SecurityMiddleware(_global_threat_detector)

def get_threat_detector() -> ThreatDetector:
    """Get global threat detector."""
    return _global_threat_detector

def get_security_middleware() -> SecurityMiddleware:
    """Get global security middleware."""
    return _global_security_middleware

def secure_hash(data: str, salt: str = None) -> str:
    """Generate secure hash with salt."""
    if salt is None:
        salt = secrets.token_hex(16)
    
    return hashlib.pbkdf2_hmac(
        'sha256',
        data.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # iterations
    ).hex()

def verify_hash(data: str, hash_value: str, salt: str) -> bool:
    """Verify hash with salt."""
    return hmac.compare_digest(secure_hash(data, salt), hash_value)