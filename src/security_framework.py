"""
Security framework for sentiment analyzer
Generation 2: Make It Robust - Comprehensive security measures
"""
import hashlib
import secrets
import time
import jwt
import logging
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from dataclasses import dataclass
from enum import Enum
from flask import request, jsonify, g
from cryptography.fernet import Fernet
import re

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event for logging and monitoring"""
    timestamp: float
    event_type: str
    severity: SecurityLevel
    source_ip: str
    user_agent: Optional[str]
    details: Dict[str, Any]
    blocked: bool = False

class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
        self.blocked_ips = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed based on rate limits"""
        current_time = time.time()
        
        # Clean old entries
        self._cleanup_old_entries(current_time)
        
        # Check if IP is blocked
        if client_id in self.blocked_ips:
            if current_time - self.blocked_ips[client_id] < 300:  # 5 min block
                return False
            else:
                del self.blocked_ips[client_id]
        
        # Initialize or get request count
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Count requests in current window
        window_start = current_time - self.time_window
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id] 
            if req_time > window_start
        ]
        
        # Check rate limit
        if len(self.requests[client_id]) >= self.max_requests:
            self.blocked_ips[client_id] = current_time
            logger.warning(f"Rate limit exceeded for {client_id}")
            return False
        
        # Add current request
        self.requests[client_id].append(current_time)
        return True
    
    def _cleanup_old_entries(self, current_time: float):
        """Clean up old request entries"""
        window_start = current_time - self.time_window
        
        # Clean request history
        for client_id in list(self.requests.keys()):
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id] 
                if req_time > window_start
            ]
            if not self.requests[client_id]:
                del self.requests[client_id]

class InputSanitizer:
    """Input sanitization and validation"""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 10000) -> str:
        """Sanitize text input"""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length]
        
        # Strip dangerous HTML/script patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'<iframe[^>]*>.*?</iframe>',
            r'javascript:',
            r'vbscript:',
            r'on\w+\s*=',
        ]
        
        for pattern in dangerous_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text.strip()
    
    @staticmethod
    def validate_json_payload(data: Dict[str, Any], 
                            required_fields: List[str],
                            max_size: int = 1024 * 1024) -> bool:
        """Validate JSON payload"""
        import json
        
        # Check size
        if len(json.dumps(data, default=str)) > max_size:
            raise ValueError("Payload too large")
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        return True
    
    @staticmethod
    def check_sql_injection_patterns(text: str) -> bool:
        """Check for SQL injection patterns"""
        sql_patterns = [
            r'\b(union|select|insert|update|delete|drop|create|alter)\b',
            r'[\'";].*--',
            r'\b(exec|execute|sp_)\b',
            r'[\'"]\s*;\s*--',
        ]
        
        text_lower = text.lower()
        for pattern in sql_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False

class JWTManager:
    """JWT token management"""
    
    def __init__(self, secret_key: Optional[str] = None, expiry_hours: int = 24):
        self.secret_key = secret_key or self._generate_secret_key()
        self.expiry_hours = expiry_hours
        self.algorithm = "HS256"
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key"""
        return secrets.token_urlsafe(64)
    
    def generate_token(self, payload: Dict[str, Any]) -> str:
        """Generate JWT token"""
        payload.update({
            'exp': time.time() + (self.expiry_hours * 3600),
            'iat': time.time(),
            'iss': 'sentiment_analyzer_pro'
        })
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

class SecurityMonitor:
    """Security event monitoring and alerting"""
    
    def __init__(self):
        self.events = []
        self.threat_scores = {}
        self.max_events = 1000
    
    def log_event(self, event: SecurityEvent):
        """Log security event"""
        self.events.append(event)
        
        # Keep only recent events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # Update threat score
        source = event.source_ip
        if source not in self.threat_scores:
            self.threat_scores[source] = 0
        
        # Increase threat score based on severity
        score_increment = {
            SecurityLevel.LOW: 1,
            SecurityLevel.MEDIUM: 3,
            SecurityLevel.HIGH: 10,
            SecurityLevel.CRITICAL: 50
        }
        
        self.threat_scores[source] += score_increment.get(event.severity, 1)
        
        # Log based on severity
        log_message = f"Security event: {event.event_type} from {event.source_ip}"
        if event.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            logger.error(log_message)
        elif event.severity == SecurityLevel.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_threat_score(self, source_ip: str) -> int:
        """Get threat score for IP"""
        return self.threat_scores.get(source_ip, 0)
    
    def is_high_risk(self, source_ip: str, threshold: int = 50) -> bool:
        """Check if IP is high risk"""
        return self.get_threat_score(source_ip) > threshold

class SecurityFramework:
    """Main security framework"""
    
    def __init__(self, 
                 rate_limit_per_minute: int = 60,
                 jwt_secret: Optional[str] = None):
        self.rate_limiter = RateLimiter(max_requests=rate_limit_per_minute)
        self.input_sanitizer = InputSanitizer()
        self.jwt_manager = JWTManager(secret_key=jwt_secret)
        self.security_monitor = SecurityMonitor()
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
    
    def encrypt_sensitive_data(self, data: str) -> bytes:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data.encode())
    
    def decrypt_sensitive_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data).decode()
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return password_hash.hex(), salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash"""
        computed_hash, _ = self.hash_password(password, salt)
        return secrets.compare_digest(computed_hash, password_hash)

def require_auth(security_framework: SecurityFramework):
    """Decorator for authentication requirement"""
    def decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            auth_header = request.headers.get('Authorization', '')
            
            if not auth_header.startswith('Bearer '):
                return jsonify({'error': 'Missing or invalid authorization header'}), 401
            
            token = auth_header.split(' ')[1]
            
            try:
                payload = security_framework.jwt_manager.validate_token(token)
                g.user_id = payload.get('user_id')
                g.token_payload = payload
                return f(*args, **kwargs)
            except ValueError as e:
                return jsonify({'error': str(e)}), 401
        
        return wrapper
    return decorator

def rate_limit(security_framework: SecurityFramework):
    """Decorator for rate limiting"""
    def decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            client_id = request.remote_addr or 'unknown'
            
            if not security_framework.rate_limiter.is_allowed(client_id):
                # Log security event
                security_framework.security_monitor.log_event(SecurityEvent(
                    timestamp=time.time(),
                    event_type="rate_limit_exceeded",
                    severity=SecurityLevel.MEDIUM,
                    source_ip=client_id,
                    user_agent=request.headers.get('User-Agent'),
                    details={"endpoint": request.endpoint},
                    blocked=True
                ))
                
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            return f(*args, **kwargs)
        return wrapper
    return decorator

def validate_input(security_framework: SecurityFramework, 
                  required_fields: Optional[List[str]] = None):
    """Decorator for input validation"""
    def decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if request.is_json:
                try:
                    data = request.get_json()
                    
                    # Validate JSON payload
                    if required_fields:
                        security_framework.input_sanitizer.validate_json_payload(
                            data, required_fields
                        )
                    
                    # Check for suspicious content
                    for key, value in data.items():
                        if isinstance(value, str):
                            # Check for SQL injection
                            if security_framework.input_sanitizer.check_sql_injection_patterns(value):
                                # Log security event
                                security_framework.security_monitor.log_event(SecurityEvent(
                                    timestamp=time.time(),
                                    event_type="sql_injection_attempt",
                                    severity=SecurityLevel.HIGH,
                                    source_ip=request.remote_addr or 'unknown',
                                    user_agent=request.headers.get('User-Agent'),
                                    details={"field": key, "value": value[:100]},
                                    blocked=True
                                ))
                                
                                return jsonify({'error': 'Invalid input detected'}), 400
                            
                            # Sanitize text
                            data[key] = security_framework.input_sanitizer.sanitize_text(value)
                    
                    # Store sanitized data for use in endpoint
                    g.validated_data = data
                    
                except ValueError as e:
                    return jsonify({'error': f'Input validation failed: {str(e)}'}), 400
            
            return f(*args, **kwargs)
        return wrapper
    return decorator

def create_secure_app_wrapper(app, security_config: Optional[Dict[str, Any]] = None):
    """Wrap Flask app with security framework"""
    
    security_config = security_config or {}
    security_framework = SecurityFramework(
        rate_limit_per_minute=security_config.get('rate_limit_per_minute', 60),
        jwt_secret=security_config.get('jwt_secret')
    )
    
    # Add security headers middleware
    @app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        return response
    
    # Add request logging
    @app.before_request
    def log_request():
        g.start_time = time.time()
        
        # Log suspicious requests
        user_agent = request.headers.get('User-Agent', '')
        if not user_agent or len(user_agent) > 500:
            security_framework.security_monitor.log_event(SecurityEvent(
                timestamp=time.time(),
                event_type="suspicious_user_agent",
                severity=SecurityLevel.LOW,
                source_ip=request.remote_addr or 'unknown',
                user_agent=user_agent[:100],
                details={"endpoint": request.endpoint}
            ))
    
    return security_framework

if __name__ == "__main__":
    # Test security framework
    security = SecurityFramework()
    
    # Test password hashing
    password = "test_password"
    password_hash, salt = security.hash_password(password)
    print(f"Password hashed: {len(password_hash)} chars")
    
    # Test password verification
    is_valid = security.verify_password(password, password_hash, salt)
    print(f"Password verification: {is_valid}")
    
    # Test JWT
    token = security.jwt_manager.generate_token({"user_id": "123", "role": "user"})
    print(f"Token generated: {token[:50]}...")
    
    try:
        payload = security.jwt_manager.validate_token(token)
        print(f"Token validated: {payload}")
    except Exception as e:
        print(f"Token validation failed: {e}")
    
    # Test input sanitization
    dangerous_input = "<script>alert('xss')</script>Hello world"
    sanitized = security.input_sanitizer.sanitize_text(dangerous_input)
    print(f"Sanitized: '{sanitized}'")
    
    print("Security framework test completed successfully!")