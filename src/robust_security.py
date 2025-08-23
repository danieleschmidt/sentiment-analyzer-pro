
import hashlib
import secrets
import hmac
import time
import jwt
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import re

class SecurityManager:
    """Comprehensive security management system."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.failed_attempts = {}
        self.rate_limits = {}
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple:
        """Secure password hashing with salt."""
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
        """Verify password against hash."""
        computed_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(password_hash, computed_hash)
    
    def generate_token(self, payload: Dict[str, Any], expiry_hours: int = 24) -> str:
        """Generate JWT token with expiry."""
        payload_copy = payload.copy()
        payload_copy['exp'] = datetime.utcnow() + timedelta(hours=expiry_hours)
        payload_copy['iat'] = datetime.utcnow()
        
        return jwt.encode(payload_copy, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        # Remove potential SQL injection patterns
        sql_patterns = [r"'", r'"', r'--', r';', r'/\*', r'\*/', r'xp_', r'sp_']
        for pattern in sql_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove potential XSS patterns
        xss_patterns = [r'<script>', r'</script>', r'javascript:', r'onload=', r'onerror=']
        for pattern in xss_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, 
                        window_minutes: int = 60) -> bool:
        """Check if request is within rate limit."""
        current_time = time.time()
        window_seconds = window_minutes * 60
        
        # Cleanup old entries
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_rate_limits()
            self._last_cleanup = current_time
        
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        # Remove old timestamps outside the window
        self.rate_limits[identifier] = [
            timestamp for timestamp in self.rate_limits[identifier]
            if current_time - timestamp < window_seconds
        ]
        
        # Check if within limit
        if len(self.rate_limits[identifier]) >= max_requests:
            return False
        
        # Add current timestamp
        self.rate_limits[identifier].append(current_time)
        return True
    
    def _cleanup_rate_limits(self):
        """Clean up old rate limit entries."""
        current_time = time.time()
        for identifier in list(self.rate_limits.keys()):
            self.rate_limits[identifier] = [
                timestamp for timestamp in self.rate_limits[identifier]
                if current_time - timestamp < 3600  # Keep last hour
            ]
            
            if not self.rate_limits[identifier]:
                del self.rate_limits[identifier]

class InputValidator:
    """Advanced input validation with security checks."""
    
    @staticmethod
    def validate_text_secure(text: str, max_length: int = 10000) -> str:
        """Validate text with security checks."""
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text input")
        
        # Check length
        if len(text) > max_length:
            raise ValueError(f"Text too long (max {max_length} characters)")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                raise ValueError("Potentially malicious content detected")
        
        return text.strip()
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model configuration with security checks."""
        validated = {}
        
        # Safe parameter ranges
        safe_ranges = {
            'batch_size': (1, 1000),
            'learning_rate': (1e-6, 1.0),
            'max_epochs': (1, 1000),
            'max_length': (1, 10000),
        }
        
        for key, value in config.items():
            if key in safe_ranges:
                min_val, max_val = safe_ranges[key]
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    raise ValueError(f"{key} must be between {min_val} and {max_val}")
                validated[key] = value
            elif key in ['model_name', 'output_dir']:
                if not isinstance(value, str) or len(value) > 200:
                    raise ValueError(f"{key} must be a string with max 200 characters")
                # Prevent path traversal
                if '..' in value or value.startswith('/'):
                    raise ValueError(f"Invalid path in {key}")
                validated[key] = value
        
        return validated

# Global security manager
security_manager = SecurityManager()
input_validator = InputValidator()
