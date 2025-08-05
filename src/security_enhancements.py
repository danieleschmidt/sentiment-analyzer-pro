"""Advanced security enhancements for sentiment analyzer."""

import hashlib
import hmac
import jwt
import time
import secrets
import logging
from typing import Dict, Any, Optional, List
from functools import wraps
from flask import request, jsonify
import re
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityConfig:
    """Security configuration constants."""
    
    # JWT Configuration
    JWT_SECRET_KEY = secrets.token_urlsafe(32)  # In production, use env var
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = 24
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 60
    MAX_REQUESTS_PER_HOUR = 1000
    
    # Input validation
    MAX_TEXT_LENGTH = 10000
    ALLOWED_CONTENT_TYPES = ['application/json']
    
    # Security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'",
        'Referrer-Policy': 'strict-origin-when-cross-origin'
    }


class SecurityAuditLogger:
    """Enhanced security audit logging."""
    
    def __init__(self):
        self.security_logger = logging.getLogger('security_audit')
        self.security_logger.setLevel(logging.INFO)
        
        # Create dedicated security log handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.security_logger.addHandler(handler)
    
    def log_authentication_attempt(self, username: str, success: bool, ip_address: str):
        """Log authentication attempts."""
        status = "SUCCESS" if success else "FAILED"
        self.security_logger.info(
            f"Authentication {status} - User: {username}, IP: {ip_address}"
        )
    
    def log_suspicious_activity(self, activity_type: str, details: Dict[str, Any], ip_address: str):
        """Log suspicious activities."""
        self.security_logger.warning(
            f"Suspicious Activity - Type: {activity_type}, IP: {ip_address}, Details: {details}"
        )
    
    def log_security_violation(self, violation_type: str, details: Dict[str, Any], ip_address: str):
        """Log security violations."""
        self.security_logger.error(
            f"Security Violation - Type: {violation_type}, IP: {ip_address}, Details: {details}"
        )
    
    def log_data_access(self, user: str, data_type: str, action: str, ip_address: str):
        """Log data access for compliance."""
        self.security_logger.info(
            f"Data Access - User: {user}, Type: {data_type}, Action: {action}, IP: {ip_address}"
        )


class AdvancedInputValidator:
    """Advanced input validation and sanitization."""
    
    # Malicious patterns to detect
    MALICIOUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',  # JavaScript protocol
        r'vbscript:',  # VBScript protocol
        r'on\w+\s*=',  # Event handlers
        r'eval\s*\(',  # eval() calls
        r'exec\s*\(',  # exec() calls
        r'system\s*\(',  # system() calls
        r'__import__\s*\(',  # Python imports
        r'subprocess\.',  # Subprocess calls
        r'os\.',  # OS module calls
        r'\.\.\/',  # Path traversal
        r'\.\.\\\\'  # Windows path traversal
    ]
    
    SQL_INJECTION_PATTERNS = [
        r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER)\b',
        r'(\;|\-\-|\#|\/\*|\*\/)',
        r'(\bOR\b|\bAND\b).*(=|LIKE)',
        r'1\s*=\s*1',
        r'1\s*OR\s*1'
    ]
    
    def __init__(self):
        self.audit_logger = SecurityAuditLogger()
    
    def validate_text_input(self, text: str, max_length: int = None) -> Dict[str, Any]:
        """Comprehensive text input validation."""
        if max_length is None:
            max_length = SecurityConfig.MAX_TEXT_LENGTH
        
        result = {
            'is_valid': True,
            'sanitized_text': text,
            'warnings': [],
            'blocked_patterns': []
        }
        
        # Check length
        if len(text) > max_length:
            result['is_valid'] = False
            result['warnings'].append(f'Text exceeds maximum length of {max_length}')
            return result
        
        # Check for malicious patterns
        for pattern in self.MALICIOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                result['blocked_patterns'].append(pattern)
                result['warnings'].append(f'Potentially malicious pattern detected: {pattern}')
        
        # Check for SQL injection
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                result['blocked_patterns'].append(pattern)
                result['warnings'].append(f'Potential SQL injection pattern: {pattern}')
        
        # If malicious patterns found, mark as invalid
        if result['blocked_patterns']:
            result['is_valid'] = False
            self.audit_logger.log_security_violation(
                'malicious_input',
                {'patterns': result['blocked_patterns'], 'text_sample': text[:100]},
                request.remote_addr if request else 'unknown'
            )
        
        # Sanitize the text
        result['sanitized_text'] = self._sanitize_text(text)
        
        return result
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text by removing dangerous patterns."""
        sanitized = text
        
        # Remove script tags
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove dangerous protocols
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'vbscript:', '', sanitized, flags=re.IGNORECASE)
        
        # Remove event handlers
        sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized.strip())
        
        return sanitized
    
    def validate_batch_input(self, texts: List[str], max_batch_size: int = 100) -> Dict[str, Any]:
        """Validate batch text input."""
        result = {
            'is_valid': True,
            'sanitized_texts': [],
            'warnings': [],
            'invalid_indices': []
        }
        
        # Check batch size
        if len(texts) > max_batch_size:
            result['is_valid'] = False
            result['warnings'].append(f'Batch size {len(texts)} exceeds maximum of {max_batch_size}')
            return result
        
        # Validate each text
        for i, text in enumerate(texts):
            validation = self.validate_text_input(text)
            result['sanitized_texts'].append(validation['sanitized_text'])
            
            if not validation['is_valid']:
                result['invalid_indices'].append(i)
                result['warnings'].extend(validation['warnings'])
        
        # Mark batch as invalid if any text is invalid
        if result['invalid_indices']:
            result['is_valid'] = False
        
        return result


class JWTAuthenticationManager:
    """JWT-based authentication manager."""
    
    def __init__(self):
        self.audit_logger = SecurityAuditLogger()
    
    def generate_token(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate JWT token with user permissions."""
        if permissions is None:
            permissions = ['read']
        
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=SecurityConfig.JWT_EXPIRATION_HOURS),
            'iat': datetime.utcnow(),
            'jti': secrets.token_hex(16)  # Unique token ID
        }
        
        token = jwt.encode(payload, SecurityConfig.JWT_SECRET_KEY, algorithm=SecurityConfig.JWT_ALGORITHM)
        
        self.audit_logger.log_authentication_attempt(
            user_id, True, request.remote_addr if request else 'unknown'
        )
        
        return token
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token and return payload."""
        try:
            payload = jwt.decode(
                token, 
                SecurityConfig.JWT_SECRET_KEY, 
                algorithms=[SecurityConfig.JWT_ALGORITHM]
            )
            return {'valid': True, 'payload': payload}
        
        except jwt.ExpiredSignatureError:
            return {'valid': False, 'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'valid': False, 'error': 'Invalid token'}
    
    def require_permission(self, required_permission: str):
        """Decorator to require specific permission."""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                auth_header = request.headers.get('Authorization')
                
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({'error': 'Missing or invalid authorization header'}), 401
                
                token = auth_header.split(' ')[1]
                validation = self.validate_token(token)
                
                if not validation['valid']:
                    return jsonify({'error': validation['error']}), 401
                
                permissions = validation['payload'].get('permissions', [])
                if required_permission not in permissions and 'admin' not in permissions:
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                # Add user info to request context
                request.user_id = validation['payload']['user_id']
                request.permissions = permissions
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator


class DataEncryption:
    """Data encryption utilities."""
    
    @staticmethod
    def encrypt_sensitive_data(data: str, key: bytes = None) -> Dict[str, str]:
        """Encrypt sensitive data using AES encryption."""
        try:
            from cryptography.fernet import Fernet
            
            if key is None:
                key = Fernet.generate_key()
            
            fernet = Fernet(key)
            encrypted_data = fernet.encrypt(data.encode())
            
            return {
                'encrypted_data': encrypted_data.decode(),
                'key': key.decode()
            }
        except ImportError:
            logger.warning("Cryptography library not available for encryption")
            return {'encrypted_data': data, 'key': 'none'}
    
    @staticmethod
    def decrypt_sensitive_data(encrypted_data: str, key: str) -> str:
        """Decrypt sensitive data."""
        try:
            from cryptography.fernet import Fernet
            
            if key == 'none':
                return encrypted_data
            
            fernet = Fernet(key.encode())
            decrypted_data = fernet.decrypt(encrypted_data.encode())
            
            return decrypted_data.decode()
        except ImportError:
            logger.warning("Cryptography library not available for decryption")
            return encrypted_data


class ComplianceManager:
    """GDPR, CCPA, and other compliance features."""
    
    def __init__(self):
        self.audit_logger = SecurityAuditLogger()
    
    def log_data_processing(self, user_id: str, data_type: str, purpose: str, legal_basis: str):
        """Log data processing activities for GDPR compliance."""
        self.audit_logger.log_data_access(
            user_id, data_type, f"Processing for {purpose} (Legal basis: {legal_basis})",
            request.remote_addr if request else 'unknown'
        )
    
    def handle_data_deletion_request(self, user_id: str) -> Dict[str, Any]:
        """Handle GDPR data deletion requests."""
        # In a real implementation, this would delete user data from databases
        self.audit_logger.log_data_access(
            user_id, 'all_user_data', 'deletion_request',
            request.remote_addr if request else 'unknown'
        )
        
        return {
            'status': 'scheduled',
            'message': 'Data deletion request has been logged and will be processed within 30 days',
            'request_id': secrets.token_hex(8)
        }
    
    def generate_data_export(self, user_id: str) -> Dict[str, Any]:
        """Generate data export for GDPR data portability requests."""
        # In a real implementation, this would collect all user data
        self.audit_logger.log_data_access(
            user_id, 'all_user_data', 'export_request',
            request.remote_addr if request else 'unknown'
        )
        
        return {
            'status': 'scheduled',
            'message': 'Data export request has been logged and will be processed within 7 days',
            'request_id': secrets.token_hex(8)
        }


class SecurityMiddleware:
    """Security middleware for Flask applications."""
    
    def __init__(self, app=None):
        self.app = app
        self.audit_logger = SecurityAuditLogger()
        self.input_validator = AdvancedInputValidator()
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize security middleware with Flask app."""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
    
    def before_request(self):
        """Security checks before processing request."""
        # Check content type for POST requests
        if request.method == 'POST':
            if request.content_type not in SecurityConfig.ALLOWED_CONTENT_TYPES:
                self.audit_logger.log_security_violation(
                    'invalid_content_type',
                    {'content_type': request.content_type},
                    request.remote_addr
                )
                return jsonify({'error': 'Invalid content type'}), 400
        
        # Log suspicious user agents
        user_agent = request.headers.get('User-Agent', '')
        if self._is_suspicious_user_agent(user_agent):
            self.audit_logger.log_suspicious_activity(
                'suspicious_user_agent',
                {'user_agent': user_agent},
                request.remote_addr
            )
    
    def after_request(self, response):
        """Add security headers to response."""
        for header, value in SecurityConfig.SECURITY_HEADERS.items():
            response.headers[header] = value
        return response
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent looks suspicious."""
        suspicious_patterns = [
            r'sqlmap',
            r'nikto',
            r'nmap',
            r'masscan',
            r'curl.*script',
            r'python-requests.*bot'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, user_agent, re.IGNORECASE):
                return True
        
        return False


# Global instances
security_audit_logger = SecurityAuditLogger()
input_validator = AdvancedInputValidator()
jwt_manager = JWTAuthenticationManager()
compliance_manager = ComplianceManager()
security_middleware = SecurityMiddleware()


def secure_endpoint(require_auth: bool = False, permission: str = None):
    """Decorator for securing endpoints with comprehensive protection."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Authentication check
            if require_auth:
                if permission:
                    # Use JWT manager for permission check
                    return jwt_manager.require_permission(permission)(f)(*args, **kwargs)
                else:
                    # Basic auth check
                    auth_header = request.headers.get('Authorization')
                    if not auth_header:
                        return jsonify({'error': 'Authentication required'}), 401
            
            # Input validation for JSON requests
            if request.is_json:
                try:
                    data = request.get_json()
                    if 'text' in data:
                        validation = input_validator.validate_text_input(data['text'])
                        if not validation['is_valid']:
                            return jsonify({
                                'error': 'Invalid input',
                                'warnings': validation['warnings']
                            }), 400
                        # Replace with sanitized text
                        data['text'] = validation['sanitized_text']
                        request._cached_json = data
                    
                    elif 'texts' in data:
                        validation = input_validator.validate_batch_input(data['texts'])
                        if not validation['is_valid']:
                            return jsonify({
                                'error': 'Invalid batch input',
                                'warnings': validation['warnings']
                            }), 400
                        # Replace with sanitized texts
                        data['texts'] = validation['sanitized_texts']
                        request._cached_json = data
                
                except Exception as e:
                    security_audit_logger.log_security_violation(
                        'input_validation_error',
                        {'error': str(e)},
                        request.remote_addr
                    )
                    return jsonify({'error': 'Input validation failed'}), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator