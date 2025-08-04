"""
Photonic-MLIR Bridge Security Module

Provides comprehensive security measures including input validation,
sanitization, access control, and threat detection for photonic circuit synthesis.
"""

import re
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security validation levels."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


class ThreatType(Enum):
    """Types of security threats."""
    INJECTION = "injection"
    DOS = "denial_of_service"
    MALFORMED_INPUT = "malformed_input"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    max_component_count: int = 10000
    max_connection_count: int = 50000
    max_parameter_value: float = 1e6
    max_string_length: int = 1000
    max_circuit_name_length: int = 100
    max_metadata_size: int = 10000
    allowed_parameter_keys: Set[str] = None
    blocked_patterns: List[str] = None
    security_level: SecurityLevel = SecurityLevel.STRICT
    
    def __post_init__(self):
        if self.allowed_parameter_keys is None:
            self.allowed_parameter_keys = {
                'length', 'width', 'height', 'radius', 'ratio', 'phase_shift',
                'coupling_length', 'gap', 'coupling_ratio', 'target_wavelength',
                'responsivity', 'dark_current', 'bandwidth', 'power_dbm',
                'modulation_depth', 'vpi', 'wavelength', 'activation_type',
                'nonlinear_coefficient', 'num_inputs', 'combining_loss'
            }
        
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r'<script.*?>.*?</script>',  # Script injection
                r'javascript:',              # JavaScript URLs
                r'data:.*base64',           # Base64 data URLs
                r'eval\s*\(',               # Code execution
                r'exec\s*\(',               # Code execution
                r'system\s*\(',             # System calls
                r'import\s+',               # Import statements
                r'__.*__',                  # Python special methods
                r'\.\./',                   # Path traversal
                r'[<>"\']',                 # Basic XSS chars (in paranoid mode)
            ]


class SecurityValidator:
    """Validates inputs for security threats."""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.threat_log: List[Dict] = []
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup security logging."""
        self.security_logger = logging.getLogger(f"{__name__}.security")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.security_logger.addHandler(handler)
        self.security_logger.setLevel(logging.WARNING)
    
    def validate_string(self, value: str, field_name: str) -> bool:
        """Validate string input for security threats."""
        if not isinstance(value, str):
            self._log_threat(ThreatType.MALFORMED_INPUT, 
                           f"Non-string value for {field_name}: {type(value)}")
            return False
        
        # Length check
        if len(value) > self.config.max_string_length:
            self._log_threat(ThreatType.RESOURCE_EXHAUSTION,
                           f"String too long for {field_name}: {len(value)}")
            return False
        
        # Pattern checks
        for pattern in self.config.blocked_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                self._log_threat(ThreatType.INJECTION,
                               f"Malicious pattern in {field_name}: {pattern}")
                return False
        
        # Special checks for paranoid mode
        if self.config.security_level == SecurityLevel.PARANOID:
            if self._contains_suspicious_content(value):
                self._log_threat(ThreatType.INJECTION,
                               f"Suspicious content in {field_name}")
                return False
        
        return True
    
    def validate_numeric(self, value: Union[int, float], field_name: str) -> bool:
        """Validate numeric input."""
        if not isinstance(value, (int, float)):
            self._log_threat(ThreatType.MALFORMED_INPUT,
                           f"Non-numeric value for {field_name}: {type(value)}")
            return False
        
        # Range check
        if abs(value) > self.config.max_parameter_value:
            self._log_threat(ThreatType.RESOURCE_EXHAUSTION,
                           f"Value too large for {field_name}: {value}")
            return False
        
        # Check for special float values
        if isinstance(value, float):
            if not (value == value):  # NaN check
                self._log_threat(ThreatType.MALFORMED_INPUT,
                               f"NaN value for {field_name}")
                return False
            
            if abs(value) == float('inf'):
                self._log_threat(ThreatType.MALFORMED_INPUT,
                               f"Infinite value for {field_name}")
                return False
        
        return True
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate component parameters."""
        if not isinstance(parameters, dict):
            self._log_threat(ThreatType.MALFORMED_INPUT,
                           "Parameters must be a dictionary")
            return False
        
        if len(parameters) > 50:  # Reasonable limit
            self._log_threat(ThreatType.RESOURCE_EXHAUSTION,
                           f"Too many parameters: {len(parameters)}")
            return False
        
        for key, value in parameters.items():
            # Validate key
            if not self.validate_string(key, "parameter_key"):
                return False
            
            # Check if key is allowed
            if self.config.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
                if key not in self.config.allowed_parameter_keys:
                    self._log_threat(ThreatType.UNAUTHORIZED_ACCESS,
                                   f"Unauthorized parameter key: {key}")
                    return False
            
            # Validate value
            if isinstance(value, str):
                if not self.validate_string(value, f"parameter_{key}"):
                    return False
            elif isinstance(value, (int, float)):
                if not self.validate_numeric(value, f"parameter_{key}"):
                    return False
            elif isinstance(value, (list, dict)):
                # Serialize and check size
                try:
                    serialized = json.dumps(value)
                    if len(serialized) > 1000:  # 1KB limit for complex values
                        self._log_threat(ThreatType.RESOURCE_EXHAUSTION,
                                       f"Parameter value too large: {key}")
                        return False
                except (TypeError, ValueError):
                    self._log_threat(ThreatType.MALFORMED_INPUT,
                                   f"Non-serializable parameter: {key}")
                    return False
            else:
                self._log_threat(ThreatType.MALFORMED_INPUT,
                               f"Invalid parameter type for {key}: {type(value)}")
                return False
        
        return True
    
    def validate_circuit_limits(self, component_count: int, connection_count: int) -> bool:
        """Validate circuit size limits."""
        if component_count > self.config.max_component_count:
            self._log_threat(ThreatType.RESOURCE_EXHAUSTION,
                           f"Too many components: {component_count}")
            return False
        
        if connection_count > self.config.max_connection_count:
            self._log_threat(ThreatType.RESOURCE_EXHAUSTION,
                           f"Too many connections: {connection_count}")
            return False
        
        # Check for unreasonable connection density
        if component_count > 0:
            connection_ratio = connection_count / component_count
            if connection_ratio > 10:  # Very high connectivity
                self._log_threat(ThreatType.DOS,
                               f"Excessive connection density: {connection_ratio}")
                return False
        
        return True
    
    def validate_circuit_name(self, name: str) -> bool:
        """Validate circuit name."""
        if not self.validate_string(name, "circuit_name"):
            return False
        
        if len(name) > self.config.max_circuit_name_length:
            self._log_threat(ThreatType.RESOURCE_EXHAUSTION,
                           f"Circuit name too long: {len(name)}")
            return False
        
        # Ensure name is filesystem-safe
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', name):
            self._log_threat(ThreatType.INJECTION,
                           f"Invalid characters in circuit name: {name}")
            return False
        
        return True
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate circuit metadata."""
        if not isinstance(metadata, dict):
            self._log_threat(ThreatType.MALFORMED_INPUT,
                           "Metadata must be a dictionary")
            return False
        
        # Size check
        try:
            serialized = json.dumps(metadata)
            if len(serialized) > self.config.max_metadata_size:
                self._log_threat(ThreatType.RESOURCE_EXHAUSTION,
                               f"Metadata too large: {len(serialized)}")
                return False
        except (TypeError, ValueError):
            self._log_threat(ThreatType.MALFORMED_INPUT,
                           "Non-serializable metadata")
            return False
        
        # Validate each metadata field
        for key, value in metadata.items():
            if not self.validate_string(str(key), "metadata_key"):
                return False
            
            if isinstance(value, str):
                if not self.validate_string(value, f"metadata_{key}"):
                    return False
            elif isinstance(value, (int, float)):
                if not self.validate_numeric(value, f"metadata_{key}"):
                    return False
        
        return True
    
    def _contains_suspicious_content(self, text: str) -> bool:
        """Check for suspicious content patterns."""
        suspicious_patterns = [
            r'rm\s+-rf',           # Dangerous shell commands
            r'DROP\s+TABLE',       # SQL injection
            r'UNION\s+SELECT',     # SQL injection
            r'CREATE\s+TABLE',     # SQL injection
            r'/etc/passwd',        # System file access
            r'/proc/',             # Process information
            r'file://',            # Local file access
            r'ftp://',             # FTP URLs
            r'\\x[0-9a-fA-F]{2}',  # Hex encoding
            r'%[0-9a-fA-F]{2}',    # URL encoding
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _log_threat(self, threat_type: ThreatType, description: str):
        """Log security threat."""
        threat_info = {
            'timestamp': time.time(),
            'threat_type': threat_type.value,
            'description': description,
            'security_level': self.config.security_level.value
        }
        
        self.threat_log.append(threat_info)
        self.security_logger.warning(f"Security threat detected: {threat_type.value} - {description}")
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of detected threats."""
        if not self.threat_log:
            return {"total_threats": 0, "threat_types": {}}
        
        threat_counts = {}
        for threat in self.threat_log:
            threat_type = threat['threat_type']
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        return {
            "total_threats": len(self.threat_log),
            "threat_types": threat_counts,
            "latest_threat": self.threat_log[-1] if self.threat_log else None
        }


class InputSanitizer:
    """Sanitizes user inputs to prevent security issues."""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            return str(value)[:max_length]
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Remove control characters except whitespace
        value = ''.join(char for char in value 
                       if ord(char) >= 32 or char in '\t\n\r')
        
        # Truncate to max length
        if len(value) > max_length:
            value = value[:max_length]
        
        # Remove leading/trailing whitespace
        value = value.strip()
        
        return value
    
    @staticmethod
    def sanitize_numeric(value: Union[int, float], 
                        min_val: float = -1e6, 
                        max_val: float = 1e6) -> Union[int, float]:
        """Sanitize numeric input."""
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                return 0.0
        
        # Handle special float values
        if isinstance(value, float):
            if not (value == value):  # NaN
                return 0.0
            if abs(value) == float('inf'):
                return max_val if value > 0 else min_val
        
        # Clamp to range
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def sanitize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameter dictionary."""
        if not isinstance(parameters, dict):
            return {}
        
        sanitized = {}
        for key, value in parameters.items():
            # Sanitize key
            clean_key = InputSanitizer.sanitize_string(str(key), 50)
            if not clean_key:
                continue
            
            # Sanitize value
            if isinstance(value, str):
                sanitized[clean_key] = InputSanitizer.sanitize_string(value)
            elif isinstance(value, (int, float)):
                sanitized[clean_key] = InputSanitizer.sanitize_numeric(value)
            elif isinstance(value, (list, tuple)):
                # Sanitize list elements
                sanitized_list = []
                for item in value[:100]:  # Limit list size
                    if isinstance(item, str):
                        sanitized_list.append(InputSanitizer.sanitize_string(item))
                    elif isinstance(item, (int, float)):
                        sanitized_list.append(InputSanitizer.sanitize_numeric(item))
                sanitized[clean_key] = sanitized_list
            elif isinstance(value, dict):
                # Recursively sanitize nested dict
                sanitized[clean_key] = InputSanitizer.sanitize_parameters(value)
        
        return sanitized


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        current_time = time.time()
        
        # Periodic cleanup
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_old_requests(current_time)
            self._last_cleanup = current_time
        
        # Get client's request history
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        client_requests = self.requests[client_id]
        
        # Remove requests outside the window
        cutoff_time = current_time - self.window_seconds
        client_requests[:] = [req_time for req_time in client_requests 
                             if req_time > cutoff_time]
        
        # Check if under limit
        if len(client_requests) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for client {client_id}")
            return False
        
        # Record this request
        client_requests.append(current_time)
        return True
    
    def _cleanup_old_requests(self, current_time: float):
        """Clean up old request records."""
        cutoff_time = current_time - self.window_seconds * 2  # Keep some history
        
        for client_id in list(self.requests.keys()):
            client_requests = self.requests[client_id]
            client_requests[:] = [req_time for req_time in client_requests 
                                 if req_time > cutoff_time]
            
            # Remove empty client records
            if not client_requests:
                del self.requests[client_id]


# Global security instances
_default_validator = SecurityValidator()
_default_sanitizer = InputSanitizer()
_default_rate_limiter = RateLimiter()


def validate_input(value: Any, field_name: str, validator: SecurityValidator = None) -> bool:
    """Convenience function for input validation."""
    if validator is None:
        validator = _default_validator
    
    if isinstance(value, str):
        return validator.validate_string(value, field_name)
    elif isinstance(value, (int, float)):
        return validator.validate_numeric(value, field_name)
    elif isinstance(value, dict):
        return validator.validate_parameters(value)
    else:
        return False


def sanitize_input(value: Any) -> Any:
    """Convenience function for input sanitization."""
    if isinstance(value, str):
        return _default_sanitizer.sanitize_string(value)
    elif isinstance(value, (int, float)):
        return _default_sanitizer.sanitize_numeric(value)
    elif isinstance(value, dict):
        return _default_sanitizer.sanitize_parameters(value)
    else:
        return value


def check_rate_limit(client_id: str, rate_limiter: RateLimiter = None) -> bool:
    """Convenience function for rate limiting."""
    if rate_limiter is None:
        rate_limiter = _default_rate_limiter
    return rate_limiter.is_allowed(client_id)