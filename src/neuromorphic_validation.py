"""
🛡️ Neuromorphic Validation & Security
=====================================

Comprehensive validation, error handling, and security measures for 
neuromorphic spikeformer processing.

Generation 2: MAKE IT ROBUST - Reliable neuromorphic computation
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
import time

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for neuromorphic validation."""
    
    # Input validation
    max_input_dim: int = 10000
    min_input_dim: int = 1
    max_batch_size: int = 1000
    max_sequence_length: int = 5000
    
    # Spike validation
    max_spike_rate: float = 1000.0  # Hz
    min_membrane_threshold: float = 0.1
    max_membrane_threshold: float = 10.0
    max_timesteps: int = 10000
    
    # Security settings
    enable_input_sanitization: bool = True
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    enable_anomaly_detection: bool = True
    
    # Performance limits
    max_processing_time: float = 300.0  # seconds
    max_memory_usage: int = 8 * 1024 * 1024 * 1024  # 8GB


class NeuromorphicValidationError(Exception):
    """Base exception for neuromorphic validation errors."""
    pass


class InputValidationError(NeuromorphicValidationError):
    """Exception raised for invalid input data."""
    pass


class SpikingValidationError(NeuromorphicValidationError):
    """Exception raised for invalid spiking parameters."""
    pass


class SecurityValidationError(NeuromorphicValidationError):
    """Exception raised for security violations."""
    pass


class PerformanceValidationError(NeuromorphicValidationError):
    """Exception raised for performance limit violations."""
    pass


class InputValidator:
    """Validates input data for neuromorphic processing."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.suspicious_patterns = [
            r"<script.*?>.*?</script>",  # XSS patterns
            r"javascript:",
            r"eval\(",
            r"exec\(",
            r"import\s+os",
            r"__import__",
            r"\.\.\/",  # Path traversal
            r"[;&|`]",  # Command injection
        ]
        
    def validate_input_features(self, features: Union[np.ndarray, torch.Tensor]) -> bool:
        """
        Validate input features for neuromorphic processing.
        
        Args:
            features: Input feature array/tensor
            
        Returns:
            True if valid
            
        Raises:
            InputValidationError: If validation fails
        """
        try:
            # Convert to numpy for validation
            if isinstance(features, torch.Tensor):
                features_np = features.detach().cpu().numpy()
            else:
                features_np = np.asarray(features)
            
            # Check dimensions
            if len(features_np.shape) < 2 or len(features_np.shape) > 3:
                raise InputValidationError(
                    f"Invalid input dimensions: {features_np.shape}. Expected 2D or 3D array."
                )
            
            # Check batch size
            batch_size = features_np.shape[0]
            if batch_size > self.config.max_batch_size:
                raise InputValidationError(
                    f"Batch size {batch_size} exceeds maximum {self.config.max_batch_size}"
                )
            
            # Check feature dimension
            feature_dim = features_np.shape[-1]
            if feature_dim < self.config.min_input_dim or feature_dim > self.config.max_input_dim:
                raise InputValidationError(
                    f"Feature dimension {feature_dim} outside valid range "
                    f"[{self.config.min_input_dim}, {self.config.max_input_dim}]"
                )
            
            # Check for sequence length if 3D
            if len(features_np.shape) == 3:
                seq_len = features_np.shape[1]
                if seq_len > self.config.max_sequence_length:
                    raise InputValidationError(
                        f"Sequence length {seq_len} exceeds maximum {self.config.max_sequence_length}"
                    )
            
            # Check for NaN or infinite values
            if np.any(np.isnan(features_np)) or np.any(np.isinf(features_np)):
                raise InputValidationError("Input contains NaN or infinite values")
            
            # Check for reasonable value ranges
            feature_max = np.max(np.abs(features_np))
            if feature_max > 1000.0:
                logger.warning(f"Large feature values detected (max: {feature_max})")
            
            logger.debug(f"Input validation passed: shape={features_np.shape}")
            return True
            
        except Exception as e:
            if isinstance(e, InputValidationError):
                raise
            else:
                raise InputValidationError(f"Validation error: {str(e)}")
    
    def validate_configuration(self, config_dict: Dict[str, Any]) -> bool:
        """
        Validate neuromorphic configuration parameters.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            True if valid
            
        Raises:
            SpikingValidationError: If validation fails
        """
        try:
            # Validate membrane threshold
            if 'membrane_threshold' in config_dict:
                threshold = config_dict['membrane_threshold']
                if not isinstance(threshold, (int, float)):
                    raise SpikingValidationError("membrane_threshold must be numeric")
                if threshold < self.config.min_membrane_threshold or threshold > self.config.max_membrane_threshold:
                    raise SpikingValidationError(
                        f"membrane_threshold {threshold} outside valid range "
                        f"[{self.config.min_membrane_threshold}, {self.config.max_membrane_threshold}]"
                    )
            
            # Validate timesteps
            if 'timesteps' in config_dict:
                timesteps = config_dict['timesteps']
                if not isinstance(timesteps, int) or timesteps <= 0:
                    raise SpikingValidationError("timesteps must be positive integer")
                if timesteps > self.config.max_timesteps:
                    raise SpikingValidationError(
                        f"timesteps {timesteps} exceeds maximum {self.config.max_timesteps}"
                    )
            
            # Validate spike rate
            if 'spike_rate_max' in config_dict:
                spike_rate = config_dict['spike_rate_max']
                if not isinstance(spike_rate, (int, float)) or spike_rate <= 0:
                    raise SpikingValidationError("spike_rate_max must be positive numeric")
                if spike_rate > self.config.max_spike_rate:
                    raise SpikingValidationError(
                        f"spike_rate_max {spike_rate} exceeds maximum {self.config.max_spike_rate}"
                    )
            
            # Validate membrane decay
            if 'membrane_decay' in config_dict:
                decay = config_dict['membrane_decay']
                if not isinstance(decay, (int, float)):
                    raise SpikingValidationError("membrane_decay must be numeric")
                if decay < 0.0 or decay > 1.0:
                    raise SpikingValidationError("membrane_decay must be in range [0, 1]")
            
            logger.debug("Configuration validation passed")
            return True
            
        except Exception as e:
            if isinstance(e, SpikingValidationError):
                raise
            else:
                raise SpikingValidationError(f"Configuration validation error: {str(e)}")
    
    def sanitize_text_input(self, text: str) -> str:
        """
        Sanitize text input to prevent injection attacks.
        
        Args:
            text: Input text string
            
        Returns:
            Sanitized text
            
        Raises:
            SecurityValidationError: If malicious patterns detected
        """
        if not self.config.enable_input_sanitization:
            return text
        
        try:
            # Check for suspicious patterns
            for pattern in self.suspicious_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    raise SecurityValidationError(f"Suspicious pattern detected: {pattern}")
            
            # Remove potentially dangerous characters
            sanitized = re.sub(r'[<>"\']', '', text)
            
            # Limit length
            if len(sanitized) > 10000:
                sanitized = sanitized[:10000]
                logger.warning("Text input truncated due to length")
            
            return sanitized
            
        except Exception as e:
            if isinstance(e, SecurityValidationError):
                raise
            else:
                raise SecurityValidationError(f"Text sanitization error: {str(e)}")


class PerformanceMonitor:
    """Monitors performance and resource usage during neuromorphic processing."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.start_time = None
        self.peak_memory = 0
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.peak_memory = 0
        logger.debug("Performance monitoring started")
    
    def check_performance_limits(self) -> bool:
        """
        Check if performance limits are being exceeded.
        
        Returns:
            True if within limits
            
        Raises:
            PerformanceValidationError: If limits exceeded
        """
        if self.start_time is None:
            return True
        
        # Check processing time
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.config.max_processing_time:
            raise PerformanceValidationError(
                f"Processing time {elapsed_time:.1f}s exceeds limit {self.config.max_processing_time}s"
            )
        
        # Check memory usage (simplified check)
        try:
            import psutil
            memory_usage = psutil.virtual_memory().used
            if memory_usage > self.config.max_memory_usage:
                raise PerformanceValidationError(
                    f"Memory usage {memory_usage/1e9:.1f}GB exceeds limit {self.config.max_memory_usage/1e9:.1f}GB"
                )
        except ImportError:
            logger.debug("psutil not available for memory monitoring")
        
        return True
    
    def stop_monitoring(self) -> Dict[str, float]:
        """
        Stop monitoring and return performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        if self.start_time is None:
            return {}
        
        elapsed_time = time.time() - self.start_time
        
        metrics = {
            'processing_time': elapsed_time,
            'peak_memory_mb': self.peak_memory / (1024 * 1024)
        }
        
        logger.debug(f"Performance monitoring stopped: {metrics}")
        return metrics


class RateLimiter:
    """Rate limiting for neuromorphic processing requests."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.request_times = []
        
    def check_rate_limit(self, client_id: str = "default") -> bool:
        """
        Check if request is within rate limits.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if within limits
            
        Raises:
            SecurityValidationError: If rate limit exceeded
        """
        if not self.config.enable_rate_limiting:
            return True
        
        current_time = time.time()
        
        # Remove old requests (older than 1 minute)
        cutoff_time = current_time - 60.0
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # Check current rate
        if len(self.request_times) >= self.config.max_requests_per_minute:
            raise SecurityValidationError(
                f"Rate limit exceeded: {len(self.request_times)} requests in last minute"
            )
        
        # Add current request
        self.request_times.append(current_time)
        
        logger.debug(f"Rate limit check passed: {len(self.request_times)} requests in window")
        return True


class AnomalyDetector:
    """Detects anomalous patterns in neuromorphic processing."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.baseline_stats = {}
        self.request_count = 0
        
    def update_baseline(self, features: np.ndarray, spike_stats: Dict[str, float]):
        """Update baseline statistics for anomaly detection."""
        if not self.config.enable_anomaly_detection:
            return
        
        # Update feature statistics
        self.baseline_stats.update({
            'mean_feature_magnitude': np.mean(np.abs(features)),
            'std_feature_magnitude': np.std(features),
            'mean_spike_rate': spike_stats.get('average_spike_rate', 0.0),
            'mean_energy': spike_stats.get('energy_consumption', 0.0)
        })
        
        self.request_count += 1
        logger.debug(f"Baseline statistics updated (n={self.request_count})")
    
    def detect_anomaly(self, features: np.ndarray, spike_stats: Dict[str, float]) -> bool:
        """
        Detect if current request is anomalous.
        
        Args:
            features: Input features
            spike_stats: Spike processing statistics
            
        Returns:
            True if anomaly detected
        """
        if not self.config.enable_anomaly_detection or self.request_count < 10:
            return False
        
        try:
            # Check feature magnitude anomalies
            current_magnitude = np.mean(np.abs(features))
            baseline_magnitude = self.baseline_stats.get('mean_feature_magnitude', 0.0)
            magnitude_std = self.baseline_stats.get('std_feature_magnitude', 1.0)
            
            if abs(current_magnitude - baseline_magnitude) > 3 * magnitude_std:
                logger.warning(f"Feature magnitude anomaly detected: {current_magnitude} vs {baseline_magnitude}")
                return True
            
            # Check spike rate anomalies
            current_spike_rate = spike_stats.get('average_spike_rate', 0.0)
            baseline_spike_rate = self.baseline_stats.get('mean_spike_rate', 0.0)
            
            if baseline_spike_rate > 0 and abs(current_spike_rate - baseline_spike_rate) > 2 * baseline_spike_rate:
                logger.warning(f"Spike rate anomaly detected: {current_spike_rate} vs {baseline_spike_rate}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return False


class NeuromorphicValidator:
    """
    Comprehensive validator for neuromorphic spikeformer processing.
    
    Provides input validation, security checks, performance monitoring,
    and anomaly detection for robust neuromorphic computation.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        
        # Initialize components
        self.input_validator = InputValidator(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        
        logger.info("NeuromorphicValidator initialized with comprehensive validation")
    
    def validate_processing_request(
        self, 
        features: Union[np.ndarray, torch.Tensor],
        config_dict: Optional[Dict[str, Any]] = None,
        client_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of neuromorphic processing request.
        
        Args:
            features: Input features
            config_dict: Optional configuration
            client_id: Client identifier
            
        Returns:
            Validation results dictionary
            
        Raises:
            NeuromorphicValidationError: If validation fails
        """
        try:
            # Start performance monitoring
            self.performance_monitor.start_monitoring()
            
            # Rate limiting
            self.rate_limiter.check_rate_limit(client_id)
            
            # Input validation
            self.input_validator.validate_input_features(features)
            
            # Configuration validation
            if config_dict:
                self.input_validator.validate_configuration(config_dict)
            
            # Convert features for anomaly detection
            if isinstance(features, torch.Tensor):
                features_np = features.detach().cpu().numpy()
            else:
                features_np = np.asarray(features)
            
            validation_results = {
                'status': 'valid',
                'input_shape': features_np.shape,
                'client_id': client_id,
                'timestamp': time.time(),
                'validation_config': {
                    'input_sanitization': self.config.enable_input_sanitization,
                    'rate_limiting': self.config.enable_rate_limiting,
                    'anomaly_detection': self.config.enable_anomaly_detection
                }
            }
            
            logger.debug(f"Validation passed for client {client_id}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def validate_processing_results(
        self, 
        spike_stats: Dict[str, float],
        features: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Validate processing results and update monitoring.
        
        Args:
            spike_stats: Spike processing statistics
            features: Original input features
            
        Returns:
            Validation results
        """
        try:
            # Performance limits check
            self.performance_monitor.check_performance_limits()
            
            # Convert features
            if isinstance(features, torch.Tensor):
                features_np = features.detach().cpu().numpy()
            else:
                features_np = np.asarray(features)
            
            # Anomaly detection
            anomaly_detected = self.anomaly_detector.detect_anomaly(features_np, spike_stats)
            
            # Update baseline
            self.anomaly_detector.update_baseline(features_np, spike_stats)
            
            # Stop performance monitoring
            perf_metrics = self.performance_monitor.stop_monitoring()
            
            results = {
                'status': 'completed',
                'anomaly_detected': anomaly_detected,
                'performance_metrics': perf_metrics,
                'spike_statistics': spike_stats
            }
            
            if anomaly_detected:
                logger.warning("Anomaly detected in processing results")
            
            return results
            
        except Exception as e:
            logger.error(f"Result validation failed: {e}")
            raise


def create_secure_neuromorphic_validator(
    max_batch_size: int = 100,
    enable_rate_limiting: bool = True,
    max_requests_per_minute: int = 60
) -> NeuromorphicValidator:
    """
    Create a neuromorphic validator with security-focused configuration.
    
    Args:
        max_batch_size: Maximum batch size allowed
        enable_rate_limiting: Enable rate limiting
        max_requests_per_minute: Rate limit threshold
        
    Returns:
        Configured NeuromorphicValidator
    """
    config = ValidationConfig(
        max_batch_size=max_batch_size,
        enable_rate_limiting=enable_rate_limiting,
        max_requests_per_minute=max_requests_per_minute,
        enable_input_sanitization=True,
        enable_anomaly_detection=True
    )
    
    validator = NeuromorphicValidator(config)
    logger.info("Created secure neuromorphic validator")
    
    return validator


# Validation decorators for neuromorphic functions
def validate_neuromorphic_input(validator: Optional[NeuromorphicValidator] = None):
    """Decorator to validate neuromorphic function inputs."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if validator is None:
                return func(*args, **kwargs)
            
            # Extract features from function arguments (assumes first arg is features)
            if len(args) > 0:
                features = args[0]
                try:
                    validator.validate_processing_request(features)
                except NeuromorphicValidationError as e:
                    logger.error(f"Input validation failed: {e}")
                    raise
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def monitor_neuromorphic_performance(validator: Optional[NeuromorphicValidator] = None):
    """Decorator to monitor neuromorphic function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if validator is None:
                return func(*args, **kwargs)
            
            validator.performance_monitor.start_monitoring()
            try:
                result = func(*args, **kwargs)
                validator.performance_monitor.stop_monitoring()
                return result
            except Exception as e:
                validator.performance_monitor.stop_monitoring()
                raise
        return wrapper
    return decorator