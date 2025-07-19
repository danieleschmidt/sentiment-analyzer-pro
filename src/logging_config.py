"""Structured logging configuration for Sentiment Analyzer Pro."""

import json
import logging
import time
from typing import Any, Dict
import sys


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry: Dict[str, Any] = {
            'timestamp': time.time(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'pathname', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'module', 'filename', 'levelno', 'levelname',
                          'message', 'exc_info', 'exc_text', 'stack_info'):
                log_entry[key] = value
        
        # Add source location for debug/error levels
        if record.levelno >= logging.WARNING:
            log_entry['source'] = {
                'file': record.filename,
                'line': record.lineno,
                'function': record.funcName
            }
        
        return json.dumps(log_entry, default=str)


def setup_logging(level: str = "INFO", structured: bool = False) -> None:
    """Configure application logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        structured: Whether to use structured JSON logging
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure handler
    handler = logging.StreamHandler(sys.stdout)
    
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)
    
    # Set library loggers to WARNING to reduce noise
    for lib in ['urllib3', 'requests', 'transformers', 'tensorflow']:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_security_event(logger: logging.Logger, event_type: str, 
                      client_ip: str = None, details: Dict[str, Any] = None) -> None:
    """Log security-related events with structured data.
    
    Args:
        logger: Logger instance
        event_type: Type of security event (rate_limit, validation_error, etc.)
        client_ip: Client IP address if available
        details: Additional event details
    """
    log_data = {
        'event_type': 'security',
        'security_event': event_type,
    }
    
    if client_ip:
        log_data['client_ip'] = client_ip
    
    if details:
        log_data.update(details)
    
    logger.warning("Security event occurred", extra=log_data)


def log_performance_metric(logger: logging.Logger, operation: str,
                          duration: float, details: Dict[str, Any] = None) -> None:
    """Log performance metrics with structured data.
    
    Args:
        logger: Logger instance
        operation: Operation being measured
        duration: Duration in seconds
        details: Additional metric details
    """
    log_data = {
        'event_type': 'performance',
        'operation': operation,
        'duration_seconds': duration,
    }
    
    if details:
        log_data.update(details)
    
    logger.info("Performance metric", extra=log_data)


def log_api_request(logger: logging.Logger, method: str, path: str,
                   status_code: int, duration: float, client_ip: str = None) -> None:
    """Log API requests with structured data.
    
    Args:
        logger: Logger instance
        method: HTTP method
        path: Request path
        status_code: HTTP response status code
        duration: Request duration in seconds
        client_ip: Client IP address if available
    """
    log_data = {
        'event_type': 'api_request',
        'method': method,
        'path': path,
        'status_code': status_code,
        'duration_seconds': duration,
    }
    
    if client_ip:
        log_data['client_ip'] = client_ip
    
    logger.info("API request completed", extra=log_data)