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
                   status_code: int, duration: float, client_ip: str = None,
                   user_agent: str = None, request_size: int = None,
                   response_size: int = None) -> None:
    """Log API requests with structured data.
    
    Args:
        logger: Logger instance
        method: HTTP method
        path: Request path
        status_code: HTTP response status code
        duration: Request duration in seconds
        client_ip: Client IP address if available
        user_agent: User agent string if available
        request_size: Request size in bytes if available
        response_size: Response size in bytes if available
    """
    log_data = {
        'event_type': 'api_request',
        'method': method,
        'path': path,
        'status_code': status_code,
        'duration_seconds': duration,
        'timestamp': time.time()
    }
    
    if client_ip:
        log_data['client_ip'] = client_ip
    
    if user_agent:
        log_data['user_agent'] = user_agent
    
    if request_size is not None:
        log_data['request_size_bytes'] = request_size
    
    if response_size is not None:
        log_data['response_size_bytes'] = response_size
    
    logger.info("API request completed", extra=log_data)


def log_model_operation(logger: logging.Logger, operation: str, model_path: str,
                       duration: float = None, details: Dict[str, Any] = None) -> None:
    """Log model operations with structured data.
    
    Args:
        logger: Logger instance
        operation: Type of operation (load, train, predict, save)
        model_path: Path to the model file
        duration: Operation duration in seconds if available
        details: Additional operation details
    """
    log_data = {
        'event_type': 'model_operation',
        'operation': operation,
        'model_path': model_path,
        'timestamp': time.time()
    }
    
    if duration is not None:
        log_data['duration_seconds'] = duration
    
    if details:
        log_data.update(details)
    
    logger.info(f"Model {operation} operation", extra=log_data)


def log_training_event(logger: logging.Logger, event: str, epoch: int = None,
                      metrics: Dict[str, float] = None, details: Dict[str, Any] = None) -> None:
    """Log training events with structured data.
    
    Args:
        logger: Logger instance
        event: Training event type (start, epoch_complete, validation, complete)
        epoch: Current epoch number if applicable
        metrics: Training metrics (loss, accuracy, etc.)
        details: Additional training details
    """
    log_data = {
        'event_type': 'training',
        'training_event': event,
        'timestamp': time.time()
    }
    
    if epoch is not None:
        log_data['epoch'] = epoch
    
    if metrics:
        log_data['metrics'] = metrics
    
    if details:
        log_data.update(details)
    
    logger.info(f"Training {event}", extra=log_data)


def log_data_processing(logger: logging.Logger, operation: str, 
                       records_processed: int = None, duration: float = None,
                       input_file: str = None, output_file: str = None,
                       details: Dict[str, Any] = None) -> None:
    """Log data processing operations with structured data.
    
    Args:
        logger: Logger instance
        operation: Type of operation (load, preprocess, save, validate)
        records_processed: Number of records processed
        duration: Processing duration in seconds
        input_file: Input file path if applicable
        output_file: Output file path if applicable
        details: Additional processing details
    """
    log_data = {
        'event_type': 'data_processing',
        'operation': operation,
        'timestamp': time.time()
    }
    
    if records_processed is not None:
        log_data['records_processed'] = records_processed
    
    if duration is not None:
        log_data['duration_seconds'] = duration
    
    if input_file:
        log_data['input_file'] = input_file
    
    if output_file:
        log_data['output_file'] = output_file
    
    if details:
        log_data.update(details)
    
    logger.info(f"Data processing {operation}", extra=log_data)


def log_system_event(logger: logging.Logger, event: str, component: str,
                    details: Dict[str, Any] = None) -> None:
    """Log system events with structured data.
    
    Args:
        logger: Logger instance
        event: System event type (startup, shutdown, config_reload, etc.)
        component: Component generating the event
        details: Additional event details
    """
    log_data = {
        'event_type': 'system',
        'system_event': event,
        'component': component,
        'timestamp': time.time()
    }
    
    if details:
        log_data.update(details)
    
    logger.info(f"System {event} in {component}", extra=log_data)


def log_prediction_batch(logger: logging.Logger, batch_size: int, 
                        total_duration: float, avg_text_length: float,
                        model_type: str = None, details: Dict[str, Any] = None) -> None:
    """Log batch prediction operations with structured data.
    
    Args:
        logger: Logger instance
        batch_size: Number of predictions in batch
        total_duration: Total processing time for batch
        avg_text_length: Average text length in batch
        model_type: Type of model used
        details: Additional batch details
    """
    log_data = {
        'event_type': 'prediction_batch',
        'batch_size': batch_size,
        'total_duration_seconds': total_duration,
        'avg_duration_per_prediction': total_duration / batch_size if batch_size > 0 else 0,
        'avg_text_length': avg_text_length,
        'timestamp': time.time()
    }
    
    if model_type:
        log_data['model_type'] = model_type
    
    if details:
        log_data.update(details)
    
    logger.info("Batch prediction completed", extra=log_data)


def create_correlation_id() -> str:
    """Create a unique correlation ID for request tracing.
    
    Returns:
        Unique correlation ID string
    """
    import uuid
    return str(uuid.uuid4())


def log_with_correlation(logger: logging.Logger, level: int, message: str,
                        correlation_id: str = None, extra_data: Dict[str, Any] = None) -> None:
    """Log with correlation ID for request tracing.
    
    Args:
        logger: Logger instance
        level: Logging level (logging.INFO, logging.ERROR, etc.)
        message: Log message
        correlation_id: Correlation ID for request tracking
        extra_data: Additional structured data
    """
    log_data = {
        'correlation_id': correlation_id or create_correlation_id(),
        'timestamp': time.time()
    }
    
    if extra_data:
        log_data.update(extra_data)
    
    logger.log(level, message, extra=log_data)