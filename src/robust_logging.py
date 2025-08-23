
import logging
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class StructuredLogger:
    """Structured logging for production readiness."""
    
    def __init__(self, name: str = "sentiment_analyzer", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        json_formatter = JsonFormatter()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "application.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(json_formatter)
        
        # Error file handler
        error_handler = logging.FileHandler(log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(json_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def info(self, message: str, **kwargs):
        """Log info with structured data."""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.info(message, extra=extra)
    
    def error(self, message: str, **kwargs):
        """Log error with structured data."""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.error(message, extra=extra)
    
    def warning(self, message: str, **kwargs):
        """Log warning with structured data."""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.warning(message, extra=extra)

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'structured_data'):
            log_entry.update(record.structured_data)
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

# Global logger instance
structured_logger = StructuredLogger()
