"""Tests for structured logging functionality."""

import json
import logging
import io
import sys
from unittest.mock import patch, MagicMock

import pytest

from src.logging_config import (
    StructuredFormatter, 
    setup_logging, 
    get_logger,
    log_security_event,
    log_performance_metric,
    log_api_request
)


class TestStructuredFormatter:
    """Test structured JSON logging formatter."""
    
    def test_formats_basic_log_record(self):
        """Test basic log record formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data['level'] == 'INFO'
        assert log_data['logger'] == 'test_logger'
        assert log_data['message'] == 'Test message'
        assert 'timestamp' in log_data
    
    def test_includes_source_for_warnings(self):
        """Test that source location is included for warnings and errors."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="/test/path.py",
            lineno=42,
            msg="Warning message",
            args=(),
            exc_info=None,
            func="test_function"
        )
        record.filename = "path.py"
        record.funcName = "test_function"
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert 'source' in log_data
        assert log_data['source']['file'] == 'path.py'
        assert log_data['source']['line'] == 42
        assert log_data['source']['function'] == 'test_function'
    
    def test_includes_extra_fields(self):
        """Test that extra fields are included in log output."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.event_type = "api_request"
        record.client_ip = "192.168.1.1"
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data['event_type'] == 'api_request'
        assert log_data['client_ip'] == '192.168.1.1'


class TestLoggingSetup:
    """Test logging configuration setup."""
    
    def test_setup_structured_logging(self):
        """Test that structured logging is configured correctly."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            setup_logging(level="INFO", structured=True)
            logger = get_logger("test")
            logger.info("Test message")
            
            output = mock_stdout.getvalue()
            log_data = json.loads(output.strip())
            
            assert log_data['level'] == 'INFO'
            assert log_data['message'] == 'Test message'
    
    def test_setup_standard_logging(self):
        """Test that standard logging is configured correctly."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            setup_logging(level="INFO", structured=False)
            logger = get_logger("test")
            logger.info("Test message")
            
            output = mock_stdout.getvalue()
            assert "Test message" in output
            assert "INFO" in output
            # Should not be JSON
            with pytest.raises(json.JSONDecodeError):
                json.loads(output.strip())


class TestSecurityLogging:
    """Test security event logging."""
    
    def test_log_security_event_basic(self):
        """Test basic security event logging."""
        logger = MagicMock()
        
        log_security_event(logger, "rate_limit", client_ip="192.168.1.1")
        
        logger.warning.assert_called_once()
        args, kwargs = logger.warning.call_args
        
        assert args[0] == "Security event occurred"
        assert kwargs['extra']['event_type'] == 'security'
        assert kwargs['extra']['security_event'] == 'rate_limit'
        assert kwargs['extra']['client_ip'] == '192.168.1.1'
    
    def test_log_security_event_with_details(self):
        """Test security event logging with additional details."""
        logger = MagicMock()
        details = {"path": "/predict", "method": "POST"}
        
        log_security_event(logger, "validation_error", details=details)
        
        logger.warning.assert_called_once()
        args, kwargs = logger.warning.call_args
        
        assert kwargs['extra']['path'] == '/predict'
        assert kwargs['extra']['method'] == 'POST'


class TestPerformanceLogging:
    """Test performance metric logging."""
    
    def test_log_performance_metric(self):
        """Test performance metric logging."""
        logger = MagicMock()
        
        log_performance_metric(logger, "prediction", 0.123, {"model": "nb"})
        
        logger.info.assert_called_once()
        args, kwargs = logger.info.call_args
        
        assert args[0] == "Performance metric"
        assert kwargs['extra']['event_type'] == 'performance'
        assert kwargs['extra']['operation'] == 'prediction'
        assert kwargs['extra']['duration_seconds'] == 0.123
        assert kwargs['extra']['model'] == 'nb'


class TestAPILogging:
    """Test API request logging."""
    
    def test_log_api_request(self):
        """Test API request logging."""
        logger = MagicMock()
        
        log_api_request(logger, "POST", "/predict", 200, 0.456, "127.0.0.1")
        
        logger.info.assert_called_once()
        args, kwargs = logger.info.call_args
        
        assert args[0] == "API request completed"
        assert kwargs['extra']['event_type'] == 'api_request'
        assert kwargs['extra']['method'] == 'POST'
        assert kwargs['extra']['path'] == '/predict'
        assert kwargs['extra']['status_code'] == 200
        assert kwargs['extra']['duration_seconds'] == 0.456
        assert kwargs['extra']['client_ip'] == '127.0.0.1'
    
    def test_log_api_request_without_ip(self):
        """Test API request logging without client IP."""
        logger = MagicMock()
        
        log_api_request(logger, "GET", "/", 200, 0.123)
        
        logger.info.assert_called_once()
        args, kwargs = logger.info.call_args
        
        assert 'client_ip' not in kwargs['extra']


class TestIntegration:
    """Test integration with actual logging system."""
    
    def test_end_to_end_structured_logging(self):
        """Test complete structured logging workflow."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            setup_logging(level="INFO", structured=True)
            logger = get_logger("test.integration")
            
            # Log various event types
            log_security_event(logger, "test_event", "192.168.1.1", {"test": "data"})
            log_performance_metric(logger, "test_op", 0.123, {"size": 100})
            log_api_request(logger, "GET", "/test", 200, 0.456, "127.0.0.1")
            
            output = mock_stdout.getvalue()
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            
            # Parse each log line
            log_entries = []
            for line in lines:
                try:
                    log_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip lines that aren't valid JSON (shouldn't happen with structured logging)
                    continue
            
            # Verify security event
            security_log = log_entries[0]
            assert security_log['security_event'] == 'test_event'
            assert security_log['client_ip'] == '192.168.1.1'
            assert security_log['test'] == 'data'
            
            # Verify performance metric
            perf_log = log_entries[1]
            assert perf_log['operation'] == 'test_op'
            assert perf_log['duration_seconds'] == 0.123
            assert perf_log['size'] == 100
            
            # Verify API request
            api_log = log_entries[2]
            assert api_log['method'] == 'GET'
            assert api_log['path'] == '/test'
            assert api_log['status_code'] == 200