"""Tests for enhanced structured logging functionality."""

import pytest
import logging
import json
import time
from unittest.mock import patch, MagicMock
from io import StringIO

from src.logging_config import (
    StructuredFormatter,
    setup_logging,
    get_logger,
    log_security_event,
    log_performance_metric,
    log_api_request,
    log_model_operation,
    log_training_event,
    log_data_processing,
    log_system_event,
    log_prediction_batch,
    create_correlation_id,
    log_with_correlation
)


class TestStructuredFormatter:
    """Test cases for StructuredFormatter class."""
    
    def test_format_basic_record(self):
        """Test basic log record formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data['level'] == 'INFO'
        assert log_data['logger'] == 'test'
        assert log_data['message'] == 'Test message'
        assert 'timestamp' in log_data
    
    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Test message',
            args=(),
            exc_info=None
        )
        record.custom_field = 'custom_value'
        record.event_type = 'test_event'
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data['custom_field'] == 'custom_value'
        assert log_data['event_type'] == 'test_event'
    
    def test_format_with_source_location(self):
        """Test that source location is added for warning+ levels."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.WARNING,
            pathname='test.py',
            lineno=25,
            msg='Warning message',
            args=(),
            exc_info=None,
            func='test_function'
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert 'source' in log_data
        assert log_data['source']['file'] == 'test.py'
        assert log_data['source']['line'] == 25
        assert log_data['source']['function'] == 'test_function'
    
    def test_format_with_exception(self):
        """Test formatting with exception information."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name='test',
            level=logging.ERROR,
            pathname='test.py',
            lineno=10,
            msg='Error occurred',
            args=(),
            exc_info=exc_info
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert 'exception' in log_data
        assert 'ValueError: Test exception' in log_data['exception']


class TestSetupLogging:
    """Test cases for logging setup."""
    
    def test_setup_basic_logging(self):
        """Test basic logging setup."""
        setup_logging(level="INFO", structured=False)
        
        # Get root logger and check configuration
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0
    
    def test_setup_structured_logging(self):
        """Test structured logging setup."""
        setup_logging(level="DEBUG", structured=True)
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
        
        # Check that handler uses StructuredFormatter
        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, StructuredFormatter)
    
    def test_get_logger(self):
        """Test logger retrieval."""
        logger = get_logger(__name__)
        assert isinstance(logger, logging.Logger)
        assert logger.name == __name__


class TestSecurityLogging:
    """Test cases for security event logging."""
    
    def test_log_security_event_basic(self):
        """Test basic security event logging."""
        logger = MagicMock()
        
        log_security_event(logger, 'rate_limit', '192.168.1.1')
        
        logger.warning.assert_called_once()
        call_args = logger.warning.call_args
        assert "Security event occurred" in call_args[0][0]
        
        extra = call_args[1]['extra']
        assert extra['event_type'] == 'security'
        assert extra['security_event'] == 'rate_limit'
        assert extra['client_ip'] == '192.168.1.1'
    
    def test_log_security_event_with_details(self):
        """Test security event logging with details."""
        logger = MagicMock()
        details = {'path': '/predict', 'method': 'POST'}
        
        log_security_event(logger, 'validation_error', '10.0.0.1', details)
        
        extra = logger.warning.call_args[1]['extra']
        assert extra['path'] == '/predict'
        assert extra['method'] == 'POST'


class TestPerformanceLogging:
    """Test cases for performance metric logging."""
    
    def test_log_performance_metric_basic(self):
        """Test basic performance metric logging."""
        logger = MagicMock()
        
        log_performance_metric(logger, 'prediction', 1.5)
        
        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "Performance metric" in call_args[0][0]
        
        extra = call_args[1]['extra']
        assert extra['event_type'] == 'performance'
        assert extra['operation'] == 'prediction'
        assert extra['duration_seconds'] == 1.5
    
    def test_log_performance_metric_with_details(self):
        """Test performance metric logging with details."""
        logger = MagicMock()
        details = {'model_type': 'lstm', 'batch_size': 32}
        
        log_performance_metric(logger, 'training', 120.5, details)
        
        extra = logger.info.call_args[1]['extra']
        assert extra['model_type'] == 'lstm'
        assert extra['batch_size'] == 32


class TestAPIRequestLogging:
    """Test cases for API request logging."""
    
    def test_log_api_request_basic(self):
        """Test basic API request logging."""
        logger = MagicMock()
        
        log_api_request(logger, 'GET', '/predict', 200, 0.5)
        
        logger.info.assert_called_once()
        extra = logger.info.call_args[1]['extra']
        
        assert extra['event_type'] == 'api_request'
        assert extra['method'] == 'GET'
        assert extra['path'] == '/predict'
        assert extra['status_code'] == 200
        assert extra['duration_seconds'] == 0.5
        assert 'timestamp' in extra
    
    def test_log_api_request_with_optional_fields(self):
        """Test API request logging with optional fields."""
        logger = MagicMock()
        
        log_api_request(
            logger, 'POST', '/predict', 200, 1.2,
            client_ip='192.168.1.1',
            user_agent='Mozilla/5.0',
            request_size=512,
            response_size=1024
        )
        
        extra = logger.info.call_args[1]['extra']
        assert extra['client_ip'] == '192.168.1.1'
        assert extra['user_agent'] == 'Mozilla/5.0'
        assert extra['request_size_bytes'] == 512
        assert extra['response_size_bytes'] == 1024


class TestModelOperationLogging:
    """Test cases for model operation logging."""
    
    def test_log_model_operation_basic(self):
        """Test basic model operation logging."""
        logger = MagicMock()
        
        log_model_operation(logger, 'load', '/path/to/model.joblib')
        
        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "Model load operation" in call_args[0][0]
        
        extra = call_args[1]['extra']
        assert extra['event_type'] == 'model_operation'
        assert extra['operation'] == 'load'
        assert extra['model_path'] == '/path/to/model.joblib'
        assert 'timestamp' in extra
    
    def test_log_model_operation_with_duration(self):
        """Test model operation logging with duration and details."""
        logger = MagicMock()
        details = {'model_size': '50MB', 'model_type': 'transformer'}
        
        log_model_operation(logger, 'save', '/path/to/model.bin', 2.5, details)
        
        extra = logger.info.call_args[1]['extra']
        assert extra['duration_seconds'] == 2.5
        assert extra['model_size'] == '50MB'
        assert extra['model_type'] == 'transformer'


class TestTrainingEventLogging:
    """Test cases for training event logging."""
    
    def test_log_training_event_basic(self):
        """Test basic training event logging."""
        logger = MagicMock()
        
        log_training_event(logger, 'start')
        
        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "Training start" in call_args[0][0]
        
        extra = call_args[1]['extra']
        assert extra['event_type'] == 'training'
        assert extra['training_event'] == 'start'
    
    def test_log_training_event_with_metrics(self):
        """Test training event logging with metrics."""
        logger = MagicMock()
        metrics = {'loss': 0.35, 'accuracy': 0.92}
        
        log_training_event(logger, 'epoch_complete', epoch=5, metrics=metrics)
        
        extra = logger.info.call_args[1]['extra']
        assert extra['epoch'] == 5
        assert extra['metrics'] == metrics


class TestDataProcessingLogging:
    """Test cases for data processing logging."""
    
    def test_log_data_processing_basic(self):
        """Test basic data processing logging."""
        logger = MagicMock()
        
        log_data_processing(logger, 'load', records_processed=1000)
        
        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "Data processing load" in call_args[0][0]
        
        extra = call_args[1]['extra']
        assert extra['event_type'] == 'data_processing'
        assert extra['operation'] == 'load'
        assert extra['records_processed'] == 1000
    
    def test_log_data_processing_with_files(self):
        """Test data processing logging with file paths."""
        logger = MagicMock()
        
        log_data_processing(
            logger, 'preprocess', 
            records_processed=800,
            duration=5.2,
            input_file='input.csv',
            output_file='output.csv'
        )
        
        extra = logger.info.call_args[1]['extra']
        assert extra['duration_seconds'] == 5.2
        assert extra['input_file'] == 'input.csv'
        assert extra['output_file'] == 'output.csv'


class TestSystemEventLogging:
    """Test cases for system event logging."""
    
    def test_log_system_event_basic(self):
        """Test basic system event logging."""
        logger = MagicMock()
        
        log_system_event(logger, 'startup', 'webapp')
        
        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "System startup in webapp" in call_args[0][0]
        
        extra = call_args[1]['extra']
        assert extra['event_type'] == 'system'
        assert extra['system_event'] == 'startup'
        assert extra['component'] == 'webapp'
    
    def test_log_system_event_with_details(self):
        """Test system event logging with details."""
        logger = MagicMock()
        details = {'port': 5000, 'host': '0.0.0.0'}
        
        log_system_event(logger, 'config_reload', 'server', details)
        
        extra = logger.info.call_args[1]['extra']
        assert extra['port'] == 5000
        assert extra['host'] == '0.0.0.0'


class TestPredictionBatchLogging:
    """Test cases for prediction batch logging."""
    
    def test_log_prediction_batch_basic(self):
        """Test basic prediction batch logging."""
        logger = MagicMock()
        
        log_prediction_batch(logger, 50, 10.5, 125.5)
        
        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "Batch prediction completed" in call_args[0][0]
        
        extra = call_args[1]['extra']
        assert extra['event_type'] == 'prediction_batch'
        assert extra['batch_size'] == 50
        assert extra['total_duration_seconds'] == 10.5
        assert extra['avg_duration_per_prediction'] == 10.5 / 50
        assert extra['avg_text_length'] == 125.5
    
    def test_log_prediction_batch_with_model_type(self):
        """Test prediction batch logging with model type."""
        logger = MagicMock()
        details = {'accuracy': 0.95}
        
        log_prediction_batch(logger, 25, 3.2, 100.0, 'transformer', details)
        
        extra = logger.info.call_args[1]['extra']
        assert extra['model_type'] == 'transformer'
        assert extra['accuracy'] == 0.95
    
    def test_log_prediction_batch_zero_batch_size(self):
        """Test prediction batch logging with zero batch size."""
        logger = MagicMock()
        
        log_prediction_batch(logger, 0, 0.0, 0.0)
        
        extra = logger.info.call_args[1]['extra']
        assert extra['avg_duration_per_prediction'] == 0


class TestCorrelationIdLogging:
    """Test cases for correlation ID functionality."""
    
    def test_create_correlation_id(self):
        """Test correlation ID creation."""
        correlation_id = create_correlation_id()
        
        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0
        
        # Should be unique
        another_id = create_correlation_id()
        assert correlation_id != another_id
    
    def test_log_with_correlation_provided_id(self):
        """Test logging with provided correlation ID."""
        logger = MagicMock()
        correlation_id = 'test-correlation-123'
        extra_data = {'component': 'test'}
        
        log_with_correlation(logger, logging.INFO, 'Test message', correlation_id, extra_data)
        
        logger.log.assert_called_once()
        call_args = logger.log.call_args
        assert call_args[0][0] == logging.INFO
        assert call_args[0][1] == 'Test message'
        
        extra = call_args[1]['extra']
        assert extra['correlation_id'] == correlation_id
        assert extra['component'] == 'test'
        assert 'timestamp' in extra
    
    def test_log_with_correlation_generated_id(self):
        """Test logging with auto-generated correlation ID."""
        logger = MagicMock()
        
        log_with_correlation(logger, logging.ERROR, 'Error message')
        
        extra = logger.log.call_args[1]['extra']
        assert 'correlation_id' in extra
        assert len(extra['correlation_id']) > 0


class TestIntegrationLogging:
    """Integration tests for structured logging."""
    
    def test_end_to_end_structured_logging(self):
        """Test complete structured logging flow."""
        # Capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(StructuredFormatter())
        
        logger = logging.getLogger('test_integration')
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        
        # Log various events
        log_api_request(logger, 'POST', '/predict', 200, 1.5, '127.0.0.1')
        log_model_operation(logger, 'load', '/path/to/model.pkl', 2.0)
        log_training_event(logger, 'complete', metrics={'accuracy': 0.95})
        
        # Parse all log entries
        log_output = log_stream.getvalue()
        log_lines = [line for line in log_output.strip().split('\n') if line]
        
        assert len(log_lines) == 3
        
        # Parse and validate each log entry
        for line in log_lines:
            log_data = json.loads(line)
            assert 'timestamp' in log_data
            assert 'level' in log_data
            assert 'event_type' in log_data
    
    def test_logging_performance(self):
        """Test that structured logging doesn't significantly impact performance."""
        logger = MagicMock()
        
        start_time = time.time()
        for i in range(1000):
            log_api_request(logger, 'GET', f'/test/{i}', 200, 0.1)
        end_time = time.time()
        
        # Should complete 1000 log calls in reasonable time (< 1 second)
        duration = end_time - start_time
        assert duration < 1.0
        assert logger.info.call_count == 1000