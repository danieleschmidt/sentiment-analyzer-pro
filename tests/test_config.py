"""Tests for configuration management."""

import os
import pytest
from unittest.mock import patch

from src.config import get_env_int, get_env_float, get_env_bool, get_env_str, Config


class TestEnvironmentVariableHelpers:
    """Test helper functions for environment variable parsing."""
    
    def test_get_env_int_with_default(self):
        """Test get_env_int returns default when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_env_int("TEST_INT", 42) == 42
    
    def test_get_env_int_with_valid_value(self):
        """Test get_env_int parses valid integer."""
        with patch.dict(os.environ, {"TEST_INT": "123"}):
            assert get_env_int("TEST_INT", 0) == 123
    
    def test_get_env_int_with_invalid_value(self):
        """Test get_env_int raises ValueError for invalid integer."""
        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            with pytest.raises(ValueError, match="must be an integer"):
                get_env_int("TEST_INT", 0)
    
    def test_get_env_float_with_default(self):
        """Test get_env_float returns default when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_env_float("TEST_FLOAT", 3.14) == 3.14
    
    def test_get_env_float_with_valid_value(self):
        """Test get_env_float parses valid float."""
        with patch.dict(os.environ, {"TEST_FLOAT": "2.718"}):
            assert get_env_float("TEST_FLOAT", 0.0) == 2.718
    
    def test_get_env_float_with_invalid_value(self):
        """Test get_env_float raises ValueError for invalid float."""
        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_number"}):
            with pytest.raises(ValueError, match="must be a float"):
                get_env_float("TEST_FLOAT", 0.0)
    
    def test_get_env_bool_with_default(self):
        """Test get_env_bool returns default when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_env_bool("TEST_BOOL", True) is True
            assert get_env_bool("TEST_BOOL", False) is False
    
    def test_get_env_bool_with_truthy_values(self):
        """Test get_env_bool recognizes truthy values."""
        truthy_values = ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]
        for value in truthy_values:
            with patch.dict(os.environ, {"TEST_BOOL": value}):
                assert get_env_bool("TEST_BOOL", False) is True
    
    def test_get_env_bool_with_falsy_values(self):
        """Test get_env_bool recognizes falsy values."""
        falsy_values = ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF", ""]
        for value in falsy_values:
            with patch.dict(os.environ, {"TEST_BOOL": value}):
                assert get_env_bool("TEST_BOOL", True) is False
    
    def test_get_env_str_with_default(self):
        """Test get_env_str returns default when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_env_str("TEST_STR", "default") == "default"
            assert get_env_str("TEST_STR") is None
    
    def test_get_env_str_with_value(self):
        """Test get_env_str returns environment variable value."""
        with patch.dict(os.environ, {"TEST_STR": "hello_world"}):
            assert get_env_str("TEST_STR", "default") == "hello_world"


class TestConfig:
    """Test configuration class."""
    
    def test_config_defaults(self):
        """Test Config class loads default values."""
        # This tests the default values without environment variables
        assert Config.MODEL_PATH == "model.joblib"
        assert Config.RATE_LIMIT_WINDOW == 60
        assert Config.RATE_LIMIT_MAX_REQUESTS == 100
        assert Config.MAX_FILE_SIZE_MB == 100
        assert Config.MAX_DATASET_ROWS == 1_000_000
        assert Config.LOG_LEVEL == "INFO"
    
    def test_config_validation_positive_values(self):
        """Test Config.validate() accepts positive values."""
        # This should not raise any exceptions
        Config.validate()
    
    def test_config_validation_with_environment_override(self, monkeypatch):
        """Test Config with environment variable overrides."""
        monkeypatch.setenv("RATE_LIMIT_WINDOW", "30")
        monkeypatch.setenv("RATE_LIMIT_MAX_REQUESTS", "50")
        monkeypatch.setenv("MAX_FILE_SIZE_MB", "200")
        monkeypatch.setenv("MAX_DATASET_ROWS", "500000")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        
        # Reload the config module to pick up environment changes
        import importlib
        from src import config
        importlib.reload(config)
        
        assert config.Config.RATE_LIMIT_WINDOW == 30
        assert config.Config.RATE_LIMIT_MAX_REQUESTS == 50
        assert config.Config.MAX_FILE_SIZE_MB == 200
        assert config.Config.MAX_DATASET_ROWS == 500000
        assert config.Config.LOG_LEVEL == "DEBUG"
    
    def test_config_validation_fails_with_negative_values(self, monkeypatch):
        """Test Config.validate() raises error for negative values."""
        # Test with invalid rate limit window
        monkeypatch.setenv("RATE_LIMIT_WINDOW", "-1")
        
        import importlib
        from src import config
        
        with pytest.raises(ValueError, match="RATE_LIMIT_WINDOW must be positive"):
            importlib.reload(config)
    
    def test_config_validation_fails_with_zero_values(self, monkeypatch):
        """Test Config.validate() raises error for zero values."""
        # Test with zero max requests
        monkeypatch.setenv("RATE_LIMIT_MAX_REQUESTS", "0")
        
        import importlib
        from src import config
        
        with pytest.raises(ValueError, match="RATE_LIMIT_MAX_REQUESTS must be positive"):
            importlib.reload(config)


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_config_used_in_webapp(self):
        """Test that webapp correctly uses Config values."""
        from src.config import Config
        from src.webapp import _check_rate_limit
        
        # This is a basic integration test to ensure Config is being used
        # The actual rate limiting behavior is tested in test_webapp.py
        assert hasattr(Config, 'RATE_LIMIT_WINDOW')
        assert hasattr(Config, 'RATE_LIMIT_MAX_REQUESTS')
        
        # Basic smoke test for rate limiting function
        result = _check_rate_limit("127.0.0.1")
        assert isinstance(result, bool)
    
    def test_config_used_in_cli(self):
        """Test that CLI correctly uses Config values."""
        from src.config import Config
        
        # Test that CLI config values are accessible
        assert hasattr(Config, 'MAX_FILE_SIZE_MB')
        assert hasattr(Config, 'MAX_DATASET_ROWS')
        assert hasattr(Config, 'MODEL_PATH')
        
        # Verify they have reasonable values
        assert Config.MAX_FILE_SIZE_MB > 0
        assert Config.MAX_DATASET_ROWS > 0
        assert isinstance(Config.MODEL_PATH, str)