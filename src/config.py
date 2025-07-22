"""Configuration management for sentiment analyzer."""

import os
from typing import Optional, Union


def get_env_int(key: str, default: int) -> int:
    """Get an integer environment variable with validation."""
    value = os.getenv(key)
    if value is None:
        return default
    
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Environment variable {key} must be an integer, got: {value}")


def get_env_float(key: str, default: float) -> float:
    """Get a float environment variable with validation."""
    value = os.getenv(key)
    if value is None:
        return default
    
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"Environment variable {key} must be a float, got: {value}")


def get_env_bool(key: str, default: bool) -> bool:
    """Get a boolean environment variable with validation."""
    value = os.getenv(key)
    if value is None:
        return default
    
    return value.lower() in ("true", "1", "yes", "on")


def get_env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a string environment variable."""
    return os.getenv(key, default)


# Configuration constants with environment variable support
class Config:
    """Application configuration loaded from environment variables."""
    
    # Model configuration
    MODEL_PATH = get_env_str("MODEL_PATH", "model.joblib")
    
    # Web server configuration
    RATE_LIMIT_WINDOW = get_env_int("RATE_LIMIT_WINDOW", 60)
    RATE_LIMIT_MAX_REQUESTS = get_env_int("RATE_LIMIT_MAX_REQUESTS", 100)
    
    # Security limits
    MAX_FILE_SIZE_MB = get_env_int("MAX_FILE_SIZE_MB", 100)
    MAX_DATASET_ROWS = get_env_int("MAX_DATASET_ROWS", 1_000_000)
    
    # Logging
    LOG_LEVEL = get_env_str("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration values."""
        if cls.RATE_LIMIT_WINDOW <= 0:
            raise ValueError("RATE_LIMIT_WINDOW must be positive")
        
        if cls.RATE_LIMIT_MAX_REQUESTS <= 0:
            raise ValueError("RATE_LIMIT_MAX_REQUESTS must be positive")
        
        if cls.MAX_FILE_SIZE_MB <= 0:
            raise ValueError("MAX_FILE_SIZE_MB must be positive")
        
        if cls.MAX_DATASET_ROWS <= 0:
            raise ValueError("MAX_DATASET_ROWS must be positive")


# Validate configuration on import
Config.validate()