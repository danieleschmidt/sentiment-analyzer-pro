
import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import logging

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "sentiment_analyzer"
    username: str = "user"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 10
    
    def get_connection_string(self) -> str:
        """Get database connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}"

@dataclass
class ModelConfig:
    default_model: str = "logistic_regression"
    model_path: str = "models/"
    cache_size: int = 100
    batch_size: int = 32
    max_text_length: int = 10000
    
@dataclass
class SecurityConfig:
    secret_key: str = ""
    jwt_expiry_hours: int = 24
    rate_limit_requests: int = 100
    rate_limit_window_minutes: int = 60
    enable_csrf: bool = True
    enable_cors: bool = False
    allowed_origins: list = None
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = []

@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"
    file_path: str = "logs/application.log"
    max_file_size: str = "10MB"
    backup_count: int = 5
    enable_console: bool = True

@dataclass
class AppConfig:
    """Main application configuration."""
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 5000
    workers: int = 1
    environment: str = "development"
    
    database: DatabaseConfig = None
    model: ModelConfig = None
    security: SecurityConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        if self.database is None:
            self.database = DatabaseConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.logging is None:
            self.logging = LoggingConfig()

class ConfigManager:
    """Robust configuration management system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config: AppConfig = AppConfig()
        self.logger = logging.getLogger(__name__)
        
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations."""
        possible_paths = [
            "config.json",
            "config.yaml",
            "config.yml",
            "app.json",
            "app.yaml",
            "app.yml",
            os.path.expanduser("~/.sentiment_analyzer/config.json"),
            "/etc/sentiment_analyzer/config.json"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        return None
    
    def load_config(self) -> AppConfig:
        """Load configuration from file and environment variables."""
        # Load from file if exists
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith(('.yaml', '.yml')):
                        file_config = yaml.safe_load(f)
                    else:
                        file_config = json.load(f)
                
                self.config = self._merge_config(self.config, file_config)
                self.logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                self.logger.error(f"Error loading config file {self.config_path}: {e}")
        
        # Override with environment variables
        self._load_from_environment()
        
        return self.config
    
    def _merge_config(self, base_config: AppConfig, file_config: Dict[str, Any]) -> AppConfig:
        """Merge file configuration into base configuration."""
        # Create new config from base
        config_dict = {
            'debug': file_config.get('debug', base_config.debug),
            'host': file_config.get('host', base_config.host),
            'port': file_config.get('port', base_config.port),
            'workers': file_config.get('workers', base_config.workers),
            'environment': file_config.get('environment', base_config.environment),
        }
        
        # Handle nested configurations
        if 'database' in file_config:
            db_config = file_config['database']
            config_dict['database'] = DatabaseConfig(
                host=db_config.get('host', base_config.database.host),
                port=db_config.get('port', base_config.database.port),
                database=db_config.get('database', base_config.database.database),
                username=db_config.get('username', base_config.database.username),
                password=db_config.get('password', base_config.database.password),
                ssl_mode=db_config.get('ssl_mode', base_config.database.ssl_mode),
                pool_size=db_config.get('pool_size', base_config.database.pool_size),
            )
        
        if 'model' in file_config:
            model_config = file_config['model']
            config_dict['model'] = ModelConfig(
                default_model=model_config.get('default_model', base_config.model.default_model),
                model_path=model_config.get('model_path', base_config.model.model_path),
                cache_size=model_config.get('cache_size', base_config.model.cache_size),
                batch_size=model_config.get('batch_size', base_config.model.batch_size),
                max_text_length=model_config.get('max_text_length', base_config.model.max_text_length),
            )
        
        return AppConfig(**config_dict)
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Main app settings
        if os.getenv('DEBUG'):
            self.config.debug = os.getenv('DEBUG', '').lower() in ('true', '1', 'yes')
        if os.getenv('HOST'):
            self.config.host = os.getenv('HOST')
        if os.getenv('PORT'):
            self.config.port = int(os.getenv('PORT'))
        if os.getenv('WORKERS'):
            self.config.workers = int(os.getenv('WORKERS'))
        if os.getenv('ENVIRONMENT'):
            self.config.environment = os.getenv('ENVIRONMENT')
        
        # Database settings
        if os.getenv('DB_HOST'):
            self.config.database.host = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            self.config.database.port = int(os.getenv('DB_PORT'))
        if os.getenv('DB_NAME'):
            self.config.database.database = os.getenv('DB_NAME')
        if os.getenv('DB_USER'):
            self.config.database.username = os.getenv('DB_USER')
        if os.getenv('DB_PASSWORD'):
            self.config.database.password = os.getenv('DB_PASSWORD')
        
        # Security settings
        if os.getenv('SECRET_KEY'):
            self.config.security.secret_key = os.getenv('SECRET_KEY')
        if os.getenv('JWT_EXPIRY_HOURS'):
            self.config.security.jwt_expiry_hours = int(os.getenv('JWT_EXPIRY_HOURS'))
    
    def save_config(self, config_path: Optional[str] = None):
        """Save current configuration to file."""
        path = config_path or self.config_path or "config.json"
        
        config_dict = {
            'debug': self.config.debug,
            'host': self.config.host,
            'port': self.config.port,
            'workers': self.config.workers,
            'environment': self.config.environment,
            'database': {
                'host': self.config.database.host,
                'port': self.config.database.port,
                'database': self.config.database.database,
                'username': self.config.database.username,
                'ssl_mode': self.config.database.ssl_mode,
                'pool_size': self.config.database.pool_size,
                # Don't save password to file
            },
            'model': {
                'default_model': self.config.model.default_model,
                'model_path': self.config.model.model_path,
                'cache_size': self.config.model.cache_size,
                'batch_size': self.config.model.batch_size,
                'max_text_length': self.config.model.max_text_length,
            },
            'security': {
                'jwt_expiry_hours': self.config.security.jwt_expiry_hours,
                'rate_limit_requests': self.config.security.rate_limit_requests,
                'rate_limit_window_minutes': self.config.security.rate_limit_window_minutes,
                'enable_csrf': self.config.security.enable_csrf,
                'enable_cors': self.config.security.enable_cors,
                'allowed_origins': self.config.security.allowed_origins,
                # Don't save secret_key to file
            },
            'logging': {
                'level': self.config.logging.level,
                'format': self.config.logging.format,
                'file_path': self.config.logging.file_path,
                'max_file_size': self.config.logging.max_file_size,
                'backup_count': self.config.logging.backup_count,
                'enable_console': self.config.logging.enable_console,
            }
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Configuration saved to {path}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate port range
        if not (1 <= self.config.port <= 65535):
            issues.append("Port must be between 1 and 65535")
        
        # Validate workers
        if self.config.workers < 1:
            issues.append("Workers must be at least 1")
        
        # Validate database port
        if not (1 <= self.config.database.port <= 65535):
            issues.append("Database port must be between 1 and 65535")
        
        # Validate model settings
        if self.config.model.cache_size < 1:
            issues.append("Model cache size must be at least 1")
        
        if self.config.model.batch_size < 1:
            issues.append("Batch size must be at least 1")
        
        # Validate security settings
        if not self.config.security.secret_key and self.config.environment == "production":
            issues.append("Secret key is required in production environment")
        
        return issues

# Global configuration manager
config_manager = ConfigManager()
