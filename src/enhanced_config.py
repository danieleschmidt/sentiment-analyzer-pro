"""
Enhanced configuration system for sentiment analyzer
Generation 1: Make It Work - Environment-aware configuration
"""
import os
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    nb_alpha: float = 1.0
    max_features: int = 10000
    min_df: int = 1
    max_df: float = 1.0
    use_stemming: bool = False
    use_lemmatization: bool = True

@dataclass
class ServerConfig:
    """Web server configuration"""
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    workers: int = 1
    timeout: int = 30

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    console: bool = True

@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret: Optional[str] = None
    jwt_expiry_hours: int = 24
    rate_limit_per_minute: int = 60
    enable_cors: bool = False
    allowed_origins: list = None

    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = []

@dataclass
class AppConfig:
    """Main application configuration"""
    model: ModelConfig
    server: ServerConfig
    logging: LoggingConfig
    security: SecurityConfig
    data_dir: str = "data"
    models_dir: str = "models"
    cache_dir: str = "cache"
    environment: str = "development"

class ConfigManager:
    """Enhanced configuration manager with environment support"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self._config: Optional[AppConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file and environment variables"""
        try:
            # Start with defaults
            config_data = self._get_default_config()
            
            # Override with file config if exists
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                config_data = self._deep_merge(config_data, file_config)
            
            # Override with environment variables
            env_config = self._load_from_environment()
            config_data = self._deep_merge(config_data, env_config)
            
            # Create config object
            self._config = AppConfig(
                model=ModelConfig(**config_data.get("model", {})),
                server=ServerConfig(**config_data.get("server", {})),
                logging=LoggingConfig(**config_data.get("logging", {})),
                security=SecurityConfig(**config_data.get("security", {})),
                data_dir=config_data.get("data_dir", "data"),
                models_dir=config_data.get("models_dir", "models"),
                cache_dir=config_data.get("cache_dir", "cache"),
                environment=config_data.get("environment", "development")
            )
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Using defaults.")
            self._config = self._get_default_app_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration dictionary"""
        return {
            "model": asdict(ModelConfig()),
            "server": asdict(ServerConfig()),
            "logging": asdict(LoggingConfig()),
            "security": asdict(SecurityConfig()),
            "data_dir": "data",
            "models_dir": "models",
            "cache_dir": "cache",
            "environment": "development"
        }
    
    def _get_default_app_config(self) -> AppConfig:
        """Get default app configuration object"""
        return AppConfig(
            model=ModelConfig(),
            server=ServerConfig(),
            logging=LoggingConfig(),
            security=SecurityConfig(),
        )
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Model config
        model_config = {}
        if os.getenv("MODEL_NB_ALPHA"):
            model_config["nb_alpha"] = float(os.getenv("MODEL_NB_ALPHA"))
        if os.getenv("MODEL_MAX_FEATURES"):
            model_config["max_features"] = int(os.getenv("MODEL_MAX_FEATURES"))
        if model_config:
            env_config["model"] = model_config
        
        # Server config
        server_config = {}
        if os.getenv("SERVER_HOST"):
            server_config["host"] = os.getenv("SERVER_HOST")
        if os.getenv("SERVER_PORT"):
            server_config["port"] = int(os.getenv("SERVER_PORT"))
        if os.getenv("SERVER_DEBUG"):
            server_config["debug"] = os.getenv("SERVER_DEBUG").lower() == "true"
        if server_config:
            env_config["server"] = server_config
        
        # Security config
        security_config = {}
        if os.getenv("JWT_SECRET"):
            security_config["jwt_secret"] = os.getenv("JWT_SECRET")
        if os.getenv("JWT_EXPIRY_HOURS"):
            security_config["jwt_expiry_hours"] = int(os.getenv("JWT_EXPIRY_HOURS"))
        if security_config:
            env_config["security"] = security_config
        
        # General config
        if os.getenv("DATA_DIR"):
            env_config["data_dir"] = os.getenv("DATA_DIR")
        if os.getenv("MODELS_DIR"):
            env_config["models_dir"] = os.getenv("MODELS_DIR")
        if os.getenv("ENVIRONMENT"):
            env_config["environment"] = os.getenv("ENVIRONMENT")
        
        return env_config
    
    def _deep_merge(self, base: Dict, overlay: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    @property
    def config(self) -> AppConfig:
        """Get current configuration"""
        if self._config is None:
            self._load_config()
        return self._config
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            config_dict = {
                "model": asdict(self.config.model),
                "server": asdict(self.config.server),
                "logging": asdict(self.config.logging),
                "security": asdict(self.config.security),
                "data_dir": self.config.data_dir,
                "models_dir": self.config.models_dir,
                "cache_dir": self.config.cache_dir,
                "environment": self.config.environment
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def update_config(self, **kwargs) -> None:
        """Update configuration values"""
        config_dict = asdict(self.config)
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown config key: {key}")
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist"""
        directories = [
            self.config.data_dir,
            self.config.models_dir,
            self.config.cache_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

# Global config manager instance
_config_manager: Optional[ConfigManager] = None

def get_config() -> AppConfig:
    """Get global configuration instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config

def init_config(config_path: Optional[str] = None) -> ConfigManager:
    """Initialize configuration manager"""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = get_config()
    print(f"Environment: {config.environment}")
    print(f"Server: {config.server.host}:{config.server.port}")
    print(f"Data dir: {config.data_dir}")