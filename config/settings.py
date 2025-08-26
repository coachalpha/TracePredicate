"""
Configuration settings for TracePredicate system.

This module manages all configuration settings including database connections,
API keys, data collection parameters, and analysis settings.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    database: str = Field(default="tracepredicate", env="DB_NAME")
    username: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    
    # Connection pool settings
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")
    
    # SSL settings
    ssl_mode: str = Field(default="prefer", env="DB_SSL_MODE")
    
    # Echo SQL queries for debugging
    echo: bool = Field(default=False, env="DB_ECHO")
    
    @property
    def database_url(self) -> str:
        """Get the database connection URL."""
        if self.password:
            auth = f"{self.username}:{self.password}"
        else:
            auth = self.username
            
        url = f"postgresql://{auth}@{self.host}:{self.port}/{self.database}"
        
        if self.ssl_mode != "prefer":
            url += f"?sslmode={self.ssl_mode}"
            
        return url
    
    class Config:
        env_prefix = "DB_"


class APISettings(BaseSettings):
    """API and external service settings."""
    
    # FDA API settings
    fda_api_key: Optional[str] = Field(default=None, env="FDA_API_KEY")
    fda_api_base_url: str = Field(default="https://api.fda.gov", env="FDA_API_BASE_URL")
    fda_request_timeout: int = Field(default=30, env="FDA_REQUEST_TIMEOUT")
    fda_max_concurrent: int = Field(default=5, env="FDA_MAX_CONCURRENT")
    
    # OpenAI API settings for LLM extraction
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=2000, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    
    # Rate limiting
    api_rate_limit_per_minute: int = Field(default=60, env="API_RATE_LIMIT_PER_MINUTE")
    
    class Config:
        env_prefix = "API_"


class DataCollectionSettings(BaseSettings):
    """Data collection and scraping settings."""
    
    # Default device codes to focus on
    default_device_codes: List[str] = Field(default=["KWA"], env="DEFAULT_DEVICE_CODES")
    
    # Scraping limits and timeouts
    max_devices_per_request: int = Field(default=100, env="MAX_DEVICES_PER_REQUEST")
    request_delay_seconds: float = Field(default=1.0, env="REQUEST_DELAY_SECONDS")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    
    # PDF processing settings
    pdf_extraction_timeout: int = Field(default=30, env="PDF_EXTRACTION_TIMEOUT")
    max_pdf_pages: int = Field(default=50, env="MAX_PDF_PAGES")
    
    # MAUDE search settings
    maude_max_years_back: int = Field(default=10, env="MAUDE_MAX_YEARS_BACK")
    maude_batch_size: int = Field(default=1000, env="MAUDE_BATCH_SIZE")
    
    # Recall search settings
    recall_max_results: int = Field(default=5000, env="RECALL_MAX_RESULTS")
    
    @validator('default_device_codes', pre=True)
    def parse_device_codes(cls, v):
        if isinstance(v, str):
            return [code.strip() for code in v.split(',')]
        return v
    
    class Config:
        env_prefix = "DATA_"


class LDISettings(BaseSettings):
    """LDI calculation and analysis settings."""
    
    # Default LDI weights (will be optimized)
    default_semantic_weight: float = Field(default=0.4, env="LDI_SEMANTIC_WEIGHT")
    default_parameter_weight: float = Field(default=0.4, env="LDI_PARAMETER_WEIGHT")
    default_chain_weight: float = Field(default=0.2, env="LDI_CHAIN_WEIGHT")
    
    # BioBERT model settings
    biobert_model_name: str = Field(default="dmis-lab/biobert-base-cased-v1.1", env="BIOBERT_MODEL")
    biobert_max_length: int = Field(default=512, env="BIOBERT_MAX_LENGTH")
    
    # Parameter standardization settings
    parameter_standardization_method: str = Field(default="standard", env="PARAM_STANDARDIZATION")  # 'standard', 'minmax', 'robust'
    
    # Chain length settings
    max_chain_depth: int = Field(default=15, env="MAX_CHAIN_DEPTH")
    chain_weight_decay: float = Field(default=0.9, env="CHAIN_WEIGHT_DECAY")  # Decay factor for longer chains
    
    # Optimization settings
    optimization_method: str = Field(default="scipy", env="OPTIMIZATION_METHOD")  # 'scipy', 'optuna'
    optimization_trials: int = Field(default=100, env="OPTIMIZATION_TRIALS")
    optimization_timeout: int = Field(default=3600, env="OPTIMIZATION_TIMEOUT")  # seconds
    
    @validator('default_semantic_weight', 'default_parameter_weight', 'default_chain_weight')
    def validate_weight_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weights must be between 0 and 1")
        return v
    
    class Config:
        env_prefix = "LDI_"


class NetworkAnalysisSettings(BaseSettings):
    """Network analysis settings."""
    
    # Graph construction settings
    include_confidence_threshold: float = Field(default=0.5, env="NETWORK_CONFIDENCE_THRESHOLD")
    max_nodes: int = Field(default=10000, env="NETWORK_MAX_NODES")
    
    # Centrality calculation settings
    calculate_all_centralities: bool = Field(default=True, env="NETWORK_CALC_ALL_CENTRALITIES")
    centrality_timeout: int = Field(default=300, env="NETWORK_CENTRALITY_TIMEOUT")  # seconds
    
    # Risk propagation settings
    risk_propagation_decay: float = Field(default=0.8, env="NETWORK_RISK_DECAY")
    risk_propagation_steps: int = Field(default=5, env="NETWORK_RISK_STEPS")
    
    # Visualization settings
    default_layout: str = Field(default="spring", env="NETWORK_LAYOUT")  # 'spring', 'circular', 'hierarchical'
    node_size_metric: str = Field(default="degree", env="NETWORK_NODE_SIZE_METRIC")
    
    class Config:
        env_prefix = "NETWORK_"


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    log_max_bytes: int = Field(default=10485760, env="LOG_MAX_BYTES")  # 10MB
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # Specific logger levels
    sqlalchemy_log_level: str = Field(default="WARNING", env="SQLALCHEMY_LOG_LEVEL")
    requests_log_level: str = Field(default="WARNING", env="REQUESTS_LOG_LEVEL")
    
    class Config:
        env_prefix = "LOG_"


class CacheSettings(BaseSettings):
    """Caching configuration settings."""
    
    # Redis cache settings (if available)
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Cache timeouts (seconds)
    device_cache_timeout: int = Field(default=3600, env="CACHE_DEVICE_TIMEOUT")  # 1 hour
    adverse_event_cache_timeout: int = Field(default=1800, env="CACHE_EVENT_TIMEOUT")  # 30 minutes
    ldi_cache_timeout: int = Field(default=7200, env="CACHE_LDI_TIMEOUT")  # 2 hours
    
    # Enable/disable caching
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    
    class Config:
        env_prefix = "CACHE_"


class ApplicationSettings(BaseSettings):
    """Main application settings."""
    
    # Application info
    app_name: str = Field(default="TracePredicate", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # File paths
    data_directory: Path = Field(default=Path("data"), env="DATA_DIRECTORY")
    output_directory: Path = Field(default=Path("output"), env="OUTPUT_DIRECTORY")
    temp_directory: Path = Field(default=Path("/tmp/tracepredicate"), env="TEMP_DIRECTORY")
    
    # Processing settings
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    chunk_size: int = Field(default=100, env="CHUNK_SIZE")
    
    # Web interface settings
    web_host: str = Field(default="localhost", env="WEB_HOST")
    web_port: int = Field(default=8000, env="WEB_PORT")
    
    @validator('data_directory', 'output_directory', 'temp_directory', pre=True)
    def ensure_path_object(cls, v):
        return Path(v) if not isinstance(v, Path) else v
    
    class Config:
        env_prefix = "APP_"


class Settings(BaseSettings):
    """Main settings class that combines all setting categories."""
    
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    data_collection: DataCollectionSettings = DataCollectionSettings()
    ldi: LDISettings = LDISettings()
    network: NetworkAnalysisSettings = NetworkAnalysisSettings()
    logging: LoggingSettings = LoggingSettings()
    cache: CacheSettings = CacheSettings()
    app: ApplicationSettings = ApplicationSettings()
    
    class Config:
        # Look for .env file
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def create_directories(self):
        """Create necessary directories."""
        directories = [
            self.app.data_directory,
            self.app.output_directory,
            self.app.temp_directory,
            self.app.data_directory / "raw",
            self.app.data_directory / "processed",
            self.app.data_directory / "exports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_settings(self) -> List[str]:
        """Validate settings and return list of warnings/errors."""
        warnings = []
        
        # Check for required API keys if needed
        if not self.api.openai_api_key:
            warnings.append("OpenAI API key not set - LLM extraction will not work")
        
        # Check LDI weights sum to 1
        weight_sum = (
            self.ldi.default_semantic_weight + 
            self.ldi.default_parameter_weight + 
            self.ldi.default_chain_weight
        )
        if abs(weight_sum - 1.0) > 0.001:
            warnings.append(f"LDI weights sum to {weight_sum:.3f}, not 1.0")
        
        # Check database connection (basic validation)
        if not self.database.database:
            warnings.append("Database name not specified")
        
        return warnings


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def load_settings_from_file(file_path: str) -> Settings:
    """Load settings from a specific file."""
    return Settings(_env_file=file_path)


def main():
    """Test settings loading and validation."""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load settings
    settings = get_settings()
    
    # Create directories
    settings.create_directories()
    
    # Validate settings
    warnings = settings.validate_settings()
    
    print(f"Loaded settings for {settings.app.app_name} v{settings.app.app_version}")
    print(f"Database: {settings.database.host}:{settings.database.port}/{settings.database.database}")
    print(f"Data directory: {settings.app.data_directory}")
    print(f"Debug mode: {settings.app.debug}")
    
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nAll settings validated successfully!")
    
    # Test database URL generation
    print(f"\nDatabase URL: {settings.database.database_url}")


if __name__ == "__main__":
    main()