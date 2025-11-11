"""
Application Settings

Centralized configuration management.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration settings.
    
    Attributes:
        app_name: Name of the application
        version: Application version
        policy_dir: Directory containing policy YAML files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    
    app_name: str = "Agent Governance Hub"
    version: str = "0.1.0"
    policy_dir: Path = Path(__file__).parent / "policies"
    log_level: str = "INFO"
    
    class Config:
        env_prefix = "AGH_"  # Environment variables prefix


# Global settings instance
settings = Settings()
