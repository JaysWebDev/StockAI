#!/usr/bin/env python3
"""
Configuration Management for StockAI
Supports environment variables and YAML config files
"""

import os
import yaml
from dataclasses import dataclass
from typing import List, Optional
import logging

@dataclass
class Config:
    """Configuration class for StockAI application"""

    # API Keys
    alpha_vantage_api_key: str = ""
    finnhub_api_key: str = ""
    anthropic_api_key: str = ""

    # Database
    db_path: str = "data/unified_stock_intelligence.db"
    database_url: str = "sqlite:///unified_stock_intelligence.db"

    # Application Settings
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 5000
    env: str = "production"
    web_port: int = 8000

    # Contact
    email: str = ""

    # Authentication
    dashboard_username: str = "admin"
    dashboard_password: str = ""

    # Security
    secret_key: str = ""
    session_timeout: int = 3600

    # Email/SMTP
    smtp_server: str = ""
    email_user: str = ""
    email_password: str = ""
    email_alerts: bool = False
    alert_recipients: List[str] = None

    # Data Collection
    update_interval_seconds: int = 300
    max_securities: int = 1000
    rate_limit_delay: int = 12
    max_retries: int = 3
    batch_size: int = 50

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/stockai.log"

    # Watchlist
    default_watchlist: List[str] = None
    priority_stocks: List[str] = None

    # Analysis
    update_frequency: str = "daily"
    max_daily_analysis: int = 500
    data_retention_days: int = 90
    archive_retention_days: int = 365
    technical_indicators: List[str] = None

    def __post_init__(self):
        """Initialize default values for lists"""
        if self.default_watchlist is None:
            self.default_watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        if self.priority_stocks is None:
            self.priority_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        if self.technical_indicators is None:
            self.technical_indicators = ["sma_50", "sma_200", "rsi_14", "macd", "bollinger_bands"]
        if self.alert_recipients is None:
            self.alert_recipients = []

    @classmethod
    def load_from_file(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file with environment variable support"""
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Process environment variable substitutions
            config_data = cls._substitute_env_vars(yaml_config)

            # Create config instance
            config = cls()

            # Update with values from file
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            # Override with environment variables if they exist
            config._load_from_env()

            return config

        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
            logging.info("Using default configuration with environment variables")
            config = cls()
            config._load_from_env()
            return config

    def _load_from_env(self):
        """Load configuration from environment variables"""
        # API Keys
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY', self.alpha_vantage_api_key)
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY', self.finnhub_api_key)
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', self.anthropic_api_key)

        # Database
        self.db_path = os.getenv('DATABASE_PATH', self.db_path)
        self.database_url = os.getenv('DATABASE_URL', self.database_url)

        # Application
        self.debug = os.getenv('DEBUG', str(self.debug)).lower() == 'true'
        self.host = os.getenv('HOST', self.host)
        self.port = int(os.getenv('PORT', str(self.port)))
        self.env = os.getenv('ENV', self.env)

        # Contact
        self.email = os.getenv('EMAIL_ADDRESS', self.email)

        # Authentication
        self.dashboard_username = os.getenv('DASHBOARD_USERNAME', self.dashboard_username)
        self.dashboard_password = os.getenv('DASHBOARD_PASSWORD', self.dashboard_password)

        # Security
        self.secret_key = os.getenv('SECRET_KEY', self.secret_key)
        self.session_timeout = int(os.getenv('SESSION_TIMEOUT', str(self.session_timeout)))

        # Email
        self.smtp_server = os.getenv('SMTP_SERVER', self.smtp_server)
        self.email_user = os.getenv('EMAIL_USER', self.email_user)
        self.email_password = os.getenv('EMAIL_PASSWORD', self.email_password)

        # Data Collection
        self.update_interval_seconds = int(os.getenv('UPDATE_INTERVAL_SECONDS', str(self.update_interval_seconds)))
        self.max_securities = int(os.getenv('MAX_SECURITIES', str(self.max_securities)))

        # Logging
        self.log_level = os.getenv('LOG_LEVEL', self.log_level)
        self.log_file = os.getenv('LOG_FILE', self.log_file)

    @staticmethod
    def _substitute_env_vars(data):
        """Recursively substitute environment variables in YAML data"""
        if isinstance(data, dict):
            return {k: Config._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [Config._substitute_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
            # Extract environment variable name
            env_var = data[2:-1]
            return os.getenv(env_var, data)  # Return original if env var not found
        else:
            return data

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Check required API keys
        if not self.alpha_vantage_api_key:
            issues.append("ALPHA_VANTAGE_API_KEY is required")

        # Check dashboard authentication
        if not self.dashboard_password:
            issues.append("DASHBOARD_PASSWORD must be set for security")
        elif self.dashboard_password in ["temp1", "password", "admin", "change_this_secure_password"]:
            issues.append("DASHBOARD_PASSWORD is using a default/weak password")

        # Check secret key
        if not self.secret_key:
            issues.append("SECRET_KEY is required for sessions")
        elif len(self.secret_key) < 32:
            issues.append("SECRET_KEY should be at least 32 characters long")

        return issues

    def get_safe_config(self) -> dict:
        """Return configuration with sensitive values masked for logging"""
        safe_config = {}
        for key, value in self.__dict__.items():
            if any(sensitive in key.lower() for sensitive in ['password', 'key', 'secret', 'token']):
                if value:
                    safe_config[key] = "***REDACTED***"
                else:
                    safe_config[key] = "NOT_SET"
            else:
                safe_config[key] = value
        return safe_config