"""
Configuration management for LLM providers.

This module handles loading and validating provider configurations from
environment variables and YAML configuration files.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ProviderConfig:
    """
    Configuration manager for LLM providers.
    
    Supports loading configuration from:
    1. Environment variables (highest priority)
    2. YAML configuration files
    3. Default values
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to YAML config file.
                        Defaults to config/providers.yaml
        """
        if config_file is None:
            # Default to config/providers.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_file = project_root / "config" / "providers.yaml"
        
        self.config_file = Path(config_file)
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
                self._config = {}
        else:
            logger.warning(f"Config file not found: {self.config_file}")
            self._config = {}
    
    def get_provider_config(
        self,
        provider_type: str,
        use_env: bool = True
    ) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.
        
        Args:
            provider_type: Provider type (e.g., 'openai', 'anthropic')
            use_env: Whether to override with environment variables
            
        Returns:
            Configuration dictionary for the provider
        """
        # Start with YAML config
        providers = self._config.get("providers", {})
        provider_config = providers.get(provider_type, {}).copy()
        
        # Override with environment variables if requested
        if use_env:
            provider_config = self._apply_env_overrides(provider_type, provider_config)
        
        return provider_config
    
    def _apply_env_overrides(
        self,
        provider_type: str,
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Args:
            provider_type: Provider type
            base_config: Base configuration from YAML
            
        Returns:
            Configuration with environment variable overrides applied
        """
        config = base_config.copy()
        
        # Provider-specific environment variable mappings
        env_mappings = {
            "openai": {
                "api_key": "OPENAI_API_KEY",
                "model": "OPENAI_MODEL",
                "base_url": "OPENAI_BASE_URL",
                "timeout": "OPENAI_TIMEOUT",
                "max_retries": "OPENAI_MAX_RETRIES",
            },
            "anthropic": {
                "api_key": "ANTHROPIC_API_KEY",
                "model": "ANTHROPIC_MODEL",
                "base_url": "ANTHROPIC_BASE_URL",
                "timeout": "ANTHROPIC_TIMEOUT",
                "max_retries": "ANTHROPIC_MAX_RETRIES",
            },
            "google": {
                "api_key": "GOOGLE_API_KEY",
                "model": "GOOGLE_MODEL",
                "base_url": "GOOGLE_BASE_URL",
                "timeout": "GOOGLE_TIMEOUT",
                "max_retries": "GOOGLE_MAX_RETRIES",
            },
        }
        
        mappings = env_mappings.get(provider_type, {})
        
        # Apply environment variable overrides
        for config_key, env_var in mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Type conversion for numeric values
                if config_key in ["timeout", "max_retries"]:
                    try:
                        config[config_key] = int(env_value)
                    except ValueError:
                        logger.warning(
                            f"Invalid {env_var} value '{env_value}', "
                            f"using default"
                        )
                else:
                    config[config_key] = env_value
                    # Also override default_model if model is set
                    if config_key == "model":
                        config["default_model"] = env_value
        
        return config
    
    def get_default_provider(self) -> str:
        """
        Get the default provider type.
        
        Returns:
            Default provider type string
        """
        # Check environment variable first
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider:
            return env_provider.lower()
        
        # Fall back to YAML config
        return self._config.get("default_provider", "openai")
    
    def is_provider_enabled(self, provider_type: str) -> bool:
        """
        Check if a provider is enabled.
        
        Args:
            provider_type: Provider type to check
            
        Returns:
            True if provider is enabled, False otherwise
        """
        providers = self._config.get("providers", {})
        provider_config = providers.get(provider_type, {})
        return provider_config.get("enabled", False)
    
    def validate_provider_config(
        self,
        provider_type: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate provider configuration.
        
        Args:
            provider_type: Provider type to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        providers = self._config.get("providers", {})
        
        if provider_type not in providers:
            return False, f"Provider '{provider_type}' not found in configuration"
        
        provider_config = providers[provider_type]
        
        if not provider_config.get("enabled", False):
            return False, f"Provider '{provider_type}' is not enabled"
        
        # Check required fields based on provider type
        required_fields = {
            "openai": ["type", "api_key_env"],
            "anthropic": ["type", "api_key_env"],
            "google": ["type", "api_key_env"],
        }
        
        required = required_fields.get(provider_type, ["type"])
        
        for field in required:
            if field not in provider_config:
                return False, f"Missing required field '{field}' for provider '{provider_type}'"
        
        return True, None
    
    def get_all_providers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all configured providers.
        
        Returns:
            Dictionary mapping provider types to their configurations
        """
        return self._config.get("providers", {})


def load_provider_config(
    provider_type: Optional[str] = None,
    config_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to load provider configuration.
    
    Args:
        provider_type: Optional provider type (uses default if not provided)
        config_file: Optional path to config file
        
    Returns:
        Configuration dictionary for the provider
    """
    config_manager = ProviderConfig(config_file)
    
    if provider_type is None:
        provider_type = config_manager.get_default_provider()
    
    return config_manager.get_provider_config(provider_type)


def get_provider_from_env() -> Optional[str]:
    """
    Get provider type from environment variable.
    
    Returns:
        Provider type string or None if not set
    """
    return os.getenv("LLM_PROVIDER")
