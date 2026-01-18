"""
Provider factory for creating LLM provider instances.

This module implements the factory pattern for creating and managing
LLM provider instances.
"""

import logging
from typing import Dict, Any, Optional, Type

from .base import BaseLLMProvider
from .openai import OpenAIProvider
from .config import ProviderConfig, load_provider_config, get_provider_from_env

logger = logging.getLogger(__name__)


# Provider registry: maps provider type to provider class
_PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    # Future providers can be added here:
    # "anthropic": AnthropicProvider,
    # "google": GoogleProvider,
}


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.
    
    Supports:
    - Creating providers from configuration
    - Creating providers from environment variables
    - Provider registry system
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the factory.
        
        Args:
            config_file: Optional path to provider configuration file
        """
        self.config_manager = ProviderConfig(config_file)
    
    @staticmethod
    def register_provider(
        provider_type: str,
        provider_class: Type[BaseLLMProvider]
    ) -> None:
        """
        Register a new provider type.
        
        Args:
            provider_type: Provider type identifier (e.g., 'openai')
            provider_class: Provider class that implements BaseLLMProvider
        """
        if not issubclass(provider_class, BaseLLMProvider):
            raise TypeError(
                f"Provider class must inherit from BaseLLMProvider, "
                f"got {provider_class}"
            )
        
        _PROVIDER_REGISTRY[provider_type] = provider_class
        logger.info(f"Registered provider: {provider_type}")
    
    @staticmethod
    def get_registered_providers() -> list[str]:
        """
        Get list of registered provider types.
        
        Returns:
            List of provider type strings
        """
        return list(_PROVIDER_REGISTRY.keys())
    
    def create(
        self,
        provider_type: str,
        config: Optional[Dict[str, Any]] = None,
        auto_initialize: bool = True
    ) -> BaseLLMProvider:
        """
        Create a provider instance.
        
        Args:
            provider_type: Provider type (e.g., 'openai')
            config: Optional configuration dictionary.
                    If not provided, loads from config file/env
            auto_initialize: Whether to automatically initialize the provider
            
        Returns:
            Initialized provider instance
            
        Raises:
            ValueError: If provider type is not registered or config is invalid
            RuntimeError: If provider initialization fails
        """
        # Check if provider is registered
        if provider_type not in _PROVIDER_REGISTRY:
            available = ", ".join(_PROVIDER_REGISTRY.keys())
            raise ValueError(
                f"Unknown provider type: '{provider_type}'. "
                f"Available providers: {available}"
            )
        
        # Get provider class
        provider_class = _PROVIDER_REGISTRY[provider_type]
        
        # Load configuration if not provided
        if config is None:
            config = self.config_manager.get_provider_config(provider_type)
        
        # Validate configuration
        is_valid, error_msg = self.config_manager.validate_provider_config(
            provider_type
        )
        if not is_valid and error_msg:
            logger.warning(f"Config validation warning: {error_msg}")
        
        # Create provider instance
        try:
            provider = provider_class(config)
            
            if auto_initialize:
                provider.initialize()
            
            logger.info(f"Created {provider_type} provider instance")
            return provider
            
        except Exception as e:
            logger.error(f"Failed to create {provider_type} provider: {e}")
            raise RuntimeError(
                f"Failed to create provider '{provider_type}': {e}"
            ) from e
    
    def create_from_env(
        self,
        auto_initialize: bool = True
    ) -> BaseLLMProvider:
        """
        Create a provider instance from environment variables.
        
        Uses LLM_PROVIDER environment variable to determine provider type,
        then loads configuration from environment variables and config file.
        
        Args:
            auto_initialize: Whether to automatically initialize the provider
            
        Returns:
            Initialized provider instance
            
        Raises:
            ValueError: If LLM_PROVIDER is not set or provider type is invalid
            RuntimeError: If provider initialization fails
        """
        # Get provider type from environment
        provider_type = get_provider_from_env()
        
        if not provider_type:
            # Fall back to default provider
            provider_type = self.config_manager.get_default_provider()
            logger.info(
                f"LLM_PROVIDER not set, using default: {provider_type}"
            )
        
        provider_type = provider_type.lower()
        
        return self.create(
            provider_type=provider_type,
            config=None,  # Will load from config/env
            auto_initialize=auto_initialize
        )


# Global factory instance
_default_factory: Optional[LLMProviderFactory] = None


def get_factory(config_file: Optional[str] = None) -> LLMProviderFactory:
    """
    Get or create the default factory instance.
    
    Args:
        config_file: Optional path to config file
        
    Returns:
        LLMProviderFactory instance
    """
    global _default_factory
    
    if _default_factory is None:
        _default_factory = LLMProviderFactory(config_file)
    
    return _default_factory


def create(
    provider_type: str,
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    config_file: Optional[str] = None
) -> BaseLLMProvider:
    """
    Convenience function to create a provider instance.
    
    Args:
        provider_type: Provider type (e.g., 'openai')
        config: Optional configuration dictionary
        auto_initialize: Whether to automatically initialize
        config_file: Optional path to config file
        
    Returns:
        Provider instance
    """
    factory = get_factory(config_file)
    return factory.create(provider_type, config, auto_initialize)


def create_from_env(
    auto_initialize: bool = True,
    config_file: Optional[str] = None
) -> BaseLLMProvider:
    """
    Convenience function to create a provider from environment variables.
    
    Args:
        auto_initialize: Whether to automatically initialize
        config_file: Optional path to config file
        
    Returns:
        Provider instance
    """
    factory = get_factory(config_file)
    return factory.create_from_env(auto_initialize)
