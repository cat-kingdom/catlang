"""
Unit tests for LLM provider factory.
"""

import os
import pytest
import tempfile
import yaml
from unittest.mock import Mock, patch, MagicMock
from llm_provider.factory import (
    LLMProviderFactory,
    create,
    create_from_env,
    get_factory,
)
from llm_provider.base import BaseLLMProvider
from llm_provider.openai import OpenAIProvider


class MockProvider(BaseLLMProvider):
    """Mock provider for testing."""
    
    def initialize(self):
        self._initialized = True
    
    def generate(self, prompt: str, params=None):
        from llm_provider.base import GenerationResponse
        return GenerationResponse(content="mock", model="mock")
    
    def stream(self, prompt: str, params=None):
        yield "mock"
    
    async def stream_async(self, prompt: str, params=None):
        yield "mock"
    
    def get_models(self):
        return ["mock"]
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model=None):
        return 0.0
    
    def validate_credentials(self):
        return True
    
    @property
    def provider_type(self):
        return "mock"
    
    @property
    def default_model(self):
        return "mock"


class TestLLMProviderFactory:
    """Test LLMProviderFactory class."""
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = LLMProviderFactory()
        assert factory is not None
    
    def test_get_registered_providers(self):
        """Test getting registered providers."""
        providers = LLMProviderFactory.get_registered_providers()
        assert "openai" in providers
    
    def test_register_provider(self):
        """Test registering a new provider."""
        LLMProviderFactory.register_provider("test", MockProvider)
        
        providers = LLMProviderFactory.get_registered_providers()
        assert "test" in providers
        
        # Clean up
        from llm_provider.factory import _PROVIDER_REGISTRY
        _PROVIDER_REGISTRY.pop("test", None)
    
    def test_register_invalid_provider(self):
        """Test registering invalid provider class."""
        with pytest.raises(TypeError):
            LLMProviderFactory.register_provider("invalid", str)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        config_data = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "type": "openai",
                    "default_model": "gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                }
            },
            "default_provider": "openai"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            factory = LLMProviderFactory(config_path)
            provider = factory.create("openai", auto_initialize=False)
            
            assert isinstance(provider, OpenAIProvider)
            assert provider.provider_type == "openai"
        finally:
            os.unlink(config_path)
    
    def test_create_unknown_provider(self):
        """Test creating unknown provider."""
        factory = LLMProviderFactory()
        
        with pytest.raises(ValueError, match="Unknown provider"):
            factory.create("nonexistent")
    
    def test_create_with_custom_config(self):
        """Test creating provider with custom config."""
        factory = LLMProviderFactory()
        
        custom_config = {
            "model": "gpt-4",
            "api_key": "test-key",
        }
        
        provider = factory.create(
            "openai",
            config=custom_config,
            auto_initialize=False
        )
        
        assert isinstance(provider, OpenAIProvider)
        assert provider._model == "gpt-4"
    
    @patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"})
    def test_create_from_env(self):
        """Test creating provider from environment."""
        config_data = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "type": "openai",
                    "default_model": "gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                }
            },
            "default_provider": "openai"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            factory = LLMProviderFactory(config_path)
            provider = factory.create_from_env(auto_initialize=False)
            
            assert isinstance(provider, OpenAIProvider)
        finally:
            os.unlink(config_path)
    
    def test_create_from_env_default(self):
        """Test creating provider with default when env not set."""
        config_data = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "type": "openai",
                    "default_model": "gpt-4o-mini",
                }
            },
            "default_provider": "openai"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Remove LLM_PROVIDER if set
            os.environ.pop("LLM_PROVIDER", None)
            
            factory = LLMProviderFactory(config_path)
            provider = factory.create_from_env(auto_initialize=False)
            
            assert isinstance(provider, OpenAIProvider)
        finally:
            os.unlink(config_path)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_function(self):
        """Test create convenience function."""
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }
        
        provider = create("openai", config=config, auto_initialize=False)
        
        assert isinstance(provider, OpenAIProvider)
    
    @patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"})
    def test_create_from_env_function(self):
        """Test create_from_env convenience function."""
        config_data = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "type": "openai",
                    "default_model": "gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                }
            },
            "default_provider": "openai"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            provider = create_from_env(
                auto_initialize=False,
                config_file=config_path
            )
            
            assert isinstance(provider, OpenAIProvider)
        finally:
            os.unlink(config_path)
    
    def test_get_factory(self):
        """Test get_factory function."""
        factory1 = get_factory()
        factory2 = get_factory()
        
        # Should return same instance
        assert factory1 is factory2
