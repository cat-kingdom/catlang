"""
Integration tests for LLM provider system.

These tests verify that all components work together correctly.
"""

import os
import pytest
import tempfile
import yaml
from llm_provider.factory import create, create_from_env
from llm_provider.base import GenerationParams


class TestProviderIntegration:
    """Integration tests for provider system."""
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set, skipping integration test"
    )
    def test_openai_provider_end_to_end(self):
        """Test OpenAI provider end-to-end with real API (if key available)."""
        config = {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
        
        provider = create("openai", config=config, auto_initialize=True)
        
        # Test generation
        response = provider.generate("Say 'Hello' in one word.")
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        assert response.model == "gpt-4o-mini"
        
        # Test streaming
        chunks = list(provider.stream("Count to 3, one number per chunk."))
        assert len(chunks) > 0
        
        # Test cost estimation
        cost = provider.estimate_cost(10, 20)
        assert isinstance(cost, float)
        assert cost >= 0
    
    def test_provider_switching(self):
        """Test switching providers via configuration."""
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
            # Test creating provider from config
            if os.getenv("OPENAI_API_KEY"):
                provider = create(
                    "openai",
                    config_file=config_path,
                    auto_initialize=False
                )
                assert provider.provider_type == "openai"
        finally:
            os.unlink(config_path)
    
    def test_factory_pattern(self):
        """Test factory pattern usage."""
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }
        
        # Should be able to create provider without knowing implementation details
        provider = create("openai", config=config, auto_initialize=False)
        
        assert provider.provider_type == "openai"
        assert hasattr(provider, "generate")
        assert hasattr(provider, "stream")
        assert hasattr(provider, "get_models")
    
    def test_configuration_precedence(self):
        """Test that environment variables override config file."""
        config_data = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "type": "openai",
                    "default_model": "gpt-4",
                    "api_key_env": "OPENAI_API_KEY",
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
            
            from llm_provider.config import ProviderConfig
            config_manager = ProviderConfig(config_path)
            provider_config = config_manager.get_provider_config("openai", use_env=True)
            
            # Environment should override config file
            assert provider_config.get("default_model") == "gpt-4o-mini"
        finally:
            os.unlink(config_path)
            os.environ.pop("OPENAI_MODEL", None)
    
    def test_error_handling(self):
        """Test error handling across the system."""
        # Test invalid provider type
        with pytest.raises(ValueError):
            create("nonexistent-provider", config={}, auto_initialize=False)
        
        # Test missing API key
        config = {"model": "gpt-4o-mini"}
        os.environ.pop("OPENAI_API_KEY", None)
        
        provider = create("openai", config=config, auto_initialize=False)
        with pytest.raises(ValueError):
            provider.initialize()
    
    def test_provider_interface_compliance(self):
        """Test that all providers implement the base interface correctly."""
        from llm_provider.base import BaseLLMProvider
        
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }
        
        provider = create("openai", config=config, auto_initialize=False)
        
        # Verify all required methods exist
        assert isinstance(provider, BaseLLMProvider)
        assert hasattr(provider, "initialize")
        assert hasattr(provider, "generate")
        assert hasattr(provider, "stream")
        assert hasattr(provider, "stream_async")
        assert hasattr(provider, "get_models")
        assert hasattr(provider, "estimate_cost")
        assert hasattr(provider, "validate_credentials")
        assert hasattr(provider, "provider_type")
        assert hasattr(provider, "default_model")
