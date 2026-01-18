"""
Unit tests for LLM provider configuration management.
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path
from llm_provider.config import ProviderConfig, load_provider_config, get_provider_from_env


class TestProviderConfig:
    """Test ProviderConfig class."""
    
    def test_load_config_file_exists(self):
        """Test loading configuration from existing file."""
        # Create temporary config file
        config_data = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "type": "openai",
                    "default_model": "gpt-4",
                }
            },
            "default_provider": "openai"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = ProviderConfig(config_path)
            assert config.get_default_provider() == "openai"
            assert config.is_provider_enabled("openai")
        finally:
            os.unlink(config_path)
    
    def test_load_config_file_not_exists(self):
        """Test loading configuration when file doesn't exist."""
        config = ProviderConfig("/nonexistent/config.yaml")
        # Should not raise error, just use empty config
        assert config.get_default_provider() == "openai"  # Default fallback
    
    def test_get_provider_config(self):
        """Test getting provider configuration."""
        config_data = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "type": "openai",
                    "default_model": "gpt-4",
                    "timeout": 60,
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = ProviderConfig(config_path)
            provider_config = config.get_provider_config("openai")
            
            assert provider_config["type"] == "openai"
            assert provider_config["default_model"] == "gpt-4"
            assert provider_config["timeout"] == 60
        finally:
            os.unlink(config_path)
    
    def test_env_overrides(self):
        """Test environment variable overrides."""
        config_data = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "type": "openai",
                    "default_model": "gpt-4",
                    "timeout": 60,
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Set environment variable
            os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
            os.environ["OPENAI_TIMEOUT"] = "120"
            
            config = ProviderConfig(config_path)
            provider_config = config.get_provider_config("openai", use_env=True)
            
            assert provider_config["default_model"] == "gpt-4o-mini"
            assert provider_config["timeout"] == 120
        finally:
            os.unlink(config_path)
            # Clean up environment
            os.environ.pop("OPENAI_MODEL", None)
            os.environ.pop("OPENAI_TIMEOUT", None)
    
    def test_get_default_provider_from_env(self):
        """Test getting default provider from environment."""
        os.environ["LLM_PROVIDER"] = "anthropic"
        
        try:
            config = ProviderConfig()
            assert config.get_default_provider() == "anthropic"
        finally:
            os.environ.pop("LLM_PROVIDER", None)
    
    def test_is_provider_enabled(self):
        """Test checking if provider is enabled."""
        config_data = {
            "providers": {
                "openai": {"enabled": True},
                "anthropic": {"enabled": False},
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = ProviderConfig(config_path)
            assert config.is_provider_enabled("openai")
            assert not config.is_provider_enabled("anthropic")
        finally:
            os.unlink(config_path)
    
    def test_validate_provider_config_valid(self):
        """Test validating valid provider configuration."""
        config_data = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "type": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = ProviderConfig(config_path)
            is_valid, error = config.validate_provider_config("openai")
            
            assert is_valid
            assert error is None
        finally:
            os.unlink(config_path)
    
    def test_validate_provider_config_missing(self):
        """Test validating missing provider."""
        config = ProviderConfig()
        is_valid, error = config.validate_provider_config("nonexistent")
        
        assert not is_valid
        assert "not found" in error.lower()
    
    def test_validate_provider_config_disabled(self):
        """Test validating disabled provider."""
        config_data = {
            "providers": {
                "openai": {"enabled": False}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = ProviderConfig(config_path)
            is_valid, error = config.validate_provider_config("openai")
            
            assert not is_valid
            assert "not enabled" in error.lower()
        finally:
            os.unlink(config_path)
    
    def test_get_all_providers(self):
        """Test getting all providers."""
        config_data = {
            "providers": {
                "openai": {"enabled": True},
                "anthropic": {"enabled": False},
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = ProviderConfig(config_path)
            providers = config.get_all_providers()
            
            assert "openai" in providers
            assert "anthropic" in providers
        finally:
            os.unlink(config_path)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_load_provider_config(self):
        """Test load_provider_config function."""
        config_data = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "type": "openai",
                    "default_model": "gpt-4",
                }
            },
            "default_provider": "openai"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_provider_config("openai", config_path)
            assert config["type"] == "openai"
        finally:
            os.unlink(config_path)
    
    def test_get_provider_from_env(self):
        """Test get_provider_from_env function."""
        os.environ["LLM_PROVIDER"] = "test-provider"
        
        try:
            provider = get_provider_from_env()
            assert provider == "test-provider"
        finally:
            os.environ.pop("LLM_PROVIDER", None)
        
        # Test when not set
        provider = get_provider_from_env()
        assert provider is None
