"""
Unit tests for OpenAI provider implementation.

Note: These tests may require valid OpenAI API key for full integration testing.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from llm_provider.openai import OpenAIProvider
from llm_provider.base import GenerationParams, GenerationResponse


class TestOpenAIProvider:
    """Test OpenAIProvider class."""
    
    def test_initialization(self):
        """Test provider initialization."""
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }
        
        provider = OpenAIProvider(config)
        assert provider._model == "gpt-4o-mini"
        assert provider._api_key == "test-key"
        assert not provider.is_initialized()
    
    def test_initialization_from_env(self):
        """Test initialization with API key from environment."""
        os.environ["OPENAI_API_KEY"] = "env-key"
        
        try:
            config = {"model": "gpt-4o-mini"}
            provider = OpenAIProvider(config)
            assert provider._api_key == "env-key"
        finally:
            os.environ.pop("OPENAI_API_KEY", None)
    
    def test_initialization_missing_api_key(self):
        """Test initialization fails without API key."""
        config = {"model": "gpt-4o-mini"}
        
        # Remove API key from env if present
        os.environ.pop("OPENAI_API_KEY", None)
        
        provider = OpenAIProvider(config)
        
        with pytest.raises(ValueError, match="API key"):
            provider.initialize()
    
    @patch('llm_provider.openai.ChatOpenAI')
    def test_initialize_success(self, mock_chat_openai):
        """Test successful initialization."""
        mock_client = MagicMock()
        mock_chat_openai.return_value = mock_client
        
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }
        
        provider = OpenAIProvider(config)
        provider.initialize()
        
        assert provider.is_initialized()
        mock_chat_openai.assert_called_once()
    
    @patch('llm_provider.openai.ChatOpenAI')
    def test_generate_success(self, mock_chat_openai):
        """Test successful generation."""
        # Mock response
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.response_metadata = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            },
            "finish_reason": "stop"
        }
        
        mock_client = MagicMock()
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client
        
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }
        
        provider = OpenAIProvider(config)
        provider.initialize()
        
        response = provider.generate("test prompt")
        
        assert isinstance(response, GenerationResponse)
        assert response.content == "Test response"
        assert response.model == "gpt-4o-mini"
        assert response.usage is not None
        mock_client.invoke.assert_called_once()
    
    @patch('llm_provider.openai.ChatOpenAI')
    def test_generate_not_initialized(self, mock_chat_openai):
        """Test generate fails when not initialized."""
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }
        
        provider = OpenAIProvider(config)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            provider.generate("test prompt")
    
    @patch('llm_provider.openai.ChatOpenAI')
    def test_generate_with_params(self, mock_chat_openai):
        """Test generation with custom parameters."""
        mock_response = MagicMock()
        mock_response.content = "Test"
        mock_response.response_metadata = {}
        
        mock_client = MagicMock()
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client
        
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }
        
        provider = OpenAIProvider(config)
        provider.initialize()
        
        params = GenerationParams(temperature=0.9, max_tokens=100)
        response = provider.generate("test", params=params)
        
        assert response.content == "Test"
        # Should create a new client with custom params
        assert mock_chat_openai.call_count >= 2
    
    @patch('llm_provider.openai.ChatOpenAI')
    def test_stream_success(self, mock_chat_openai):
        """Test successful streaming."""
        mock_chunk1 = MagicMock()
        mock_chunk1.content = "Hello"
        mock_chunk2 = MagicMock()
        mock_chunk2.content = " World"
        
        mock_client = MagicMock()
        mock_client.stream.return_value = [mock_chunk1, mock_chunk2]
        mock_chat_openai.return_value = mock_client
        
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }
        
        provider = OpenAIProvider(config)
        provider.initialize()
        
        chunks = list(provider.stream("test prompt"))
        
        assert len(chunks) == 2
        assert chunks[0] == "Hello"
        assert chunks[1] == " World"
    
    @pytest.mark.asyncio
    @patch('llm_provider.openai.ChatOpenAI')
    async def test_stream_async_success(self, mock_chat_openai):
        """Test successful async streaming."""
        async def async_generator():
            mock_chunk = MagicMock()
            mock_chunk.content = "Async"
            yield mock_chunk
        
        mock_client = MagicMock()
        mock_client.astream.return_value = async_generator()
        mock_chat_openai.return_value = mock_client
        
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }
        
        provider = OpenAIProvider(config)
        provider.initialize()
        
        chunks = [chunk async for chunk in provider.stream_async("test")]
        
        assert len(chunks) > 0
        assert chunks[0] == "Async"
    
    def test_get_models(self):
        """Test getting available models."""
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }
        
        provider = OpenAIProvider(config)
        models = provider.get_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-4o-mini" in models
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }
        
        provider = OpenAIProvider(config)
        cost = provider.estimate_cost(1000, 2000, "gpt-4o-mini")
        
        assert isinstance(cost, float)
        assert cost > 0
    
    def test_estimate_cost_unknown_model(self):
        """Test cost estimation with unknown model."""
        config = {
            "model": "unknown-model",
            "api_key": "test-key",
        }
        
        provider = OpenAIProvider(config)
        # Should fall back to default pricing
        cost = provider.estimate_cost(1000, 2000)
        
        assert isinstance(cost, float)
        assert cost > 0
    
    @patch('llm_provider.openai.ChatOpenAI')
    def test_validate_credentials_valid(self, mock_chat_openai):
        """Test credential validation with valid key."""
        mock_response = MagicMock()
        mock_response.content = "test"
        
        mock_client = MagicMock()
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client
        
        config = {
            "model": "gpt-4o-mini",
            "api_key": "valid-key",
        }
        
        provider = OpenAIProvider(config)
        result = provider.validate_credentials()
        
        assert result is True
    
    def test_validate_credentials_missing_key(self):
        """Test credential validation without API key."""
        config = {"model": "gpt-4o-mini"}
        os.environ.pop("OPENAI_API_KEY", None)
        
        provider = OpenAIProvider(config)
        result = provider.validate_credentials()
        
        assert result is False
    
    @patch('llm_provider.openai.ChatOpenAI')
    def test_validate_credentials_invalid(self, mock_chat_openai):
        """Test credential validation with invalid key."""
        mock_client = MagicMock()
        mock_client.invoke.side_effect = Exception("Invalid API key")
        mock_chat_openai.return_value = mock_client
        
        config = {
            "model": "gpt-4o-mini",
            "api_key": "invalid-key",
        }
        
        provider = OpenAIProvider(config)
        result = provider.validate_credentials()
        
        assert result is False
    
    def test_provider_type(self):
        """Test provider_type property."""
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }
        
        provider = OpenAIProvider(config)
        assert provider.provider_type == "openai"
    
    def test_default_model(self):
        """Test default_model property."""
        config = {
            "model": "gpt-4",
            "api_key": "test-key",
        }
        
        provider = OpenAIProvider(config)
        assert provider.default_model == "gpt-4"
    
    def test_generate_empty_prompt(self):
        """Test generate with empty prompt."""
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }
        
        provider = OpenAIProvider(config)
        provider.initialize()
        
        with pytest.raises(ValueError, match="empty"):
            provider.generate("")
        
        with pytest.raises(ValueError, match="empty"):
            provider.generate("   ")
