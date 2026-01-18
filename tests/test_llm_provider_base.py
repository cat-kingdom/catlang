"""
Unit tests for base LLM provider interface.
"""

import pytest
from unittest.mock import Mock, MagicMock
from llm_provider.base import (
    BaseLLMProvider,
    GenerationParams,
    GenerationResponse,
)


class MockProvider(BaseLLMProvider):
    """Mock provider for testing base interface."""
    
    def initialize(self):
        self._initialized = True
    
    def generate(self, prompt: str, params=None):
        if not self._initialized:
            raise RuntimeError("Not initialized")
        return GenerationResponse(
            content=f"Mock response to: {prompt}",
            model="mock-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
    
    def stream(self, prompt: str, params=None):
        if not self._initialized:
            raise RuntimeError("Not initialized")
        yield f"Mock stream: {prompt}"
    
    async def stream_async(self, prompt: str, params=None):
        if not self._initialized:
            raise RuntimeError("Not initialized")
        yield f"Mock async stream: {prompt}"
    
    def get_models(self):
        return ["mock-model-1", "mock-model-2"]
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model=None):
        return (prompt_tokens + completion_tokens) * 0.001
    
    def validate_credentials(self):
        return True
    
    @property
    def provider_type(self):
        return "mock"
    
    @property
    def default_model(self):
        return "mock-model"


class TestBaseLLMProvider:
    """Test base provider interface."""
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = MockProvider({"test": "config"})
        assert not provider.is_initialized()
        assert provider.config == {"test": "config"}
    
    def test_initialize(self):
        """Test initialize method."""
        provider = MockProvider({})
        provider.initialize()
        assert provider.is_initialized()
    
    def test_generate_not_initialized(self):
        """Test generate fails when not initialized."""
        provider = MockProvider({})
        with pytest.raises(RuntimeError, match="Not initialized"):
            provider.generate("test prompt")
    
    def test_generate_success(self):
        """Test successful generation."""
        provider = MockProvider({})
        provider.initialize()
        response = provider.generate("test prompt")
        
        assert isinstance(response, GenerationResponse)
        assert "test prompt" in response.content
        assert response.model == "mock-model"
        assert response.usage is not None
    
    def test_stream_not_initialized(self):
        """Test stream fails when not initialized."""
        provider = MockProvider({})
        with pytest.raises(RuntimeError, match="Not initialized"):
            list(provider.stream("test prompt"))
    
    def test_stream_success(self):
        """Test successful streaming."""
        provider = MockProvider({})
        provider.initialize()
        chunks = list(provider.stream("test prompt"))
        
        assert len(chunks) > 0
        assert "test prompt" in chunks[0]
    
    @pytest.mark.asyncio
    async def test_stream_async_success(self):
        """Test successful async streaming."""
        provider = MockProvider({})
        provider.initialize()
        chunks = [chunk async for chunk in provider.stream_async("test prompt")]
        
        assert len(chunks) > 0
        assert "test prompt" in chunks[0]
    
    def test_get_models(self):
        """Test get_models method."""
        provider = MockProvider({})
        models = provider.get_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        provider = MockProvider({})
        cost = provider.estimate_cost(100, 200)
        
        assert isinstance(cost, float)
        assert cost > 0
    
    def test_validate_credentials(self):
        """Test credential validation."""
        provider = MockProvider({})
        result = provider.validate_credentials()
        
        assert isinstance(result, bool)
    
    def test_provider_type(self):
        """Test provider_type property."""
        provider = MockProvider({})
        assert provider.provider_type == "mock"
    
    def test_default_model(self):
        """Test default_model property."""
        provider = MockProvider({})
        assert provider.default_model == "mock-model"


class TestGenerationParams:
    """Test GenerationParams dataclass."""
    
    def test_default_params(self):
        """Test default parameters."""
        params = GenerationParams()
        assert params.temperature == 0.7
        assert params.max_tokens is None
        assert params.top_p is None
    
    def test_custom_params(self):
        """Test custom parameters."""
        params = GenerationParams(
            temperature=0.9,
            max_tokens=100,
            top_p=0.95
        )
        assert params.temperature == 0.9
        assert params.max_tokens == 100
        assert params.top_p == 0.95
    
    def test_extra_params(self):
        """Test extra parameters."""
        params = GenerationParams(extra_params={"custom": "value"})
        assert params.extra_params == {"custom": "value"}


class TestGenerationResponse:
    """Test GenerationResponse dataclass."""
    
    def test_basic_response(self):
        """Test basic response creation."""
        response = GenerationResponse(
            content="Test content",
            model="test-model"
        )
        assert response.content == "Test content"
        assert response.model == "test-model"
        assert response.usage is None
    
    def test_response_with_usage(self):
        """Test response with usage information."""
        usage = {"prompt_tokens": 10, "completion_tokens": 20}
        response = GenerationResponse(
            content="Test",
            model="test-model",
            usage=usage
        )
        assert response.usage == usage
