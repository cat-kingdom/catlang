"""
OpenRouter provider implementation.

OpenRouter provides access to multiple LLM models through an OpenAI-compatible API.
This provider uses langchain_openai.ChatOpenAI with OpenRouter's base URL.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Iterator, AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration

from .base import BaseLLMProvider, GenerationParams, GenerationResponse

logger = logging.getLogger(__name__)

# OpenRouter base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Common OpenRouter models and their approximate pricing
# Note: Pricing varies, check https://openrouter.ai/models for current rates
OPENROUTER_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "openai/gpt-3.5-turbo",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-haiku",
    "google/gemini-pro",
    "x-ai/grok-beta",
    "x-ai/grok-2",
    "meta-llama/llama-3.1-70b-instruct",
    "mistralai/mistral-large",
]


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter LLM provider implementation.
    
    OpenRouter provides access to multiple LLM models through an OpenAI-compatible API.
    This provider wraps langchain_openai.ChatOpenAI with OpenRouter's base URL.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenRouter provider.
        
        Args:
            config: Configuration dictionary with keys:
                - api_key: OpenRouter API key (or use OPENROUTER_API_KEY env var)
                - model: Model name (e.g., "openai/gpt-4o", "x-ai/grok-beta")
                - default_model: Default model name
                - timeout: Request timeout in seconds (default: 60)
                - max_retries: Maximum retry attempts (default: 3)
                - temperature: Default temperature (default: 0.7)
        """
        super().__init__(config)
        self._client: Optional[ChatOpenAI] = None
        self._model: str = config.get("model", config.get("default_model", "openai/gpt-4o-mini"))
        self._api_key: Optional[str] = config.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        self._timeout: int = config.get("timeout", 60)
        self._max_retries: int = config.get("max_retries", 3)
        self._default_temperature: float = config.get("temperature", 0.7)
        
        # OpenRouter requires HTTP Referer header (optional but recommended)
        self._http_headers: Dict[str, str] = config.get("http_headers", {})
        if "HTTP-Referer" not in self._http_headers:
            # Use default referer if not provided
            self._http_headers["HTTP-Referer"] = config.get("referer", "https://github.com/catlang")
        if "X-Title" not in self._http_headers:
            self._http_headers["X-Title"] = config.get("app_name", "CatLang")
    
    def initialize(self) -> None:
        """
        Initialize the OpenRouter client.
        
        Raises:
            ValueError: If API key is missing or invalid
            RuntimeError: If initialization fails
        """
        if not self._api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Set it in config['api_key'] or OPENROUTER_API_KEY environment variable."
            )
        
        try:
            client_kwargs = {
                "model": self._model,
                "api_key": self._api_key,
                "base_url": OPENROUTER_BASE_URL,
                "temperature": self._default_temperature,
                "timeout": self._timeout,
                "max_retries": self._max_retries,
                "default_headers": self._http_headers,
            }
            
            self._client = ChatOpenAI(**client_kwargs)
            self._initialized = True
            logger.info(f"OpenRouter provider initialized with model: {self._model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter provider: {e}")
            raise RuntimeError(f"OpenRouter initialization failed: {e}") from e
    
    def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None
    ) -> GenerationResponse:
        """Generate a response from OpenRouter synchronously."""
        if not self._initialized or not self._client:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            # Prepare generation parameters
            generation_kwargs = {}
            if params:
                if params.temperature is not None:
                    generation_kwargs["temperature"] = params.temperature
                if params.max_tokens is not None:
                    generation_kwargs["max_tokens"] = params.max_tokens
                if params.top_p is not None:
                    generation_kwargs["top_p"] = params.top_p
                if params.frequency_penalty is not None:
                    generation_kwargs["frequency_penalty"] = params.frequency_penalty
                if params.presence_penalty is not None:
                    generation_kwargs["presence_penalty"] = params.presence_penalty
                if params.stop is not None:
                    generation_kwargs["stop"] = params.stop
                if params.extra_params:
                    generation_kwargs.update(params.extra_params)
            
            # Create a temporary client with generation parameters if needed
            if generation_kwargs:
                temp_client = ChatOpenAI(
                    model=self._model,
                    api_key=self._api_key,
                    base_url=OPENROUTER_BASE_URL,
                    timeout=self._timeout,
                    max_retries=self._max_retries,
                    default_headers=self._http_headers,
                    **generation_kwargs
                )
            else:
                temp_client = self._client
            
            # Invoke the model
            response = temp_client.invoke(prompt)
            
            # Extract content
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract usage information if available
            usage = None
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if metadata and 'token_usage' in metadata:
                    usage = metadata['token_usage']
            
            # Extract finish reason if available
            finish_reason = None
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if metadata:
                    finish_reason = metadata.get('finish_reason')
            
            return GenerationResponse(
                content=content,
                model=self._model,
                usage=usage,
                finish_reason=finish_reason,
                metadata={"provider": "openrouter", "base_url": OPENROUTER_BASE_URL}
            )
            
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            raise
    
    def stream(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None
    ) -> Iterator[str]:
        """Stream a response from OpenRouter synchronously."""
        if not self._initialized or not self._client:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            # Prepare generation parameters
            generation_kwargs = {}
            if params:
                if params.temperature is not None:
                    generation_kwargs["temperature"] = params.temperature
                if params.max_tokens is not None:
                    generation_kwargs["max_tokens"] = params.max_tokens
                if params.top_p is not None:
                    generation_kwargs["top_p"] = params.top_p
                if params.extra_params:
                    generation_kwargs.update(params.extra_params)
            
            # Create a temporary client with generation parameters if needed
            if generation_kwargs:
                temp_client = ChatOpenAI(
                    model=self._model,
                    api_key=self._api_key,
                    base_url=OPENROUTER_BASE_URL,
                    timeout=self._timeout,
                    max_retries=self._max_retries,
                    default_headers=self._http_headers,
                    **generation_kwargs
                )
            else:
                temp_client = self._client
            
            # Stream the response
            for chunk in temp_client.stream(prompt):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
                    
        except Exception as e:
            logger.error(f"OpenRouter streaming failed: {e}")
            raise
    
    async def stream_async(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None
    ) -> AsyncIterator[str]:
        """Stream a response from OpenRouter asynchronously."""
        if not self._initialized or not self._client:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            # Prepare generation parameters
            generation_kwargs = {}
            if params:
                if params.temperature is not None:
                    generation_kwargs["temperature"] = params.temperature
                if params.max_tokens is not None:
                    generation_kwargs["max_tokens"] = params.max_tokens
                if params.top_p is not None:
                    generation_kwargs["top_p"] = params.top_p
                if params.extra_params:
                    generation_kwargs.update(params.extra_params)
            
            # Create a temporary client with generation parameters if needed
            if generation_kwargs:
                temp_client = ChatOpenAI(
                    model=self._model,
                    api_key=self._api_key,
                    base_url=OPENROUTER_BASE_URL,
                    timeout=self._timeout,
                    max_retries=self._max_retries,
                    default_headers=self._http_headers,
                    **generation_kwargs
                )
            else:
                temp_client = self._client
            
            # Stream the response asynchronously
            async for chunk in temp_client.astream(prompt):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
                    
        except Exception as e:
            logger.error(f"OpenRouter async streaming failed: {e}")
            raise
    
    def get_models(self) -> List[str]:
        """
        Get list of available OpenRouter models.
        
        Returns:
            List of OpenRouter model names
        """
        return OPENROUTER_MODELS.copy()
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None
    ) -> float:
        """
        Estimate the cost for token usage.
        
        Note: OpenRouter pricing varies by model and changes frequently.
        This is a rough estimate. Check https://openrouter.ai/models for accurate pricing.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Optional model name (uses default if not provided)
            
        Returns:
            Estimated cost in USD (approximate)
        """
        # OpenRouter pricing is dynamic and varies by model
        # Return 0.0 as we can't accurately estimate without querying their API
        logger.warning(
            "OpenRouter pricing varies by model. "
            "Cannot provide accurate cost estimate. "
            "Check https://openrouter.ai/models for current pricing."
        )
        return 0.0
    
    def validate_credentials(self) -> bool:
        """
        Validate OpenRouter credentials by making a test API call.
        
        Returns:
            True if credentials are valid, False otherwise
        """
        if not self._api_key:
            return False
        
        try:
            # Make a minimal test call
            test_client = ChatOpenAI(
                model="openai/gpt-4o-mini",
                api_key=self._api_key,
                base_url=OPENROUTER_BASE_URL,
                timeout=10,
                max_tokens=5,
                default_headers=self._http_headers,
            )
            test_client.invoke("test")
            return True
        except Exception as e:
            logger.debug(f"Credential validation failed: {e}")
            return False
    
    @property
    def provider_type(self) -> str:
        """Get the provider type identifier."""
        return "openrouter"
    
    @property
    def default_model(self) -> str:
        """Get the default model."""
        return self._model
    
    def get_langchain_client(self) -> ChatOpenAI:
        """
        Get the underlying LangChain ChatOpenAI client.
        
        Returns:
            The ChatOpenAI client instance configured for OpenRouter
            
        Raises:
            RuntimeError: If provider is not initialized
        """
        if not self._initialized or not self._client:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        return self._client
