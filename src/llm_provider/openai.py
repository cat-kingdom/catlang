"""
OpenAI provider implementation.

This module implements the OpenAI LLM provider using langchain_openai.ChatOpenAI.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Iterator, AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration

from .base import BaseLLMProvider, GenerationParams, GenerationResponse

logger = logging.getLogger(__name__)


# OpenAI pricing per 1M tokens (as of 2024)
# Format: {model_name: {"prompt": price_per_1M_tokens, "completion": price_per_1M_tokens}}
OPENAI_PRICING = {
    "gpt-4": {"prompt": 30.0, "completion": 60.0},
    "gpt-4-turbo": {"prompt": 10.0, "completion": 30.0},
    "gpt-4-turbo-preview": {"prompt": 10.0, "completion": 30.0},
    "gpt-4o": {"prompt": 5.0, "completion": 15.0},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
    "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
    "gpt-3.5-turbo-16k": {"prompt": 3.0, "completion": 4.0},
}


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider implementation.
    
    Wraps langchain_openai.ChatOpenAI to provide a consistent interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI provider.
        
        Args:
            config: Configuration dictionary with keys:
                - api_key: OpenAI API key (or use OPENAI_API_KEY env var)
                - model: Model name (default: "gpt-4o-mini")
                - base_url: Optional custom base URL
                - timeout: Request timeout in seconds (default: 60)
                - max_retries: Maximum retry attempts (default: 3)
                - temperature: Default temperature (default: 0.7)
        """
        super().__init__(config)
        self._client: Optional[ChatOpenAI] = None
        self._model: str = config.get("model", config.get("default_model", "gpt-4o-mini"))
        self._api_key: Optional[str] = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self._base_url: Optional[str] = config.get("base_url")
        self._timeout: int = config.get("timeout", 60)
        self._max_retries: int = config.get("max_retries", 3)
        self._default_temperature: float = config.get("temperature", 0.7)
    
    def initialize(self) -> None:
        """
        Initialize the OpenAI client.
        
        Raises:
            ValueError: If API key is missing or invalid
            RuntimeError: If initialization fails
        """
        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set it in config['api_key'] or OPENAI_API_KEY environment variable."
            )
        
        try:
            client_kwargs = {
                "model": self._model,
                "api_key": self._api_key,
                "temperature": self._default_temperature,
                "timeout": self._timeout,
                "max_retries": self._max_retries,
            }
            
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            
            self._client = ChatOpenAI(**client_kwargs)
            self._initialized = True
            logger.info(f"OpenAI provider initialized with model: {self._model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            raise RuntimeError(f"OpenAI initialization failed: {e}") from e
    
    def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None
    ) -> GenerationResponse:
        """
        Generate a response from OpenAI synchronously.
        
        Args:
            prompt: Input prompt text
            params: Optional generation parameters
            
        Returns:
            GenerationResponse with generated content and metadata
            
        Raises:
            RuntimeError: If provider is not initialized
            ValueError: If prompt is invalid
        """
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
                    base_url=self._base_url,
                    timeout=self._timeout,
                    max_retries=self._max_retries,
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
                metadata={"provider": "openai"}
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def stream(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None
    ) -> Iterator[str]:
        """
        Stream a response from OpenAI synchronously.
        
        Args:
            prompt: Input prompt text
            params: Optional generation parameters
            
        Yields:
            Chunks of generated text as strings
        """
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
                    base_url=self._base_url,
                    timeout=self._timeout,
                    max_retries=self._max_retries,
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
            logger.error(f"OpenAI streaming failed: {e}")
            raise
    
    async def stream_async(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None
    ) -> AsyncIterator[str]:
        """
        Stream a response from OpenAI asynchronously.
        
        Args:
            prompt: Input prompt text
            params: Optional generation parameters
            
        Yields:
            Chunks of generated text as strings
        """
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
                    base_url=self._base_url,
                    timeout=self._timeout,
                    max_retries=self._max_retries,
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
            logger.error(f"OpenAI async streaming failed: {e}")
            raise
    
    def get_models(self) -> List[str]:
        """
        Get list of available OpenAI models.
        
        Returns:
            List of OpenAI model names
        """
        return list(OPENAI_PRICING.keys())
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None
    ) -> float:
        """
        Estimate the cost for token usage.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Optional model name (uses default if not provided)
            
        Returns:
            Estimated cost in USD
        """
        model_name = model or self._model
        
        # Normalize model name (handle variations)
        if model_name not in OPENAI_PRICING:
            # Try to find a matching model
            for key in OPENAI_PRICING.keys():
                if key in model_name or model_name in key:
                    model_name = key
                    break
            else:
                # Default to gpt-4o-mini pricing if unknown
                logger.warning(f"Unknown model {model}, using gpt-4o-mini pricing")
                model_name = "gpt-4o-mini"
        
        pricing = OPENAI_PRICING[model_name]
        prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
        
        return prompt_cost + completion_cost
    
    def validate_credentials(self) -> bool:
        """
        Validate OpenAI credentials by making a test API call.
        
        Returns:
            True if credentials are valid, False otherwise
        """
        if not self._api_key:
            return False
        
        try:
            # Make a minimal test call
            test_client = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=self._api_key,
                timeout=10,
                max_tokens=5
            )
            test_client.invoke("test")
            return True
        except Exception as e:
            logger.debug(f"Credential validation failed: {e}")
            return False
    
    @property
    def provider_type(self) -> str:
        """Get the provider type identifier."""
        return "openai"
    
    @property
    def default_model(self) -> str:
        """Get the default model."""
        return self._model
    
    def get_langchain_client(self) -> ChatOpenAI:
        """
        Get the underlying LangChain ChatOpenAI client.
        
        This method is provided for backward compatibility with code that
        expects a LangChain ChatModel instance (e.g., LangGraph workflows).
        
        Returns:
            The ChatOpenAI client instance
            
        Raises:
            RuntimeError: If provider is not initialized
        """
        if not self._initialized or not self._client:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        return self._client