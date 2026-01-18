"""
Base provider interface for LLM providers.

This module defines the abstract base class that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncIterator, Iterator
from dataclasses import dataclass


@dataclass
class GenerationParams:
    """Parameters for LLM generation."""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    # Additional provider-specific parameters
    extra_params: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResponse:
    """Response from LLM generation."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None  # e.g., {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM providers must implement this interface to ensure consistent
    behavior across different providers (OpenAI, Anthropic, Google, etc.).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the provider (e.g., validate credentials, setup client).
        
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If initialization fails
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None
    ) -> GenerationResponse:
        """
        Generate a response from the LLM synchronously.
        
        Args:
            prompt: Input prompt text
            params: Optional generation parameters
            
        Returns:
            GenerationResponse with generated content and metadata
            
        Raises:
            RuntimeError: If provider is not initialized
            ValueError: If prompt is invalid
            Exception: Provider-specific errors
        """
        pass
    
    @abstractmethod
    def stream(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None
    ) -> Iterator[str]:
        """
        Stream a response from the LLM synchronously.
        
        Args:
            prompt: Input prompt text
            params: Optional generation parameters
            
        Yields:
            Chunks of generated text as strings
            
        Raises:
            RuntimeError: If provider is not initialized
            ValueError: If prompt is invalid
            Exception: Provider-specific errors
        """
        pass
    
    @abstractmethod
    async def stream_async(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None
    ) -> AsyncIterator[str]:
        """
        Stream a response from the LLM asynchronously.
        
        Args:
            prompt: Input prompt text
            params: Optional generation parameters
            
        Yields:
            Chunks of generated text as strings
            
        Raises:
            RuntimeError: If provider is not initialized
            ValueError: If prompt is invalid
            Exception: Provider-specific errors
        """
        pass
    
    @abstractmethod
    def get_models(self) -> List[str]:
        """
        Get list of available models for this provider.
        
        Returns:
            List of model names/identifiers
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """
        Validate that credentials are correct and provider is accessible.
        
        Returns:
            True if credentials are valid, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def provider_type(self) -> str:
        """
        Get the provider type identifier (e.g., 'openai', 'anthropic').
        
        Returns:
            Provider type string
        """
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """
        Get the default model for this provider.
        
        Returns:
            Default model name
        """
        pass
    
    def is_initialized(self) -> bool:
        """
        Check if provider has been initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        return self._initialized
