"""Context management for MCP Server handlers.

This module provides async-safe context management using contextvars,
eliminating the need for global state variables.
"""

import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import MCPServer

logger = logging.getLogger(__name__)

# Context variable untuk store server instance
_server_context: ContextVar["MCPServer" | None] = ContextVar(
    "server_context", default=None
)


class HandlerContext:
    """Context manager untuk handler functions.
    
    Provides access to server dependencies without global state.
    """

    @staticmethod
    def set(server: "MCPServer") -> None:
        """Set the server instance in context.
        
        Args:
            server: MCPServer instance
        """
        _server_context.set(server)
        logger.debug("Server context set")

    @staticmethod
    def get() -> "MCPServer | None":
        """Get the server instance from context.
        
        Returns:
            MCPServer instance if set, None otherwise
        """
        return _server_context.get()

    @staticmethod
    def get_llm_provider():
        """Get LLM provider from server context.
        
        Returns:
            LLM provider instance
            
        Raises:
            RuntimeError: If server context is not set or provider unavailable
        """
        server = _server_context.get()
        if server is None:
            raise RuntimeError(
                "Server context not set. Cannot access LLM provider."
            )
        
        try:
            return server._get_llm_provider()
        except Exception as e:
            raise RuntimeError(
                f"Failed to get LLM provider from server: {e}"
            ) from e

    @staticmethod
    def get_resource_manager():
        """Get resource manager from server context.
        
        Returns:
            GuideResourceManager instance
            
        Raises:
            RuntimeError: If server context is not set or resource manager unavailable
        """
        server = _server_context.get()
        if server is None:
            raise RuntimeError(
                "Server context not set. Cannot access resource manager."
            )
        
        if not hasattr(server, "resource_manager"):
            raise RuntimeError(
                "Server does not have resource_manager attribute"
            )
        
        return server.resource_manager


def get_server() -> "MCPServer | None":
    """Get server instance from context (convenience function).
    
    Returns:
        MCPServer instance if set, None otherwise
    """
    return HandlerContext.get()


def get_llm_provider():
    """Get LLM provider from context (convenience function).
    
    Returns:
        LLM provider instance
        
    Raises:
        RuntimeError: If context not set or provider unavailable
    """
    return HandlerContext.get_llm_provider()


def get_resource_manager():
    """Get resource manager from context (convenience function).
    
    Returns:
        GuideResourceManager instance
        
    Raises:
        RuntimeError: If context not set or resource manager unavailable
    """
    return HandlerContext.get_resource_manager()
