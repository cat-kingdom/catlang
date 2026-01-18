"""Transport layer for MCP Server.

This module handles transport abstraction for different communication
methods (stdio, SSE, WebSocket). Currently implements stdio transport.
"""

import asyncio
import logging
from typing import AsyncIterator, Tuple, Any, Optional

from mcp.server.stdio import stdio_server

logger = logging.getLogger(__name__)


class Transport:
    """Base transport class for MCP server communication."""
    
    def __init__(self, transport_type: str = "stdio"):
        """Initialize transport.
        
        Args:
            transport_type: Type of transport ("stdio", "sse", "websocket")
        """
        self.transport_type = transport_type
        logger.info(f"Initializing {transport_type} transport")
    
    async def start(self) -> Tuple[AsyncIterator[Any], Any]:
        """Start the transport and return read/write streams.
        
        Returns:
            Tuple of (read_stream, write_stream)
        """
        raise NotImplementedError("Subclasses must implement start()")
    
    async def stop(self) -> None:
        """Stop the transport and cleanup resources."""
        pass


class StdioTransport(Transport):
    """stdio transport implementation for MCP server.
    
    This transport uses stdin/stdout for communication, which is
    the standard way to communicate with Claude Desktop and other
    MCP clients.
    """
    
    def __init__(self):
        """Initialize stdio transport."""
        super().__init__("stdio")
        self._stdio_context = None
    
    async def start(self) -> Tuple[AsyncIterator[Any], Any]:
        """Start stdio transport.
        
        Returns:
            Tuple of (read_stream, write_stream) from stdio_server
        """
        logger.info("Starting stdio transport...")
        self._stdio_context = stdio_server()
        read_stream, write_stream = await self._stdio_context.__aenter__()
        logger.info("✓ stdio transport started")
        return read_stream, write_stream
    
    async def stop(self) -> None:
        """Stop stdio transport and cleanup."""
        if self._stdio_context:
            try:
                await self._stdio_context.__aexit__(None, None, None)
                logger.info("✓ stdio transport stopped")
            except Exception as e:
                logger.error(f"Error stopping stdio transport: {e}")
            finally:
                self._stdio_context = None


class SSETransport(Transport):
    """SSE (Server-Sent Events) transport implementation.
    
    This transport is not implemented yet (planned for post-MVP).
    """
    
    def __init__(self, port: int = 8000, host: str = "localhost"):
        """Initialize SSE transport.
        
        Args:
            port: Port to listen on
            host: Host to bind to
        """
        super().__init__("sse")
        self.port = port
        self.host = host
    
    async def start(self) -> Tuple[AsyncIterator[Any], Any]:
        """Start SSE transport."""
        raise NotImplementedError("SSE transport not implemented yet (post-MVP)")
    
    async def stop(self) -> None:
        """Stop SSE transport."""
        raise NotImplementedError("SSE transport not implemented yet (post-MVP)")


class WebSocketTransport(Transport):
    """WebSocket transport implementation.
    
    This transport is not implemented yet (planned for post-MVP).
    """
    
    def __init__(self, port: int = 8001, host: str = "localhost"):
        """Initialize WebSocket transport.
        
        Args:
            port: Port to listen on
            host: Host to bind to
        """
        super().__init__("websocket")
        self.port = port
        self.host = host
    
    async def start(self) -> Tuple[AsyncIterator[Any], Any]:
        """Start WebSocket transport."""
        raise NotImplementedError("WebSocket transport not implemented yet (post-MVP)")
    
    async def stop(self) -> None:
        """Stop WebSocket transport."""
        raise NotImplementedError("WebSocket transport not implemented yet (post-MVP)")


def create_transport(transport_type: str, **kwargs: Any) -> Transport:
    """Factory function to create a transport instance.
    
    Args:
        transport_type: Type of transport ("stdio", "sse", "websocket")
        **kwargs: Additional arguments for transport initialization
        
    Returns:
        Transport instance
        
    Raises:
        ValueError: If transport_type is not supported
    """
    if transport_type == "stdio":
        return StdioTransport()
    elif transport_type == "sse":
        return SSETransport(**kwargs)
    elif transport_type == "websocket":
        return WebSocketTransport(**kwargs)
    else:
        raise ValueError(
            f"Unsupported transport type: {transport_type}. "
            "Supported types: stdio, sse, websocket"
        )
