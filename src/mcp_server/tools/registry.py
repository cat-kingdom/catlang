"""Tool registry for MCP Server.

This module implements the tool registration system that manages tool discovery,
versioning, and routing.
"""

import logging
from typing import Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    
    name: str
    version: str
    handler: Callable
    schema: dict[str, Any]
    enabled: bool = True
    description: str | None = None
    tags: list[str] | None = None


class ToolRegistry:
    """Registry for managing MCP tools.
    
    This registry provides:
    - Tool registration and discovery
    - Tool versioning support
    - Tool metadata management
    - Tool routing mechanism
    """
    
    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: dict[str, ToolMetadata] = {}
        self._versions: dict[str, list[str]] = {}  # name -> [versions]
        logger.info("Tool registry initialized")
    
    def register(
        self,
        name: str,
        handler: Callable,
        schema: dict[str, Any],
        version: str = "1.0.0",
        description: str | None = None,
        tags: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        """Register a tool in the registry.
        
        Args:
            name: Tool name (must be unique)
            handler: Callable function that implements the tool
            schema: JSON Schema for the tool parameters
            version: Tool version (default: "1.0.0")
            description: Tool description
            tags: List of tags for categorization
            enabled: Whether the tool is enabled (default: True)
            
        Raises:
            ValueError: If tool name already exists
        """
        if name in self._tools:
            existing = self._tools[name]
            if existing.version == version:
                logger.warning(
                    f"Tool '{name}' v{version} already registered. "
                    "Skipping duplicate registration."
                )
                return
            else:
                raise ValueError(
                    f"Tool '{name}' already registered with version "
                    f"{existing.version}. Cannot register version {version}."
                )
        
        metadata = ToolMetadata(
            name=name,
            version=version,
            handler=handler,
            schema=schema,
            enabled=enabled,
            description=description or schema.get("description", ""),
            tags=tags or [],
        )
        
        self._tools[name] = metadata
        
        # Track versions
        if name not in self._versions:
            self._versions[name] = []
        self._versions[name].append(version)
        
        logger.info(
            f"Registered tool: {name} v{version} "
            f"(enabled={enabled})"
        )
    
    def get(self, name: str) -> ToolMetadata | None:
        """Get tool metadata by name.
        
        Args:
            name: Tool name
            
        Returns:
            ToolMetadata if found, None otherwise
        """
        return self._tools.get(name)
    
    def get_handler(self, name: str) -> Callable | None:
        """Get tool handler function by name.
        
        Args:
            name: Tool name
            
        Returns:
            Handler function if found and enabled, None otherwise
        """
        tool = self._tools.get(name)
        if tool and tool.enabled:
            return tool.handler
        return None
    
    def list_tools(self, enabled_only: bool = True) -> list[str]:
        """List all registered tool names.
        
        Args:
            enabled_only: If True, only return enabled tools
            
        Returns:
            List of tool names
        """
        if enabled_only:
            return [
                name for name, tool in self._tools.items()
                if tool.enabled
            ]
        return list(self._tools.keys())
    
    def get_all_schemas(self) -> dict[str, dict[str, Any]]:
        """Get all tool schemas.
        
        Returns:
            Dictionary mapping tool names to their schemas
        """
        return {
            name: tool.schema
            for name, tool in self._tools.items()
            if tool.enabled
        }
    
    def get_versions(self, name: str) -> list[str]:
        """Get all versions for a tool.
        
        Args:
            name: Tool name
            
        Returns:
            List of version strings
        """
        return self._versions.get(name, [])
    
    def enable(self, name: str) -> None:
        """Enable a tool.
        
        Args:
            name: Tool name
            
        Raises:
            KeyError: If tool not found
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        self._tools[name].enabled = True
        logger.info(f"Enabled tool: {name}")
    
    def disable(self, name: str) -> None:
        """Disable a tool.
        
        Args:
            name: Tool name
            
        Raises:
            KeyError: If tool not found
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        self._tools[name].enabled = False
        logger.info(f"Disabled tool: {name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a tool.
        
        Args:
            name: Tool name
            
        Raises:
            KeyError: If tool not found
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        del self._tools[name]
        if name in self._versions:
            del self._versions[name]
        logger.info(f"Unregistered tool: {name}")
    
    def count(self) -> int:
        """Get total number of registered tools.
        
        Returns:
            Number of registered tools
        """
        return len(self._tools)
    
    def count_enabled(self) -> int:
        """Get number of enabled tools.
        
        Returns:
            Number of enabled tools
        """
        return sum(1 for tool in self._tools.values() if tool.enabled)


# Global registry instance
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance.
    
    Returns:
        Global ToolRegistry instance
    """
    return _registry


def register_tools(
    tools: list[dict[str, Any]],
    registry: ToolRegistry | None = None,
) -> None:
    """Register multiple tools at once.
    
    Args:
        tools: List of tool dictionaries, each containing:
            - name: Tool name
            - handler: Handler function
            - schema: JSON Schema
            - version: Version (optional)
            - description: Description (optional)
            - tags: Tags (optional)
            - enabled: Enabled flag (optional)
        registry: ToolRegistry instance (default: global registry)
    """
    if registry is None:
        registry = _registry
    
    for tool_config in tools:
        registry.register(**tool_config)
    
    logger.info(f"Registered {len(tools)} tools")
