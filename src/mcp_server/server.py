"""MCP Server implementation for CatLang.

This module implements the MCP server with stdio transport, lifecycle management,
and capabilities declaration.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server

from ..llm_provider import create_from_env
from .resources.guides import GuideResourceManager
from .resources.handlers import (
    set_resource_manager,
    list_all_guide_resources,
    get_guide_resource,
)
from .tools.registry import get_registry
from .tools.schemas import TOOL_SCHEMAS
from .tools.handlers import (
    analyze_n8n_workflow,
    extract_custom_logic,
    generate_langgraph_implementation,
    validate_implementation,
    list_guides,
    query_guide,
    set_server_instance,
)

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP Server for CatLang.
    
    This server provides tools and resources for converting n8n workflows
    to LangGraph implementations.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize MCP Server.
        
        Args:
            config: Server configuration dictionary. Should contain:
                - name: Server name (default: "catlang")
                - version: Server version (default: "0.1.0")
                - description: Server description
                - transport: Transport configuration
                - logging: Logging configuration
                - workspace: Workspace configuration
        """
        self.config = config or {}
        self.name = self.config.get("name", "catlang")
        self.version = self.config.get("version", "0.1.0")
        self.description = self.config.get("description", "n8n to LangGraph Conversion Tool")
        
        # Transport configuration
        transport_config = self.config.get("transport", {})
        self.transport_type = transport_config.get("type", "stdio")
        
        # Logging configuration
        logging_config = self.config.get("logging", {})
        self.log_level = logging_config.get("level", "INFO")
        self.log_format = logging_config.get("format", "json")
        self.log_file = logging_config.get("file")
        
        # Workspace configuration
        workspace_config = self.config.get("workspace", {})
        self.workspace_path = Path(workspace_config.get("path", ".")).resolve()
        self.guides_path = self.workspace_path / workspace_config.get("guides_path", "guides")
        
        # Initialize LLM provider (will be initialized lazily)
        self._llm_provider = None
        
        # Initialize tool registry
        self.tool_registry = get_registry()
        
        # Initialize resource manager
        self.resource_manager = GuideResourceManager(self.guides_path)
        
        # Setup logging
        self._setup_logging()
        
        # Create FastMCP instance
        # Note: FastMCP doesn't accept version parameter, version is stored in server info
        self.mcp = FastMCP(
            name=self.name,
            json_response=True,  # Use JSON content blocks for structured responses
        )
        
        logger.info(f"Initialized {self.name} MCP Server v{self.version}")
        logger.info(f"Transport: {self.transport_type}")
        logger.info(f"Workspace: {self.workspace_path}")
        logger.info(f"Guides path: {self.guides_path}")

    def _setup_logging(self) -> None:
        """Setup structured logging for the server."""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))
        
        # Remove existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        if self.log_format == "json":
            import json
            import time
            
            class JSONFormatter(logging.Formatter):
                def format(self, record: logging.LogRecord) -> str:
                    log_data = {
                        "timestamp": time.time(),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                    }
                    if record.exc_info:
                        log_data["exception"] = self.formatException(record.exc_info)
                    return json.dumps(log_data)
            
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        
        # Console handler (always add)
        console_handler = logging.StreamHandler(sys.stderr)  # Use stderr for logs
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (if configured)
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.info(f"Logging to file: {self.log_file}")

    def _get_llm_provider(self):
        """Get or initialize LLM provider (lazy initialization)."""
        if self._llm_provider is None:
            try:
                logger.info("Initializing LLM provider...")
                self._llm_provider = create_from_env(auto_initialize=True)
                logger.info(
                    f"✓ LLM provider initialized "
                    f"(provider: {self._llm_provider.provider_type}, "
                    f"model: {self._llm_provider.default_model})"
                )
            except Exception as e:
                logger.error(f"Failed to initialize LLM provider: {e}")
                raise RuntimeError(f"LLM provider initialization failed: {e}") from e
        return self._llm_provider

    def _register_capabilities(self) -> None:
        """Register server capabilities (tools, resources, prompts).
        
        This method registers all tools with FastMCP and the tool registry.
        """
        logger.info("Registering server capabilities...")
        
        # Set server instance for handler context (allows handlers to access LLM provider)
        set_server_instance(self)
        
        # Map of tool names to handler functions
        tool_handlers = {
            "analyze_n8n_workflow": analyze_n8n_workflow,
            "extract_custom_logic": extract_custom_logic,
            "generate_langgraph_implementation": generate_langgraph_implementation,
            "validate_implementation": validate_implementation,
            "list_guides": list_guides,
            "query_guide": query_guide,
        }
        
        # Register each tool with both FastMCP and the registry
        for tool_name, handler in tool_handlers.items():
            if tool_name not in TOOL_SCHEMAS:
                logger.warning(f"Tool '{tool_name}' not found in schemas, skipping")
                continue
            
            schema = TOOL_SCHEMAS[tool_name]
            
            # Register with FastMCP using programmatic registration
            # FastMCP.tool() must be called as a decorator: mcp.tool(name="...", description="...")(handler)
            # First call returns a decorator, second call applies it to the handler
            decorator = self.mcp.tool(name=tool_name, description=schema.get("description", ""))
            decorator(handler)
            
            # Register with our internal registry for management
            self.tool_registry.register(
                name=tool_name,
                handler=handler,
                schema=schema,
                version="1.0.0",
                description=schema.get("description", ""),
                enabled=True,
            )
            
            logger.debug(f"Registered tool: {tool_name}")
        
        logger.info(
            f"✓ Registered {self.tool_registry.count_enabled()} tools "
            f"({self.tool_registry.count()} total)"
        )
        
        # Register resources
        self._register_resources()
        
        # Prompts: (if needed)

    def _register_resources(self) -> None:
        """Register MCP resources (guides).
        
        This method registers guide resources with FastMCP.
        """
        logger.info("Registering resources...")
        
        # Initialize resource manager
        self.resource_manager.initialize()
        
        # Set resource manager for handler access
        set_resource_manager(self.resource_manager)
        
        # Register resource for listing all guides
        @self.mcp.resource("guide://list")
        async def list_guides_resource() -> list[dict[str, Any]]:
            """List all available guide resources."""
            return await list_all_guide_resources()
        
        # Register resource template for individual guides
        @self.mcp.resource("guide://docs/{category}/{name}")
        async def get_guide_resource_handler(category: str, name: str) -> dict[str, Any]:
            """Get a specific guide resource by category and name."""
            return await get_guide_resource(category, name)
        
        logger.info("✓ Registered guide resources")

    async def start(self) -> None:
        """Start the MCP server.
        
        This method starts the server with stdio transport and runs
        the main event loop.
        """
        logger.info("Starting MCP server...")
        
        # Register capabilities
        self._register_capabilities()
        
        # Only support stdio transport for now (other transports in post-MVP)
        if self.transport_type != "stdio":
            raise ValueError(
                f"Transport type '{self.transport_type}' not supported yet. "
                "Only 'stdio' is currently supported."
            )
        
        try:
            # Open stdio transport
            async with stdio_server() as (read_stream, write_stream):
                logger.info("✓ stdio transport initialized")
                logger.info("Server ready. Waiting for requests...")
                
                # Run the MCP server
                # FastMCP.run() accepts read_stream and write_stream directly
                await self.mcp.run(
                    read_stream=read_stream,
                    write_stream=write_stream,
                )
        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Shutting down...")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the MCP server gracefully."""
        logger.info("Stopping MCP server...")
        
        # Cleanup resources
        if self._llm_provider:
            logger.info("Cleaning up LLM provider...")
            # Provider cleanup if needed
        
        logger.info("MCP server stopped")

    def get_capabilities(self) -> dict[str, Any]:
        """Get server capabilities declaration.
        
        Returns:
            Dictionary containing server capabilities information
        """
        return {
            "server": {
                "name": self.name,
                "version": self.version,
                "description": self.description,
            },
            "protocol": {
                "version": "2024-11-05",  # MCP protocol version
            },
            "capabilities": {
                "tools": {
                    "listChanged": False,  # Tools don't change dynamically
                },
                "resources": {
                    "subscribe": False,  # Resources don't support subscriptions yet
                    "listChanged": False,
                },
                "prompts": {
                    "listChanged": False,
                },
            },
        }


def create_server(config: Optional[dict[str, Any]] = None) -> MCPServer:
    """Factory function to create an MCP server instance.
    
    Args:
        config: Server configuration dictionary. If None, will try to load
                from config/server.yaml
        
    Returns:
        MCPServer instance
    """
    if config is None:
        # Try to load from config file
        config_path = Path(__file__).parent.parent.parent / "config" / "server.yaml"
        if config_path.exists():
            try:
                import yaml
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
                    # Merge server config
                    config = config_data.get("server", {})
                    config.update(config_data)  # Add transport, logging, workspace
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default configuration")
                config = {}
    
    return MCPServer(config)


async def main() -> None:
    """Main entry point for running the MCP server."""
    try:
        server = create_server()
        await server.start()
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
