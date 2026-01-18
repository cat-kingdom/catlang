"""Tests for MCP Server Fase 4 - Tool Registration System.

Tests cover:
- Tool registry system
- Tool schema definitions
- Parameter validation
- Tool routing mechanism
- Tool registration with FastMCP
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Any

from src.mcp_server.tools.registry import ToolRegistry, ToolMetadata, get_registry
from src.mcp_server.tools.schemas import (
    ToolSchema,
    get_tool_schema,
    get_tool_schemas,
    TOOL_SCHEMAS,
)
from src.mcp_server.tools.handlers import (
    analyze_n8n_workflow,
    extract_custom_logic,
    generate_langgraph_implementation,
    validate_implementation,
    list_guides,
    query_guide,
)
from src.mcp_server.server import MCPServer, create_server


@pytest.mark.unit
class TestToolRegistry:
    """Test tool registry system."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = ToolRegistry()
        assert registry.count() == 0
        assert registry.count_enabled() == 0
        assert registry.list_tools() == []

    def test_register_tool(self):
        """Test tool registration."""
        registry = ToolRegistry()
        
        def dummy_handler(x: int) -> int:
            return x * 2
        
        schema = {
            "name": "test_tool",
            "description": "Test tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                },
                "required": ["x"],
            },
        }
        
        registry.register(
            name="test_tool",
            handler=dummy_handler,
            schema=schema,
            version="1.0.0",
            description="Test tool",
        )
        
        assert registry.count() == 1
        assert registry.count_enabled() == 1
        assert "test_tool" in registry.list_tools()
        
        tool = registry.get("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"
        assert tool.version == "1.0.0"
        assert tool.handler == dummy_handler
        assert tool.enabled is True

    def test_register_tool_duplicate_name(self):
        """Test registering tool with duplicate name."""
        registry = ToolRegistry()
        
        def handler1(x: int) -> int:
            return x
        
        def handler2(x: int) -> int:
            return x * 2
        
        schema = {
            "name": "test_tool",
            "description": "Test tool",
            "inputSchema": {"type": "object", "properties": {}},
        }
        
        registry.register(
            name="test_tool",
            handler=handler1,
            schema=schema,
        )
        
        # Same version should be skipped with warning
        registry.register(
            name="test_tool",
            handler=handler2,
            schema=schema,
        )
        
        # Should still have handler1
        assert registry.get("test_tool").handler == handler1
        
        # Different version should raise error
        with pytest.raises(ValueError, match="already registered"):
            registry.register(
                name="test_tool",
                handler=handler2,
                schema=schema,
                version="2.0.0",
            )

    def test_get_handler(self):
        """Test getting tool handler."""
        registry = ToolRegistry()
        
        def handler(x: int) -> int:
            return x * 2
        
        schema = {
            "name": "test_tool",
            "description": "Test tool",
            "inputSchema": {"type": "object", "properties": {}},
        }
        
        registry.register(
            name="test_tool",
            handler=handler,
            schema=schema,
        )
        
        retrieved_handler = registry.get_handler("test_tool")
        assert retrieved_handler == handler
        
        # Test non-existent tool
        assert registry.get_handler("nonexistent") is None

    def test_disable_enable_tool(self):
        """Test disabling and enabling tools."""
        registry = ToolRegistry()
        
        def handler(x: int) -> int:
            return x
        
        schema = {
            "name": "test_tool",
            "description": "Test tool",
            "inputSchema": {"type": "object", "properties": {}},
        }
        
        registry.register(
            name="test_tool",
            handler=handler,
            schema=schema,
        )
        
        assert registry.count_enabled() == 1
        
        registry.disable("test_tool")
        assert registry.count_enabled() == 0
        assert registry.get_handler("test_tool") is None
        
        registry.enable("test_tool")
        assert registry.count_enabled() == 1
        assert registry.get_handler("test_tool") == handler

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        
        def handler(x: int) -> int:
            return x
        
        schema = {
            "name": "test_tool",
            "description": "Test tool",
            "inputSchema": {"type": "object", "properties": {}},
        }
        
        registry.register(
            name="test_tool",
            handler=handler,
            schema=schema,
        )
        
        assert registry.count() == 1
        
        registry.unregister("test_tool")
        assert registry.count() == 0
        assert registry.get("test_tool") is None
        
        with pytest.raises(KeyError):
            registry.unregister("nonexistent")

    def test_get_all_schemas(self):
        """Test getting all tool schemas."""
        registry = ToolRegistry()
        
        schema1 = {
            "name": "tool1",
            "description": "Tool 1",
            "inputSchema": {"type": "object", "properties": {}},
        }
        schema2 = {
            "name": "tool2",
            "description": "Tool 2",
            "inputSchema": {"type": "object", "properties": {}},
        }
        
        registry.register(
            name="tool1",
            handler=lambda x: x,
            schema=schema1,
        )
        registry.register(
            name="tool2",
            handler=lambda x: x,
            schema=schema2,
        )
        
        schemas = registry.get_all_schemas()
        assert len(schemas) == 2
        assert "tool1" in schemas
        assert "tool2" in schemas
        
        # Disable one tool
        registry.disable("tool1")
        schemas = registry.get_all_schemas()
        assert len(schemas) == 1
        assert "tool2" in schemas

    def test_get_versions(self):
        """Test getting tool versions."""
        registry = ToolRegistry()
        
        schema = {
            "name": "test_tool",
            "description": "Test tool",
            "inputSchema": {"type": "object", "properties": {}},
        }
        
        registry.register(
            name="test_tool",
            handler=lambda x: x,
            schema=schema,
            version="1.0.0",
        )
        
        versions = registry.get_versions("test_tool")
        assert "1.0.0" in versions
        
        # Non-existent tool
        assert registry.get_versions("nonexistent") == []

    def test_global_registry(self):
        """Test global registry instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2


@pytest.mark.unit
class TestToolSchemas:
    """Test tool schema definitions."""

    def test_all_tools_have_schemas(self):
        """Test that all expected tools have schemas."""
        expected_tools = [
            "analyze_n8n_workflow",
            "extract_custom_logic",
            "generate_langgraph_implementation",
            "validate_implementation",
            "list_guides",
            "query_guide",
        ]
        
        for tool_name in expected_tools:
            assert tool_name in TOOL_SCHEMAS, f"Missing schema for {tool_name}"
            schema = TOOL_SCHEMAS[tool_name]
            assert "name" in schema
            assert "description" in schema
            assert "inputSchema" in schema

    def test_get_tool_schema(self):
        """Test getting tool schema."""
        schema = get_tool_schema("analyze_n8n_workflow")
        assert schema is not None
        assert isinstance(schema, ToolSchema)
        assert schema.name == "analyze_n8n_workflow"
        
        # Non-existent tool
        assert get_tool_schema("nonexistent") is None

    def test_tool_schema_validation(self):
        """Test tool schema parameter validation."""
        schema = ToolSchema("test_tool", {
            "description": "Test",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "required_param": {"type": "string"},
                    "optional_param": {"type": "integer"},
                },
                "required": ["required_param"],
            },
        })
        
        # Valid params
        is_valid, missing = schema.validate_required({
            "required_param": "value",
            "optional_param": 123,
        })
        assert is_valid is True
        assert missing == []
        
        # Missing required param
        is_valid, missing = schema.validate_required({
            "optional_param": 123,
        })
        assert is_valid is False
        assert "required_param" in missing

    def test_get_tool_schemas(self):
        """Test getting all tool schemas."""
        schemas = get_tool_schemas()
        assert len(schemas) == len(TOOL_SCHEMAS)
        
        for name, schema in schemas.items():
            assert isinstance(schema, ToolSchema)
            assert schema.name == name

    def test_schema_properties(self):
        """Test schema property access."""
        schema = get_tool_schema("analyze_n8n_workflow")
        assert schema is not None
        
        properties = schema.get_properties()
        assert isinstance(properties, dict)
        assert "workflow_json" in properties
        
        required = schema.get_required_params()
        assert isinstance(required, list)
        assert "workflow_json" in required


@pytest.mark.unit
class TestToolHandlers:
    """Test tool handler functions."""

    @pytest.mark.asyncio
    async def test_analyze_n8n_workflow_handler(self):
        """Test analyze_n8n_workflow handler."""
        # Mock server instance with provider
        from unittest.mock import Mock, patch
        from src.mcp_server.tools.handlers import set_server_instance
        from src.llm_provider.base import GenerationResponse
        
        mock_server = Mock()
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        mock_provider.generate.return_value = GenerationResponse(
            content="Test requirements",
            model="gpt-4o-mini",
        )
        mock_server._get_llm_provider.return_value = mock_provider
        set_server_instance(mock_server)
        
        result = await analyze_n8n_workflow(
            workflow_json='{"nodes": [], "connections": {}}',
            include_metadata=True,
        )
        assert isinstance(result, dict)
        assert "status" in result
        # Handler is now implemented, should return success or error (not not_implemented)
        assert result["status"] in ["success", "error"]

    @pytest.mark.asyncio
    async def test_extract_custom_logic_handler(self):
        """Test extract_custom_logic handler."""
        result = await extract_custom_logic(
            code="def test(): pass",
            language="python",
            node_name="test_node",
        )
        assert isinstance(result, dict)
        assert "status" in result
        assert result["language"] == "python"

    @pytest.mark.asyncio
    async def test_generate_langgraph_implementation_handler(self):
        """Test generate_langgraph_implementation handler."""
        result = await generate_langgraph_implementation(
            requirements="Test requirements",
            paradigm="functional",
        )
        assert isinstance(result, dict)
        assert "status" in result
        assert result["paradigm"] == "functional"

    @pytest.mark.asyncio
    async def test_validate_implementation_handler(self):
        """Test validate_implementation handler."""
        result = await validate_implementation(
            code="def test(): pass",
            check_syntax=True,
        )
        assert isinstance(result, dict)
        assert "status" in result
        assert "checks" in result

    @pytest.mark.asyncio
    async def test_list_guides_handler(self):
        """Test list_guides handler."""
        result = await list_guides(category="test")
        assert isinstance(result, dict)
        assert "status" in result
        assert "guides" in result

    @pytest.mark.asyncio
    async def test_query_guide_handler(self):
        """Test query_guide handler."""
        result = await query_guide(guide_name="test_guide")
        assert isinstance(result, dict)
        assert "status" in result
        assert result["guide_name"] == "test_guide"


@pytest.mark.unit
class TestServerToolRegistration:
    """Test tool registration in MCP server."""

    def test_server_registers_tools(self):
        """Test that server registers all tools."""
        config = {
            "name": "test-server",
            "version": "1.0.0",
            "transport": {"type": "stdio"},
            "logging": {"level": "INFO"},
        }
        server = MCPServer(config)
        
        # Tools should be registered in _register_capabilities
        # But we need to call it explicitly for testing
        server._register_capabilities()
        
        # Check registry
        assert server.tool_registry.count() == 6
        assert server.tool_registry.count_enabled() == 6
        
        # Check all expected tools are registered
        expected_tools = [
            "analyze_n8n_workflow",
            "extract_custom_logic",
            "generate_langgraph_implementation",
            "validate_implementation",
            "list_guides",
            "query_guide",
        ]
        
        for tool_name in expected_tools:
            assert tool_name in server.tool_registry.list_tools()
            tool = server.tool_registry.get(tool_name)
            assert tool is not None
            assert tool.enabled is True

    def test_server_tool_routing(self):
        """Test that server can route tool calls."""
        config = {
            "name": "test-server",
            "version": "1.0.0",
            "transport": {"type": "stdio"},
            "logging": {"level": "INFO"},
        }
        server = MCPServer(config)
        server._register_capabilities()
        
        # Get handler for a tool
        handler = server.tool_registry.get_handler("analyze_n8n_workflow")
        assert handler is not None
        assert callable(handler)
        
        # Test that handler is the correct function
        assert handler == analyze_n8n_workflow

    def test_server_tool_schemas(self):
        """Test that server has correct tool schemas."""
        config = {
            "name": "test-server",
            "version": "1.0.0",
            "transport": {"type": "stdio"},
            "logging": {"level": "INFO"},
        }
        server = MCPServer(config)
        server._register_capabilities()
        
        schemas = server.tool_registry.get_all_schemas()
        assert len(schemas) == 6
        
        # Check schema structure
        for tool_name, schema in schemas.items():
            assert "name" in schema
            assert "description" in schema
            assert "inputSchema" in schema
            assert schema["name"] == tool_name


@pytest.mark.integration
class TestToolRegistrationIntegration:
    """Integration tests for tool registration."""

    def test_create_server_registers_tools(self):
        """Test that create_server factory registers tools."""
        server = create_server()
        
        # Should have tool registry
        assert hasattr(server, "tool_registry")
        assert server.tool_registry is not None
        
        # Tools will be registered when _register_capabilities is called
        # which happens in start(), but we can test registration directly
        server._register_capabilities()
        
        assert server.tool_registry.count() > 0
