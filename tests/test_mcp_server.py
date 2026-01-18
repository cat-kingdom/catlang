"""Tests for MCP Server (Fase 0 - Basic Structure)."""

import pytest
from src.mcp_server.server import MCPServer, create_server


@pytest.mark.unit
def test_server_initialization():
    """Test that MCP server can be initialized."""
    config = {
        "name": "test-server",
        "version": "0.1.0",
    }
    server = MCPServer(config)
    assert server.name == "test-server"
    assert server.version == "0.1.0"


@pytest.mark.unit
def test_server_default_config():
    """Test server with default configuration."""
    server = MCPServer()
    assert server.name == "catlang"
    assert server.version == "0.1.0"


@pytest.mark.unit
def test_create_server_factory():
    """Test server factory function."""
    config = {"name": "factory-test"}
    server = create_server(config)
    assert isinstance(server, MCPServer)
    assert server.name == "factory-test"
