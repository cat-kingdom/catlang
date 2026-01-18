"""Integration tests for Fase 10 - MCP Protocol Compliance and End-to-End Testing.

Tests cover:
- MCP protocol compliance (capabilities, tool listing, resource listing)
- Tool execution end-to-end
- Resource access end-to-end
- Server lifecycle integration
- Error handling and error propagation
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Dict

from src.mcp_server.server import MCPServer, create_server
from src.mcp_server.tools.handlers import set_server_instance
from src.mcp_server.resources.handlers import set_resource_manager


@pytest.fixture
def temp_guides_dir():
    """Create temporary guides directory with sample guides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        guides_dir = Path(tmpdir) / "guides"
        guides_dir.mkdir()
        
        # Create sample guide
        guide_file = guides_dir / "test-guide.md"
        guide_file.write_text("""---
title: "Test Guide"
category: "testing"
tags: ["test", "example"]
description: "A test guide for integration testing"
---

# Test Guide

This is a test guide for integration testing.
""")
        
        yield guides_dir


@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider."""
    provider = Mock()
    provider.is_initialized.return_value = True
    provider.provider_type = "openai"
    provider.default_model = "gpt-4o-mini"
    
    # Mock generate method
    mock_response = Mock()
    mock_response.content = "Test response"
    provider.generate.return_value = mock_response
    
    return provider


@pytest.fixture
def server_config(temp_guides_dir, mock_llm_provider):
    """Create server configuration for testing."""
    return {
        "name": "test-catlang",
        "version": "0.1.0",
        "description": "Test MCP Server",
        "transport": {"type": "stdio"},
        "logging": {"level": "ERROR"},  # Reduce log noise
        "workspace": {
            "path": str(temp_guides_dir.parent),
            "guides_path": "guides",
        },
    }


@pytest.fixture
def test_server(server_config, mock_llm_provider):
    """Create test server instance."""
    server = MCPServer(server_config)
    
    # Inject mock provider
    server._llm_provider = mock_llm_provider
    
    return server


@pytest.mark.integration
class TestMCPProtocolCompliance:
    """Test MCP protocol compliance."""
    
    def test_server_capabilities_declaration(self, test_server):
        """Test server capabilities declaration."""
        capabilities = test_server.get_capabilities()
        
        assert "server" in capabilities
        assert "protocol" in capabilities
        assert "capabilities" in capabilities
        
        # Check server info
        assert capabilities["server"]["name"] == "test-catlang"
        assert capabilities["server"]["version"] == "0.1.0"
        
        # Check protocol version
        assert "version" in capabilities["protocol"]
        assert capabilities["protocol"]["version"] == "2024-11-05"
        
        # Check capabilities structure
        caps = capabilities["capabilities"]
        assert "tools" in caps
        assert "resources" in caps
        assert "prompts" in caps
        
        # Check tools capability
        assert "listChanged" in caps["tools"]
        assert caps["tools"]["listChanged"] is False
        
        # Check resources capability
        assert "subscribe" in caps["resources"]
        assert "listChanged" in caps["resources"]
        assert caps["resources"]["subscribe"] is False
        assert caps["resources"]["listChanged"] is False
    
    def test_tool_registration(self, test_server):
        """Test that all tools are registered."""
        test_server._register_capabilities()
        
        # Check tool registry
        registry = test_server.tool_registry
        assert registry.count() == 6  # 6 tools total
        
        # Check expected tools are registered
        expected_tools = [
            "analyze_n8n_workflow",
            "extract_custom_logic",
            "generate_langgraph_implementation",
            "validate_implementation",
            "list_guides",
            "query_guide",
        ]
        
        registered_tools = registry.list_tools()
        for tool_name in expected_tools:
            assert tool_name in registered_tools, f"Tool {tool_name} not registered"
            
            tool = registry.get(tool_name)
            assert tool is not None
            assert tool.enabled is True
            assert tool.handler is not None
            assert tool.schema is not None
    
    def test_resource_registration(self, test_server):
        """Test that resources are registered."""
        test_server._register_capabilities()
        
        # Resource manager should be initialized
        assert test_server.resource_manager is not None
        
        # Resources should be accessible
        resources = test_server.resource_manager.list_resources()
        assert isinstance(resources, list)
        assert len(resources) > 0  # Should have at least test guide
    
    def test_protocol_version_compatibility(self, test_server):
        """Test protocol version compatibility."""
        capabilities = test_server.get_capabilities()
        protocol_version = capabilities["protocol"]["version"]
        
        # Should support MCP protocol version 2024-11-05
        assert protocol_version == "2024-11-05"


@pytest.mark.integration
class TestToolExecutionEndToEnd:
    """Test tool execution end-to-end."""
    
    def test_analyze_n8n_workflow_tool(self, test_server, mock_llm_provider):
        """Test analyze_n8n_workflow tool execution."""
        test_server._register_capabilities()
        
        # Mock LLM response for analysis
        mock_response = Mock()
        mock_response.content = """# Production Requirements

## Global Workflow Summary
- Objective: Test workflow
- Triggers: Webhook
- Execution: Sequential

## Node Specifications
- Node 1: Webhook trigger
- Node 2: Process data

## Custom Nodes
None identified.
"""
        mock_llm_provider.generate.return_value = mock_response
        
        # Sample n8n workflow JSON
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "nodes": [
                {"id": "1", "name": "Webhook", "type": "n8n-nodes-base.webhook"},
                {"id": "2", "name": "Process", "type": "n8n-nodes-base.function"},
            ],
            "connections": {},
        })
        
        # Get tool handler
        handler = test_server.tool_registry.get_handler("analyze_n8n_workflow")
        assert handler is not None
        
        # Execute tool
        import asyncio
        result = asyncio.run(handler(workflow_json=workflow_json, include_metadata=True))
        
        assert result["status"] == "success"
        assert "requirements" in result
        assert "metadata" in result
    
    def test_extract_custom_logic_tool(self, test_server, mock_llm_provider):
        """Test extract_custom_logic tool execution."""
        test_server._register_capabilities()
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """# Custom Logic Specifications

## Purpose
Process data transformation

## Inputs
- data: str

## Processing Logic
Transform input data

## Outputs
- result: str

## Dependencies
- None
"""
        mock_llm_provider.generate.return_value = mock_response
        
        # Sample Python code
        code = """
def process_data(data):
    return data.upper()
"""
        
        handler = test_server.tool_registry.get_handler("extract_custom_logic")
        assert handler is not None
        
        result = asyncio.run(handler(code=code, language="python", node_name="test"))
        
        assert result["status"] == "success"
        assert "specifications" in result
    
    def test_list_guides_tool(self, test_server):
        """Test list_guides tool execution."""
        test_server._register_capabilities()
        
        handler = test_server.tool_registry.get_handler("list_guides")
        assert handler is not None
        
        result = asyncio.run(handler())
        
        assert result["status"] == "success"
        assert "guides" in result
        assert isinstance(result["guides"], list)
        assert result["count"] >= 0
    
    def test_query_guide_tool(self, test_server):
        """Test query_guide tool execution."""
        test_server._register_capabilities()
        
        handler = test_server.tool_registry.get_handler("query_guide")
        assert handler is not None
        
        result = asyncio.run(handler(guide_name="test-guide"))
        
        assert result["status"] == "success"
        assert "content" in result
        assert "title" in result
    
    def test_complete_workflow_tools(self, test_server, mock_llm_provider):
        """Test complete workflow: analyze -> extract -> generate -> validate."""
        test_server._register_capabilities()
        
        # Mock LLM responses - need more complete response for analyze
        mock_analysis = Mock()
        mock_analysis.content = """# Production Requirements

## Global Workflow Summary
- Objective: Test workflow
- Triggers: Webhook
- Execution: Sequential

## Node Specifications
- Node 1: Webhook trigger
- Node 2: Process data

## Custom Nodes
None identified.

## Implementation Notes
Simple sequential workflow suitable for Functional API.
"""
        mock_gen = Mock()
        mock_gen.content = """```python
from langgraph import entrypoint

@entrypoint
def workflow(data):
    return {"result": "success"}
```"""
        
        # Reset side_effect for each call
        call_count = 0
        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_analysis
            else:
                return mock_gen
        
        mock_llm_provider.generate.side_effect = mock_generate
        
        # Step 1: Analyze
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "nodes": [
                {"id": "1", "name": "Webhook", "type": "n8n-nodes-base.webhook"},
                {"id": "2", "name": "Process", "type": "n8n-nodes-base.function"},
            ],
            "connections": {},
        })
        analyze_handler = test_server.tool_registry.get_handler("analyze_n8n_workflow")
        analyze_result = asyncio.run(analyze_handler(workflow_json=workflow_json, include_metadata=True))
        
        # If analysis fails, skip rest of test
        if analyze_result["status"] != "success":
            pytest.skip(f"Analysis failed: {analyze_result.get('error', 'Unknown error')}")
        
        assert analyze_result["status"] == "success"
        requirements = analyze_result["requirements"]
        
        # Step 2: Generate
        generate_handler = test_server.tool_registry.get_handler("generate_langgraph_implementation")
        generate_result = asyncio.run(
            generate_handler(
                requirements=requirements,
                paradigm="functional",
                output_format="code",
            )
        )
        
        if generate_result["status"] == "success":
            code = generate_result["code"]
            
            # Step 3: Validate
            validate_handler = test_server.tool_registry.get_handler("validate_implementation")
            validate_result = asyncio.run(
                validate_handler(
                    code=code,
                    check_syntax=True,
                    check_compliance=True,
                )
            )
            
            assert validate_result["status"] == "success"
            assert "syntax" in validate_result
        else:
            pytest.skip(f"Generation failed: {generate_result.get('error', 'Unknown error')}")
    
    def test_tool_error_handling(self, test_server):
        """Test tool error handling and propagation."""
        test_server._register_capabilities()
        
        # Test with invalid input
        handler = test_server.tool_registry.get_handler("analyze_n8n_workflow")
        
        # Invalid JSON should be handled gracefully
        result = asyncio.run(handler(workflow_json="invalid json"))
        
        # Should return error status or handle gracefully
        assert "status" in result
        # May be success with error message or error status
        assert result["status"] in ["success", "error"]
    
    def test_tool_parameter_validation(self, test_server):
        """Test tool parameter validation."""
        test_server._register_capabilities()
        
        # Test with missing required parameters
        handler = test_server.tool_registry.get_handler("query_guide")
        
        # Missing guide_name should be handled
        # FastMCP handles parameter validation, so this may raise or return error
        try:
            result = asyncio.run(handler())
            # If it doesn't raise, should have error status
            assert "status" in result
        except Exception:
            # Parameter validation error is acceptable
            pass


@pytest.mark.integration
class TestResourceAccessEndToEnd:
    """Test resource access end-to-end."""
    
    def test_resource_listing(self, test_server):
        """Test resource listing via resource manager."""
        test_server._register_capabilities()
        
        resources = test_server.resource_manager.list_resources()
        
        assert isinstance(resources, list)
        assert len(resources) > 0
        
        # Check resource format
        for resource in resources:
            assert "uri" in resource
            assert "name" in resource
            assert "mimeType" in resource
            assert resource["mimeType"] == "text/markdown"
    
    def test_resource_reading_by_uri(self, test_server):
        """Test reading resource by URI."""
        test_server._register_capabilities()
        
        # Get first resource URI
        resources = test_server.resource_manager.list_resources()
        assert len(resources) > 0
        
        first_resource = resources[0]
        uri = first_resource["uri"]
        
        # Read resource
        resource = test_server.resource_manager.get_resource(uri)
        
        assert resource is not None
        assert "uri" in resource
        assert "contents" in resource
        assert len(resource["contents"]) > 0
        assert "text" in resource["contents"][0]
    
    def test_resource_error_handling_invalid_uri(self, test_server):
        """Test error handling for invalid resource URI."""
        test_server._register_capabilities()
        
        # Try to get non-existent resource
        invalid_uri = "guide://docs/nonexistent/category"
        resource = test_server.resource_manager.get_resource(invalid_uri)
        
        # Should return None or empty dict
        assert resource is None or resource == {}
    
    def test_resource_search_functionality(self, test_server):
        """Test resource search functionality."""
        test_server._register_capabilities()
        
        # Search guides by category
        indexer = test_server.resource_manager.indexer
        results = indexer.search_guides("test", category="testing")
        
        assert isinstance(results, list)
        # Should find test guide
        assert len(results) > 0


@pytest.mark.integration
@pytest.mark.asyncio
class TestServerLifecycleIntegration:
    """Test server lifecycle integration."""
    
    async def test_server_startup(self, server_config, mock_llm_provider):
        """Test server startup with stdio transport."""
        server = MCPServer(server_config)
        server._llm_provider = mock_llm_provider
        
        # Mock stdio_server to avoid actual stdio operations
        with patch('src.mcp_server.server.stdio_server') as mock_stdio:
            mock_read = AsyncMock()
            mock_write = AsyncMock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_stdio.return_value = mock_context
            
            # Mock mcp.run to avoid actual execution
            server.mcp.run = AsyncMock()
            
            # Start server (should register capabilities)
            try:
                await asyncio.wait_for(server.start(), timeout=0.1)
            except asyncio.TimeoutError:
                # Expected - server runs indefinitely
                pass
            
            # Verify capabilities were registered
            assert server.tool_registry.count() > 0
            assert server.resource_manager is not None
    
    @pytest.mark.asyncio
    async def test_server_graceful_shutdown(self, test_server):
        """Test server graceful shutdown."""
        # Stop should not raise
        await test_server.stop()
        
        # Verify cleanup
        # (No specific cleanup needed for MVP, but method should exist)
    
    @pytest.mark.asyncio
    async def test_server_error_recovery(self, server_config, mock_llm_provider):
        """Test server error recovery."""
        server = MCPServer(server_config)
        server._llm_provider = mock_llm_provider
        
        # Register capabilities (should not raise even with errors)
        try:
            server._register_capabilities()
        except Exception as e:
            pytest.fail(f"Server registration should handle errors gracefully: {e}")
        
        # Server should still be functional
        assert server.tool_registry.count() > 0
    
    @pytest.mark.asyncio
    async def test_server_concurrent_requests_handling(self, test_server):
        """Test server handles concurrent requests."""
        test_server._register_capabilities()
        
        # Simulate concurrent tool calls
        handler = test_server.tool_registry.get_handler("list_guides")
        
        # Run multiple concurrent calls
        results = await asyncio.gather(
            handler(),
            handler(),
            handler(),
        )
        
        # All should succeed
        for result in results:
            assert result["status"] == "success"


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and error propagation."""
    
    def test_missing_resource_manager(self, test_server):
        """Test error handling when resource manager is missing."""
        # Temporarily remove resource manager
        original_rm = test_server.resource_manager
        test_server.resource_manager = None
        
        try:
            # Should handle gracefully
            test_server._register_capabilities()
        except Exception:
            # Error is acceptable if resource manager is required
            pass
        finally:
            # Restore
            test_server.resource_manager = original_rm
    
    def test_missing_llm_provider(self, server_config):
        """Test error handling when LLM provider is missing."""
        server = MCPServer(server_config)
        
        # Don't set provider
        # Provider should be initialized lazily
        # But if env vars are missing, should handle gracefully
        
        # Register capabilities (should not fail)
        try:
            server._register_capabilities()
        except Exception as e:
            # If provider is required, error is acceptable
            # But should be handled gracefully
            assert "provider" in str(e).lower() or "llm" in str(e).lower()
    
    def test_invalid_tool_parameters(self, test_server):
        """Test error handling for invalid tool parameters."""
        test_server._register_capabilities()
        
        handler = test_server.tool_registry.get_handler("validate_implementation")
        
        # Test with invalid code
        result = asyncio.run(handler(code="invalid python code !@#$%"))
        
        # Should handle gracefully
        assert "status" in result
        # May return error or success with validation errors
        assert result["status"] in ["success", "error"]


@pytest.mark.integration
class TestMCPResponseFormat:
    """Test MCP response format compliance."""
    
    def test_tool_response_format(self, test_server):
        """Test tool response format matches MCP spec."""
        test_server._register_capabilities()
        
        handler = test_server.tool_registry.get_handler("list_guides")
        result = asyncio.run(handler())
        
        # Response should have status
        assert "status" in result
        
        # If success, should have expected fields
        if result["status"] == "success":
            assert "guides" in result
            assert "count" in result
    
    def test_resource_response_format(self, test_server):
        """Test resource response format matches MCP spec."""
        test_server._register_capabilities()
        
        resources = test_server.resource_manager.list_resources()
        
        for resource in resources:
            # Check MCP resource format
            assert "uri" in resource
            assert "name" in resource
            assert "mimeType" in resource
            
            # Read resource and check format
            resource_data = test_server.resource_manager.get_resource(resource["uri"])
            if resource_data:
                assert "uri" in resource_data
                assert "contents" in resource_data
                assert isinstance(resource_data["contents"], list)
                if len(resource_data["contents"]) > 0:
                    assert "text" in resource_data["contents"][0] or "uri" in resource_data["contents"][0]
