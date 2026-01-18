"""Tests for Fase 5: Analyze n8n Workflow tool implementation."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.mcp_server.tools.analyze import (
    load_step1_prompt_template,
    parse_n8n_workflow,
    validate_n8n_schema,
    extract_nodes,
    extract_connections,
    identify_custom_nodes,
    extract_workflow_metadata,
    build_analysis_prompt,
    generate_requirements,
    format_llm_response,
)
from src.mcp_server.tools.handlers import (
    analyze_n8n_workflow,
    set_server_instance,
    _get_llm_provider,
)
from src.llm_provider.base import GenerationResponse


# Sample n8n workflow JSON for testing
SAMPLE_WORKFLOW_JSON = json.dumps({
    "name": "Test Workflow",
    "id": "test-123",
    "version": 1,
    "nodes": [
        {
            "id": "node1",
            "name": "Webhook",
            "type": "n8n-nodes-base.webhook",
            "parameters": {
                "httpMethod": "POST",
                "path": "test",
            },
        },
        {
            "id": "node2",
            "name": "Function Node",
            "type": "n8n-nodes-base.function",
            "parameters": {
                "functionCode": "return items;",
            },
        },
        {
            "id": "node3",
            "name": "HTTP Request",
            "type": "n8n-nodes-base.httpRequest",
            "parameters": {
                "url": "https://api.example.com",
                "method": "GET",
            },
        },
    ],
    "connections": {
        "Webhook": {
            "main": [[{"node": "Function Node", "type": "main", "index": 0}]],
        },
        "Function Node": {
            "main": [[{"node": "HTTP Request", "type": "main", "index": 0}]],
        },
    },
})

SIMPLE_WORKFLOW_JSON = json.dumps({
    "name": "Simple Workflow",
    "nodes": [
        {
            "id": "node1",
            "name": "Trigger",
            "type": "n8n-nodes-base.manualTrigger",
        },
    ],
    "connections": {},
})

INVALID_JSON = "{ invalid json }"

MISSING_FIELDS_JSON = json.dumps({
    "name": "Invalid Workflow",
    # Missing nodes and connections
})


class TestLoadStep1PromptTemplate:
    """Tests for loading Step1 prompt template."""
    
    def test_load_template_success(self):
        """Test successful loading of prompt template."""
        template = load_step1_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0
        # Check for key phrases that should be in the template
        template_lower = template.lower()
        assert ("n8n" in template_lower and "workflow" in template_lower) or "n8n json" in template_lower
    
    @patch("src.mcp_server.tools.analyze.STEP1_PROMPT_PATH")
    def test_load_template_not_found(self, mock_path):
        """Test handling of missing template file."""
        mock_path.exists.return_value = False
        with pytest.raises(FileNotFoundError):
            load_step1_prompt_template()
    
    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_load_template_io_error(self, mock_open):
        """Test handling of IO errors when reading template."""
        with pytest.raises(IOError):
            load_step1_prompt_template()


class TestParseN8nWorkflow:
    """Tests for parsing n8n workflow JSON."""
    
    def test_parse_valid_json(self):
        """Test parsing valid n8n workflow JSON."""
        result = parse_n8n_workflow(SAMPLE_WORKFLOW_JSON)
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "connections" in result
        assert result["name"] == "Test Workflow"
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        with pytest.raises(ValueError, match="Invalid JSON format"):
            parse_n8n_workflow(INVALID_JSON)
    
    def test_parse_empty_string(self):
        """Test parsing empty string."""
        with pytest.raises(ValueError):
            parse_n8n_workflow("")


class TestValidateN8nSchema:
    """Tests for validating n8n workflow schema."""
    
    def test_validate_valid_schema(self):
        """Test validation of valid workflow schema."""
        workflow_data = json.loads(SAMPLE_WORKFLOW_JSON)
        is_valid, error_msg = validate_n8n_schema(workflow_data)
        assert is_valid is True
        assert error_msg is None
    
    def test_validate_missing_nodes(self):
        """Test validation with missing nodes field."""
        workflow_data = json.loads(MISSING_FIELDS_JSON)
        is_valid, error_msg = validate_n8n_schema(workflow_data)
        assert is_valid is False
        assert "nodes" in error_msg.lower()
    
    def test_validate_missing_connections(self):
        """Test validation with missing connections field."""
        workflow_data = {"nodes": []}
        is_valid, error_msg = validate_n8n_schema(workflow_data)
        assert is_valid is False
        assert "connections" in error_msg.lower()
    
    def test_validate_invalid_nodes_type(self):
        """Test validation with invalid nodes type."""
        workflow_data = {"nodes": "not a list", "connections": {}}
        is_valid, error_msg = validate_n8n_schema(workflow_data)
        assert is_valid is False
        assert "list" in error_msg.lower()
    
    def test_validate_invalid_connections_type(self):
        """Test validation with invalid connections type."""
        workflow_data = {"nodes": [], "connections": "not a dict"}
        is_valid, error_msg = validate_n8n_schema(workflow_data)
        assert is_valid is False
        assert "dictionary" in error_msg.lower()
    
    def test_validate_not_dict(self):
        """Test validation with non-dict input."""
        is_valid, error_msg = validate_n8n_schema([])
        assert is_valid is False
        assert "dictionary" in error_msg.lower()


class TestExtractNodes:
    """Tests for extracting nodes from workflow."""
    
    def test_extract_nodes_success(self):
        """Test successful node extraction."""
        workflow_data = json.loads(SAMPLE_WORKFLOW_JSON)
        nodes = extract_nodes(workflow_data)
        assert isinstance(nodes, list)
        assert len(nodes) == 3
        assert nodes[0]["name"] == "Webhook"
    
    def test_extract_nodes_empty(self):
        """Test extraction from workflow with no nodes."""
        workflow_data = {"nodes": [], "connections": {}}
        nodes = extract_nodes(workflow_data)
        assert nodes == []
    
    def test_extract_nodes_invalid_type(self):
        """Test extraction with invalid nodes type."""
        workflow_data = {"nodes": "not a list", "connections": {}}
        nodes = extract_nodes(workflow_data)
        assert nodes == []


class TestExtractConnections:
    """Tests for extracting connections from workflow."""
    
    def test_extract_connections_success(self):
        """Test successful connection extraction."""
        workflow_data = json.loads(SAMPLE_WORKFLOW_JSON)
        connections = extract_connections(workflow_data)
        assert isinstance(connections, dict)
        assert "Webhook" in connections
    
    def test_extract_connections_empty(self):
        """Test extraction from workflow with no connections."""
        workflow_data = {"nodes": [], "connections": {}}
        connections = extract_connections(workflow_data)
        assert connections == {}
    
    def test_extract_connections_invalid_type(self):
        """Test extraction with invalid connections type."""
        workflow_data = {"nodes": [], "connections": "not a dict"}
        connections = extract_connections(workflow_data)
        assert connections == {}


class TestIdentifyCustomNodes:
    """Tests for identifying custom nodes."""
    
    def test_identify_custom_nodes_function(self):
        """Test identification of Function nodes."""
        nodes = [
            {
                "name": "Function Node",
                "type": "n8n-nodes-base.function",
            },
            {
                "name": "Regular Node",
                "type": "n8n-nodes-base.httpRequest",
            },
        ]
        custom_nodes = identify_custom_nodes(nodes)
        assert "Function Node" in custom_nodes
        assert len(custom_nodes) == 1
    
    def test_identify_custom_nodes_code(self):
        """Test identification of Code nodes."""
        nodes = [
            {
                "name": "Code Node",
                "type": "n8n-nodes-base.code",
            },
        ]
        custom_nodes = identify_custom_nodes(nodes)
        assert "Code Node" in custom_nodes
    
    def test_identify_custom_nodes_with_code(self):
        """Test identification of nodes with custom code."""
        nodes = [
            {
                "name": "Custom Code Node",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "functionCode": "return items;",
                },
            },
        ]
        custom_nodes = identify_custom_nodes(nodes)
        assert "Custom Code Node" in custom_nodes
    
    def test_identify_custom_nodes_empty(self):
        """Test identification with no custom nodes."""
        nodes = [
            {
                "name": "Regular Node",
                "type": "n8n-nodes-base.httpRequest",
            },
        ]
        custom_nodes = identify_custom_nodes(nodes)
        assert custom_nodes == []


class TestExtractWorkflowMetadata:
    """Tests for extracting workflow metadata."""
    
    def test_extract_metadata_success(self):
        """Test successful metadata extraction."""
        workflow_data = json.loads(SAMPLE_WORKFLOW_JSON)
        metadata = extract_workflow_metadata(workflow_data)
        assert metadata["node_count"] == 3
        assert metadata["custom_node_count"] == 1
        assert metadata["workflow_name"] == "Test Workflow"
        assert "complexity" in metadata
        assert metadata["complexity"] in ["simple", "moderate", "complex"]
    
    def test_extract_metadata_simple_complexity(self):
        """Test complexity classification for simple workflow."""
        workflow_data = json.loads(SIMPLE_WORKFLOW_JSON)
        metadata = extract_workflow_metadata(workflow_data)
        assert metadata["complexity"] == "simple"
    
    def test_extract_metadata_custom_nodes(self):
        """Test custom nodes identification in metadata."""
        workflow_data = json.loads(SAMPLE_WORKFLOW_JSON)
        metadata = extract_workflow_metadata(workflow_data)
        assert "Function Node" in metadata["custom_nodes"]


class TestBuildAnalysisPrompt:
    """Tests for building analysis prompt."""
    
    def test_build_prompt_success(self):
        """Test successful prompt building."""
        prompt = build_analysis_prompt(SAMPLE_WORKFLOW_JSON)
        assert isinstance(prompt, str)
        assert len(prompt) > len(SAMPLE_WORKFLOW_JSON)
        assert SAMPLE_WORKFLOW_JSON in prompt
    
    @patch("src.mcp_server.tools.analyze.load_step1_prompt_template")
    def test_build_prompt_template_error(self, mock_load):
        """Test handling of template loading errors."""
        mock_load.side_effect = IOError("Template not found")
        with pytest.raises(IOError):
            build_analysis_prompt(SAMPLE_WORKFLOW_JSON)


class TestGenerateRequirements:
    """Tests for generating requirements using LLM."""
    
    def test_generate_requirements_success(self):
        """Test successful requirements generation."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        mock_provider.generate.return_value = GenerationResponse(
            content="Test requirements output",
            model="gpt-4o-mini",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        
        result = generate_requirements(SAMPLE_WORKFLOW_JSON, mock_provider)
        assert result == "Test requirements output"
        mock_provider.generate.assert_called_once()
    
    def test_generate_requirements_not_initialized(self):
        """Test initialization of uninitialized provider."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = False
        mock_provider.generate.return_value = GenerationResponse(
            content="Test requirements",
            model="gpt-4o-mini",
        )
        
        generate_requirements(SAMPLE_WORKFLOW_JSON, mock_provider)
        mock_provider.initialize.assert_called_once()
    
    def test_generate_requirements_error(self):
        """Test handling of LLM generation errors."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        mock_provider.generate.side_effect = Exception("API error")
        
        with pytest.raises(RuntimeError, match="Failed to generate requirements"):
            generate_requirements(SAMPLE_WORKFLOW_JSON, mock_provider)


class TestFormatLlmResponse:
    """Tests for formatting LLM response."""
    
    def test_format_response_with_metadata(self):
        """Test formatting response with metadata."""
        metadata = {
            "node_count": 5,
            "custom_node_count": 2,
            "custom_nodes": ["Node1", "Node2"],
            "complexity": "moderate",
            "workflow_name": "Test Workflow",
        }
        result = format_llm_response("Requirements text", metadata, include_metadata=True)
        
        assert result["status"] == "success"
        assert result["requirements"] == "Requirements text"
        assert "metadata" in result
        assert result["metadata"]["node_count"] == 5
        assert "analysis_timestamp" in result
    
    def test_format_response_without_metadata(self):
        """Test formatting response without metadata."""
        metadata = {"node_count": 5}
        result = format_llm_response("Requirements text", metadata, include_metadata=False)
        
        assert result["status"] == "success"
        assert result["requirements"] == "Requirements text"
        assert "metadata" not in result
        assert "analysis_timestamp" in result


class TestAnalyzeN8nWorkflowHandler:
    """Tests for analyze_n8n_workflow handler."""
    
    @pytest.fixture
    def mock_server(self):
        """Create a mock server instance."""
        server = Mock()
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        mock_provider.generate.return_value = GenerationResponse(
            content="Generated requirements text",
            model="gpt-4o-mini",
        )
        server._get_llm_provider.return_value = mock_provider
        set_server_instance(server)
        return server
    
    @pytest.mark.asyncio
    async def test_handler_success(self, mock_server):
        """Test successful workflow analysis."""
        result = await analyze_n8n_workflow(SAMPLE_WORKFLOW_JSON, include_metadata=True)
        
        assert result["status"] == "success"
        assert "requirements" in result
        assert "metadata" in result
        assert result["metadata"]["node_count"] == 3
    
    @pytest.mark.asyncio
    async def test_handler_invalid_json(self):
        """Test handler with invalid JSON."""
        result = await analyze_n8n_workflow(INVALID_JSON)
        
        assert result["status"] == "error"
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_handler_missing_fields(self):
        """Test handler with missing required fields."""
        result = await analyze_n8n_workflow(MISSING_FIELDS_JSON)
        
        assert result["status"] == "error"
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_handler_without_metadata(self, mock_server):
        """Test handler without metadata."""
        result = await analyze_n8n_workflow(SAMPLE_WORKFLOW_JSON, include_metadata=False)
        
        assert result["status"] == "success"
        assert "metadata" not in result
    
    @pytest.mark.asyncio
    async def test_handler_llm_provider_error(self):
        """Test handler when LLM provider fails."""
        # Set server instance with failing provider
        server = Mock()
        server._get_llm_provider.side_effect = RuntimeError("Provider unavailable")
        set_server_instance(server)
        
        result = await analyze_n8n_workflow(SAMPLE_WORKFLOW_JSON)
        
        assert result["status"] == "error"
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_handler_llm_generation_error(self, mock_server):
        """Test handler when LLM generation fails."""
        # Make provider generate fail
        mock_provider = mock_server._get_llm_provider.return_value
        mock_provider.generate.side_effect = Exception("LLM API error")
        
        result = await analyze_n8n_workflow(SAMPLE_WORKFLOW_JSON)
        
        assert result["status"] == "error"
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_handler_fallback_provider(self):
        """Test handler fallback to environment provider."""
        # Set server instance to None to trigger fallback
        set_server_instance(None)
        
        with patch("src.mcp_server.tools.handlers.create_from_env") as mock_create:
            mock_provider = Mock()
            mock_provider.is_initialized.return_value = True
            mock_provider.generate.return_value = GenerationResponse(
                content="Fallback requirements",
                model="gpt-4o-mini",
            )
            mock_create.return_value = mock_provider
            
            result = await analyze_n8n_workflow(SAMPLE_WORKFLOW_JSON)
            
            assert result["status"] == "success"
            mock_create.assert_called_once()


class TestGetLlmProvider:
    """Tests for _get_llm_provider helper function."""
    
    def test_get_provider_from_server(self):
        """Test getting provider from server instance."""
        mock_server = Mock()
        mock_provider = Mock()
        mock_server._get_llm_provider.return_value = mock_provider
        set_server_instance(mock_server)
        
        provider = _get_llm_provider()
        assert provider == mock_provider
    
    def test_get_provider_fallback(self):
        """Test fallback to environment provider."""
        set_server_instance(None)
        
        with patch("src.mcp_server.tools.handlers.create_from_env") as mock_create:
            mock_provider = Mock()
            mock_create.return_value = mock_provider
            
            provider = _get_llm_provider()
            assert provider == mock_provider
            mock_create.assert_called_once()
    
    def test_get_provider_server_error(self):
        """Test fallback when server provider fails."""
        mock_server = Mock()
        mock_server._get_llm_provider.side_effect = Exception("Server error")
        set_server_instance(mock_server)
        
        with patch("src.mcp_server.tools.handlers.create_from_env") as mock_create:
            mock_provider = Mock()
            mock_create.return_value = mock_provider
            
            provider = _get_llm_provider()
            assert provider == mock_provider
    
    def test_get_provider_all_fail(self):
        """Test error when all provider sources fail."""
        set_server_instance(None)
        
        with patch("src.mcp_server.tools.handlers.create_from_env") as mock_create:
            mock_create.side_effect = Exception("All providers failed")
            
            with pytest.raises(RuntimeError):
                _get_llm_provider()
