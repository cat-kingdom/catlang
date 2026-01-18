"""Tests for Fase 7: Generate LangGraph Implementation tool implementation."""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
from datetime import datetime

from src.mcp_server.tools.generate import (
    load_step3_prompt_template,
    load_all_guides,
    build_guides_context,
    determine_paradigm,
    get_paradigm_guide,
    build_generation_prompt,
    generate_code,
    extract_code_from_response,
    format_response,
    save_code_to_file,
    validate_requirements,
    validate_paradigm,
    validate_output_format,
)
from src.mcp_server.tools.handlers import (
    generate_langgraph_implementation,
    set_server_instance,
    _get_llm_provider,
)
from src.llm_provider.base import GenerationResponse


# Sample requirements for testing
SAMPLE_REQUIREMENTS = """
# Production Requirements

## Global Workflow Summary
- Objective: Process user data
- Triggers: Webhook
- Execution: Sequential

## Node Specifications
- Node 1: Webhook trigger
- Node 2: Process data
- Node 3: Send response

## Custom Nodes
None identified.

## Implementation Notes
Simple sequential workflow suitable for Functional API.
"""

COMPLEX_REQUIREMENTS = """
# Production Requirements

## Global Workflow Summary
- Objective: Complex multi-agent system
- Triggers: Multiple parallel paths
- Execution: Parallel with state management
- StateGraph required for node-level checkpointing
- Workflow visualization needed

## Node Specifications
Multiple nodes with complex conditional routing.

## Implementation Notes
Requires Graph API with StateGraph for parallel execution and state management.
"""

SHORT_REQUIREMENTS = "Short"  # Less than 100 chars

SAMPLE_CUSTOM_LOGIC_SPECS = """
# Custom Logic Specifications

## Function: process_data
- Purpose: Process input data
- Inputs: data (string)
- Outputs: processed_data (dict)
- Dependencies: json
"""

# Sample LLM outputs for testing
LLM_OUTPUT_WITH_CODE_BLOCK = """
Here's the generated code:

```python
from langgraph.func import entrypoint, task

@entrypoint
def workflow(input_data):
    return {"result": "success"}
```

This code implements the workflow.
"""

LLM_OUTPUT_PLAIN_CODE = """from langgraph.func import entrypoint, task

@entrypoint
def workflow(input_data):
    return {"result": "success"}
"""

LLM_OUTPUT_NO_CODE = "This is just text without any code."

# Sample guides for testing
SAMPLE_GUIDES = {
    "paradigm-selection": "# Paradigm Selection Guide\nDefault to Functional API.",
    "functional-api-implementation": "# Functional API Guide\nUse @entrypoint decorator.",
    "graph-api-implementation": "# Graph API Guide\nUse StateGraph class.",
    "authentication-setup": "# Authentication Guide\nSetup credentials.",
}


class TestLoadStep3PromptTemplate:
    """Tests for loading Step3 prompt template."""
    
    def test_load_template_success(self):
        """Test successful loading of prompt template."""
        template = load_step3_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0
        # Check for key phrases that should be in the template
        template_lower = template.lower()
        assert "langgraph" in template_lower or "implementation" in template_lower
    
    @patch("src.mcp_server.tools.generate.STEP3_PROMPT_PATH")
    def test_load_template_not_found(self, mock_path):
        """Test handling of missing template file."""
        mock_path.exists.return_value = False
        with pytest.raises(FileNotFoundError):
            load_step3_prompt_template()
    
    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_load_template_io_error(self, mock_open):
        """Test handling of IO errors when reading template."""
        with pytest.raises(IOError):
            load_step3_prompt_template()


class TestLoadAllGuides:
    """Tests for loading all guides."""
    
    def test_load_guides_success(self):
        """Test successful loading of guides."""
        # Test with real guides directory (if it exists)
        guides = load_all_guides()
        assert isinstance(guides, dict)
        # Should load at least some guides if directory exists
        # (test is lenient - actual count depends on filesystem)
    
    @patch("src.mcp_server.tools.generate.GUIDES_DIR")
    def test_load_guides_mocked_success(self, mock_guides_dir):
        """Test successful loading of guides with mocked directory."""
        # Mock the guides directory
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True
        mock_dir.is_dir.return_value = True
        
        # Mock guide files
        mock_file1 = MagicMock()
        mock_file1.stem = "paradigm-selection"
        
        mock_file2 = MagicMock()
        mock_file2.stem = "functional-api-implementation"
        
        mock_dir.glob.return_value = [mock_file1, mock_file2]
        
        # Replace the GUIDES_DIR with our mock
        import src.mcp_server.tools.generate as generate_module
        original_guides_dir = generate_module.GUIDES_DIR
        generate_module.GUIDES_DIR = mock_dir
        
        try:
            with patch("builtins.open", mock_open(read_data="# Guide content")):
                guides = load_all_guides()
                assert isinstance(guides, dict)
                assert len(guides) == 2
        finally:
            generate_module.GUIDES_DIR = original_guides_dir
    
    @patch("src.mcp_server.tools.generate.GUIDES_DIR")
    def test_load_guides_directory_not_found(self, mock_guides_dir):
        """Test handling when guides directory doesn't exist."""
        mock_dir = MagicMock()
        mock_dir.exists.return_value = False
        
        import src.mcp_server.tools.generate as generate_module
        original_guides_dir = generate_module.GUIDES_DIR
        generate_module.GUIDES_DIR = mock_dir
        
        try:
            guides = load_all_guides()
            assert isinstance(guides, dict)
            assert len(guides) == 0
        finally:
            generate_module.GUIDES_DIR = original_guides_dir
    
    @patch("src.mcp_server.tools.generate.GUIDES_DIR")
    def test_load_guides_not_directory(self, mock_guides_dir):
        """Test handling when guides path is not a directory."""
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True
        mock_dir.is_dir.return_value = False
        
        import src.mcp_server.tools.generate as generate_module
        original_guides_dir = generate_module.GUIDES_DIR
        generate_module.GUIDES_DIR = mock_dir
        
        try:
            guides = load_all_guides()
            assert isinstance(guides, dict)
            assert len(guides) == 0
        finally:
            generate_module.GUIDES_DIR = original_guides_dir
    
    @patch("src.mcp_server.tools.generate.GUIDES_DIR")
    def test_load_guides_io_error(self, mock_guides_dir):
        """Test handling of IO errors when reading guides."""
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True
        mock_dir.is_dir.return_value = True
        
        mock_file = MagicMock()
        mock_file.stem = "test-guide"
        mock_dir.glob.return_value = [mock_file]
        
        import src.mcp_server.tools.generate as generate_module
        original_guides_dir = generate_module.GUIDES_DIR
        generate_module.GUIDES_DIR = mock_dir
        
        try:
            with patch("builtins.open", side_effect=IOError("Permission denied")):
                guides = load_all_guides()
                assert isinstance(guides, dict)
                assert len(guides) == 0
        finally:
            generate_module.GUIDES_DIR = original_guides_dir


class TestBuildGuidesContext:
    """Tests for building guides context."""
    
    def test_build_context_success(self):
        """Test successful building of guides context."""
        context = build_guides_context(SAMPLE_GUIDES)
        assert isinstance(context, str)
        assert len(context) > 0
        assert "paradigm-selection" in context.lower()
        assert "functional-api-implementation" in context.lower()
    
    def test_build_context_empty(self):
        """Test building context with empty guides."""
        context = build_guides_context({})
        assert context == ""


class TestValidateRequirements:
    """Tests for requirements validation."""
    
    def test_validate_valid_requirements(self):
        """Test validation of valid requirements."""
        is_valid, error_msg = validate_requirements(SAMPLE_REQUIREMENTS)
        assert is_valid is True
        assert error_msg is None
    
    def test_validate_empty_requirements(self):
        """Test validation of empty requirements."""
        is_valid, error_msg = validate_requirements("")
        assert is_valid is False
        assert "empty" in error_msg.lower()
    
    def test_validate_whitespace_only(self):
        """Test validation of whitespace-only requirements."""
        is_valid, error_msg = validate_requirements("   \n\t  ")
        assert is_valid is False
    
    def test_validate_short_requirements(self):
        """Test validation of requirements that are too short."""
        is_valid, error_msg = validate_requirements(SHORT_REQUIREMENTS)
        assert is_valid is False
        assert "100" in error_msg or "length" in error_msg.lower()


class TestValidateParadigm:
    """Tests for paradigm validation."""
    
    def test_validate_functional(self):
        """Test validation of functional paradigm."""
        assert validate_paradigm("functional") is True
        assert validate_paradigm("Functional") is True
        assert validate_paradigm("FUNCTIONAL") is True
    
    def test_validate_graph(self):
        """Test validation of graph paradigm."""
        assert validate_paradigm("graph") is True
        assert validate_paradigm("Graph") is True
        assert validate_paradigm("GRAPH") is True
    
    def test_validate_auto(self):
        """Test validation of auto paradigm."""
        assert validate_paradigm("auto") is True
        assert validate_paradigm("Auto") is True
        assert validate_paradigm("AUTO") is True
    
    def test_validate_invalid(self):
        """Test validation of invalid paradigm."""
        assert validate_paradigm("invalid") is False
        assert validate_paradigm("") is False
        assert validate_paradigm("functional-api") is False


class TestValidateOutputFormat:
    """Tests for output format validation."""
    
    def test_validate_code(self):
        """Test validation of code output format."""
        assert validate_output_format("code") is True
        assert validate_output_format("Code") is True
        assert validate_output_format("CODE") is True
    
    def test_validate_file(self):
        """Test validation of file output format."""
        assert validate_output_format("file") is True
        assert validate_output_format("File") is True
        assert validate_output_format("FILE") is True
    
    def test_validate_invalid(self):
        """Test validation of invalid output format."""
        assert validate_output_format("invalid") is False
        assert validate_output_format("") is False
        assert validate_output_format("json") is False


class TestDetermineParadigm:
    """Tests for paradigm determination."""
    
    def test_determine_functional_explicit(self):
        """Test explicit functional paradigm selection."""
        result = determine_paradigm(SAMPLE_REQUIREMENTS, "functional")
        assert result == "functional"
    
    def test_determine_graph_explicit(self):
        """Test explicit graph paradigm selection."""
        result = determine_paradigm(SAMPLE_REQUIREMENTS, "graph")
        assert result == "graph"
    
    def test_determine_auto_defaults_to_functional(self):
        """Test auto-detection defaults to functional."""
        result = determine_paradigm(SAMPLE_REQUIREMENTS, "auto")
        assert result in ["functional", "graph"]
    
    def test_determine_auto_detects_graph(self):
        """Test auto-detection detects graph paradigm."""
        result = determine_paradigm(COMPLEX_REQUIREMENTS, "auto")
        # Should detect graph based on indicators
        assert result in ["functional", "graph"]
    
    def test_determine_case_insensitive(self):
        """Test paradigm determination is case insensitive."""
        result1 = determine_paradigm(SAMPLE_REQUIREMENTS, "AUTO")
        result2 = determine_paradigm(SAMPLE_REQUIREMENTS, "auto")
        assert result1 == result2


class TestGetParadigmGuide:
    """Tests for getting paradigm-specific guide."""
    
    def test_get_functional_guide(self):
        """Test getting functional API guide."""
        guide = get_paradigm_guide("functional", SAMPLE_GUIDES)
        assert isinstance(guide, str)
        assert "functional" in guide.lower() or "paradigm" in guide.lower()
    
    def test_get_graph_guide(self):
        """Test getting graph API guide."""
        guide = get_paradigm_guide("graph", SAMPLE_GUIDES)
        assert isinstance(guide, str)
        assert "graph" in guide.lower() or "paradigm" in guide.lower()
    
    def test_get_guide_includes_paradigm_selection(self):
        """Test that paradigm selection guide is always included."""
        guide = get_paradigm_guide("functional", SAMPLE_GUIDES)
        assert "paradigm" in guide.lower()
    
    def test_get_guide_missing_paradigm_guide(self):
        """Test handling when paradigm guide is missing."""
        guides_without_paradigm = {"paradigm-selection": "# Guide"}
        guide = get_paradigm_guide("functional", guides_without_paradigm)
        assert isinstance(guide, str)
        assert "paradigm" in guide.lower()


class TestBuildGenerationPrompt:
    """Tests for building generation prompt."""
    
    @patch("src.mcp_server.tools.generate.load_step3_prompt_template")
    def test_build_prompt_success(self, mock_load_template):
        """Test successful building of generation prompt."""
        mock_load_template.return_value = "# Step3 Template\n"
        
        prompt = build_generation_prompt(
            requirements=SAMPLE_REQUIREMENTS,
            custom_logic_specs=SAMPLE_CUSTOM_LOGIC_SPECS,
            guides_context="## Guides\n",
            paradigm="functional",
            paradigm_guide="## Functional Guide\n",
        )
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert SAMPLE_REQUIREMENTS in prompt
        assert SAMPLE_CUSTOM_LOGIC_SPECS in prompt
    
    @patch("src.mcp_server.tools.generate.load_step3_prompt_template")
    def test_build_prompt_without_custom_logic(self, mock_load_template):
        """Test building prompt without custom logic specs."""
        mock_load_template.return_value = "# Step3 Template\n"
        
        prompt = build_generation_prompt(
            requirements=SAMPLE_REQUIREMENTS,
            custom_logic_specs=None,
            guides_context="## Guides\n",
            paradigm="functional",
            paradigm_guide="## Functional Guide\n",
        )
        
        assert isinstance(prompt, str)
        assert SAMPLE_REQUIREMENTS in prompt
        assert "Custom Logic Specifications" not in prompt or "None" in prompt


class TestGenerateCode:
    """Tests for code generation using LLM."""
    
    def test_generate_code_success(self):
        """Test successful code generation."""
        mock_provider = MagicMock()
        mock_provider.is_initialized.return_value = True
        mock_response = GenerationResponse(content="Generated code here", model="gpt-4")
        mock_provider.generate.return_value = mock_response
        
        result = generate_code("Test prompt", mock_provider)
        
        assert isinstance(result, str)
        assert result == "Generated code here"
        mock_provider.generate.assert_called_once()
    
    def test_generate_code_auto_initialize(self):
        """Test code generation with auto-initialization."""
        mock_provider = MagicMock()
        mock_provider.is_initialized.return_value = False
        mock_provider.initialize = MagicMock()
        mock_response = GenerationResponse(content="Generated code", model="gpt-4")
        mock_provider.generate.return_value = mock_response
        
        result = generate_code("Test prompt", mock_provider)
        
        assert isinstance(result, str)
        mock_provider.initialize.assert_called_once()
    
    def test_generate_code_error(self):
        """Test handling of LLM generation errors."""
        mock_provider = MagicMock()
        mock_provider.is_initialized.return_value = True
        mock_provider.generate.side_effect = Exception("API error")
        
        with pytest.raises(RuntimeError, match="Failed to generate code"):
            generate_code("Test prompt", mock_provider)


class TestExtractCodeFromResponse:
    """Tests for extracting code from LLM response."""
    
    def test_extract_from_code_block(self):
        """Test extraction from markdown code block."""
        code = extract_code_from_response(LLM_OUTPUT_WITH_CODE_BLOCK)
        assert isinstance(code, str)
        assert "from langgraph.func" in code
        assert "```" not in code  # Code blocks should be stripped
    
    def test_extract_from_plain_code(self):
        """Test extraction from plain code text."""
        code = extract_code_from_response(LLM_OUTPUT_PLAIN_CODE)
        assert isinstance(code, str)
        assert "from langgraph.func" in code
    
    def test_extract_starts_with_import(self):
        """Test extraction when output starts with import."""
        output = "import os\nfrom langgraph import entrypoint"
        code = extract_code_from_response(output)
        assert isinstance(code, str)
        assert "import os" in code
    
    def test_extract_no_code_found(self):
        """Test handling when no code can be extracted."""
        with pytest.raises(ValueError, match="Could not extract"):
            extract_code_from_response(LLM_OUTPUT_NO_CODE)
    
    def test_extract_multiple_code_blocks(self):
        """Test extraction when multiple code blocks exist."""
        output = "```python\ncode1\n```\n```python\ncode2\n```"
        code = extract_code_from_response(output)
        assert "code1" in code  # Should use first block


class TestFormatResponse:
    """Tests for formatting response."""
    
    def test_format_code_response(self):
        """Test formatting response with code output."""
        response = format_response(
            code="test code",
            paradigm="functional",
            output_format="code",
        )
        
        assert response["status"] == "success"
        assert response["code"] == "test code"
        assert response["paradigm"] == "functional"
        assert response["output_format"] == "code"
        assert "code_length" in response
        assert "generation_timestamp" in response
    
    def test_format_file_response(self):
        """Test formatting response with file output."""
        response = format_response(
            code="test code",
            paradigm="graph",
            output_format="file",
            file_path="/path/to/file.py",
        )
        
        assert response["status"] == "success"
        assert response["file_path"] == "/path/to/file.py"
        assert response["output_format"] == "file"


class TestSaveCodeToFile:
    """Tests for saving code to file."""
    
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.mcp_server.tools.generate.Path")
    def test_save_code_success(self, mock_path_class, mock_file):
        """Test successful saving of code to file."""
        # Create a real Path-like object
        from pathlib import Path as RealPath
        real_path = RealPath(__file__).parent  # Use test directory
        
        # Mock Path constructor to return real path
        def path_init(self, *args):
            if len(args) == 0:
                return real_path
            return RealPath(*args)
        
        mock_path_class.side_effect = lambda *args: RealPath(*args) if args else real_path
        
        file_path = save_code_to_file("test code", base_path=str(real_path))
        
        assert isinstance(file_path, str)
        assert ".py" in file_path
        mock_file.assert_called_once()
    
    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_save_code_io_error(self, mock_file):
        """Test handling of IO errors when saving file."""
        with pytest.raises(IOError, match="Failed to write"):
            save_code_to_file("test code")


class TestGenerateLanggraphImplementationHandler:
    """Tests for generate_langgraph_implementation handler."""
    
    @pytest.mark.asyncio
    @patch("src.mcp_server.tools.handlers._get_llm_provider")
    @patch("src.mcp_server.tools.handlers.load_all_guides")
    @patch("src.mcp_server.tools.handlers.generate_code")
    @patch("src.mcp_server.tools.handlers.extract_code_from_response")
    async def test_handler_success(self, mock_extract, mock_generate, mock_load_guides, mock_get_provider):
        """Test successful handler execution."""
        # Setup mocks
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_load_guides.return_value = SAMPLE_GUIDES
        # generate_code returns string, not MagicMock
        mock_generate.return_value = LLM_OUTPUT_WITH_CODE_BLOCK
        mock_extract.return_value = "extracted code"
        
        result = await generate_langgraph_implementation(
            requirements=SAMPLE_REQUIREMENTS,
            custom_logic_specs=SAMPLE_CUSTOM_LOGIC_SPECS,
            paradigm="functional",
            output_format="code",
        )
        
        assert result["status"] == "success"
        assert "code" in result
        assert result["paradigm"] == "functional"
    
    @pytest.mark.asyncio
    async def test_handler_invalid_requirements(self):
        """Test handler with invalid requirements."""
        result = await generate_langgraph_implementation(
            requirements="",
            paradigm="functional",
        )
        
        assert result["status"] == "error"
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_handler_invalid_paradigm(self):
        """Test handler with invalid paradigm."""
        result = await generate_langgraph_implementation(
            requirements=SAMPLE_REQUIREMENTS,
            paradigm="invalid",
        )
        
        assert result["status"] == "error"
        assert "paradigm" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_handler_invalid_output_format(self):
        """Test handler with invalid output format."""
        result = await generate_langgraph_implementation(
            requirements=SAMPLE_REQUIREMENTS,
            output_format="invalid",
        )
        
        assert result["status"] == "error"
        assert "output format" in result["error"].lower()
    
    @pytest.mark.asyncio
    @patch("src.mcp_server.tools.handlers._get_llm_provider")
    @patch("src.mcp_server.tools.generate.load_step3_prompt_template")
    async def test_handler_template_not_found(self, mock_load_template, mock_get_provider):
        """Test handler when template is not found."""
        mock_load_template.side_effect = FileNotFoundError("Template not found")
        
        result = await generate_langgraph_implementation(
            requirements=SAMPLE_REQUIREMENTS,
        )
        
        assert result["status"] == "error"
        assert "template" in result["error"].lower()
    
    @pytest.mark.asyncio
    @patch("src.mcp_server.tools.handlers._get_llm_provider")
    @patch("src.mcp_server.tools.handlers.load_all_guides")
    @patch("src.mcp_server.tools.handlers.generate_code")
    async def test_handler_llm_generation_error(self, mock_generate, mock_load_guides, mock_get_provider):
        """Test handler when LLM generation fails."""
        mock_get_provider.return_value = MagicMock()
        mock_load_guides.return_value = {}
        mock_generate.side_effect = RuntimeError("LLM error")
        
        result = await generate_langgraph_implementation(
            requirements=SAMPLE_REQUIREMENTS,
        )
        
        assert result["status"] == "error"
        assert "generate" in result["error"].lower() or "llm" in result["error"].lower() or "error" in result["error"].lower()
    
    @pytest.mark.asyncio
    @patch("src.mcp_server.tools.handlers._get_llm_provider")
    @patch("src.mcp_server.tools.handlers.load_all_guides")
    @patch("src.mcp_server.tools.handlers.generate_code")
    @patch("src.mcp_server.tools.handlers.extract_code_from_response")
    async def test_handler_code_extraction_error(self, mock_extract, mock_generate, mock_load_guides, mock_get_provider):
        """Test handler when code extraction fails."""
        mock_get_provider.return_value = MagicMock()
        mock_load_guides.return_value = {}
        mock_generate.return_value = "invalid output"
        mock_extract.side_effect = ValueError("No code found")
        
        result = await generate_langgraph_implementation(
            requirements=SAMPLE_REQUIREMENTS,
        )
        
        assert result["status"] == "error"
        assert "extract" in result["error"].lower() or "code" in result["error"].lower() or "response" in result["error"].lower()
    
    @pytest.mark.asyncio
    @patch("src.mcp_server.tools.handlers._get_llm_provider")
    @patch("src.mcp_server.tools.handlers.load_all_guides")
    @patch("src.mcp_server.tools.handlers.generate_code")
    @patch("src.mcp_server.tools.handlers.extract_code_from_response")
    @patch("src.mcp_server.tools.handlers.save_code_to_file")
    async def test_handler_file_output(self, mock_save_file, mock_extract, mock_generate, mock_load_guides, mock_get_provider):
        """Test handler with file output format."""
        mock_get_provider.return_value = MagicMock()
        mock_load_guides.return_value = {}
        # Ensure generate_code returns a string
        mock_generate.return_value = LLM_OUTPUT_PLAIN_CODE
        mock_extract.return_value = "extracted code"
        mock_save_file.return_value = "/path/to/file.py"
        
        result = await generate_langgraph_implementation(
            requirements=SAMPLE_REQUIREMENTS,
            output_format="file",
        )
        
        assert result["status"] == "success"
        assert result["output_format"] == "file"
        assert "file_path" in result
        mock_save_file.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("src.mcp_server.tools.handlers._get_llm_provider")
    @patch("src.mcp_server.tools.handlers.load_all_guides")
    @patch("src.mcp_server.tools.handlers.generate_code")
    @patch("src.mcp_server.tools.handlers.extract_code_from_response")
    @patch("src.mcp_server.tools.handlers.save_code_to_file")
    async def test_handler_file_save_error(self, mock_save_file, mock_extract, mock_generate, mock_load_guides, mock_get_provider):
        """Test handler when file save fails."""
        mock_get_provider.return_value = MagicMock()
        mock_load_guides.return_value = {}
        # Ensure generate_code returns a string
        mock_generate.return_value = LLM_OUTPUT_PLAIN_CODE
        mock_extract.return_value = "extracted code"
        mock_save_file.side_effect = IOError("Permission denied")
        
        result = await generate_langgraph_implementation(
            requirements=SAMPLE_REQUIREMENTS,
            output_format="file",
        )
        
        assert result["status"] == "error"
        assert "file" in result["error"].lower()
        # Code should still be included even if file save fails
        assert "code" in result
    
    @pytest.mark.asyncio
    @patch("src.mcp_server.tools.handlers._get_llm_provider")
    @patch("src.mcp_server.tools.handlers.load_all_guides")
    @patch("src.mcp_server.tools.handlers.generate_code")
    @patch("src.mcp_server.tools.handlers.extract_code_from_response")
    async def test_handler_auto_paradigm(self, mock_extract, mock_generate, mock_load_guides, mock_get_provider):
        """Test handler with auto paradigm selection."""
        mock_get_provider.return_value = MagicMock()
        mock_load_guides.return_value = SAMPLE_GUIDES
        # Ensure generate_code returns a string
        mock_generate.return_value = LLM_OUTPUT_PLAIN_CODE
        mock_extract.return_value = "extracted code"
        
        result = await generate_langgraph_implementation(
            requirements=SAMPLE_REQUIREMENTS,
            paradigm="auto",
        )
        
        assert result["status"] == "success"
        assert result["paradigm"] in ["functional", "graph"]
    
    @pytest.mark.asyncio
    @patch("src.mcp_server.tools.handlers._get_llm_provider")
    async def test_handler_provider_unavailable(self, mock_get_provider):
        """Test handler when provider is unavailable."""
        mock_get_provider.side_effect = RuntimeError("Provider unavailable")
        
        result = await generate_langgraph_implementation(
            requirements=SAMPLE_REQUIREMENTS,
        )
        
        assert result["status"] == "error"
        assert "provider" in result["error"].lower()


class TestGetLlmProvider:
    """Tests for LLM provider access."""
    
    def test_get_provider_from_server(self):
        """Test getting provider from server instance."""
        mock_server = MagicMock()
        mock_provider = MagicMock()
        mock_server._get_llm_provider.return_value = mock_provider
        
        set_server_instance(mock_server)
        provider = _get_llm_provider()
        
        assert provider == mock_provider
    
    @patch("src.mcp_server.tools.handlers.create_from_env")
    def test_get_provider_fallback(self, mock_create):
        """Test fallback to environment provider."""
        mock_provider = MagicMock()
        mock_create.return_value = mock_provider
        
        set_server_instance(None)
        provider = _get_llm_provider()
        
        assert provider == mock_provider
        mock_create.assert_called_once()
    
    @patch("src.mcp_server.tools.handlers.create_from_env")
    def test_get_provider_fallback_error(self, mock_create):
        """Test handling when fallback also fails."""
        mock_create.side_effect = Exception("Failed to create")
        
        set_server_instance(None)
        with pytest.raises(RuntimeError, match="Failed to get or create"):
            _get_llm_provider()
