"""Tests for Fase 6: Extract Custom Logic tool implementation."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.mcp_server.tools.extract import (
    load_step2_prompt_template,
    validate_language,
    validate_code,
    parse_python_code,
    parse_javascript_code,
    extract_dependencies,
    extract_functions,
    extract_classes,
    build_extraction_prompt,
    generate_specifications,
    format_llm_response,
)
from src.mcp_server.tools.handlers import (
    extract_custom_logic,
    set_server_instance,
    _get_llm_provider,
)
from src.llm_provider.base import GenerationResponse


# Sample Python code for testing
SAMPLE_PYTHON_CODE = """
import json
import requests
from datetime import datetime

def process_data(data):
    '''Process input data and return result.'''
    result = json.loads(data)
    return result

class DataProcessor:
    '''Process data with validation.'''
    
    def __init__(self):
        self.counter = 0
    
    def validate(self, data):
        '''Validate input data.'''
        return isinstance(data, dict)
"""

SIMPLE_PYTHON_CODE = """
def hello():
    return "world"
"""

PYTHON_WITH_IMPORTS = """
import os
import sys
import requests
import pandas as pd
from typing import List, Dict
"""

PYTHON_WITH_CLASSES = """
class MyClass:
    def method1(self):
        pass
    
    def method2(self, arg):
        return arg
"""

INVALID_PYTHON_CODE = "def invalid syntax here"

# Sample JavaScript code for testing
SAMPLE_JAVASCRIPT_CODE = """
const fs = require('fs');
const axios = require('axios');

function processData(data) {
    const result = JSON.parse(data);
    return result;
}

const arrowFunction = (x) => {
    return x * 2;
};

const anotherFunction = function(name) {
    return `Hello ${name}`;
};
"""

SIMPLE_JAVASCRIPT_CODE = """
function hello() {
    return "world";
}
"""

JAVASCRIPT_WITH_REQUIRES = """
const express = require('express');
const lodash = require('lodash');
import { useState } from 'react';
"""

JAVASCRIPT_WITH_FUNCTIONS = """
function func1() {}
const func2 = () => {};
let func3 = function() {};
"""

INVALID_JAVASCRIPT_CODE = "function { invalid }"


class TestLoadStep2PromptTemplate:
    """Tests for loading Step2 prompt template."""
    
    def test_load_template_success(self):
        """Test successful loading of prompt template."""
        template = load_step2_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0
        # Check for key phrases that should be in the template
        template_lower = template.lower()
        assert "python" in template_lower or "code" in template_lower
    
    @patch("src.mcp_server.tools.extract.STEP2_PROMPT_PATH")
    def test_load_template_not_found(self, mock_path):
        """Test handling of missing template file."""
        mock_path.exists.return_value = False
        with pytest.raises(FileNotFoundError):
            load_step2_prompt_template()
    
    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_load_template_io_error(self, mock_open):
        """Test handling of IO errors when reading template."""
        with pytest.raises(IOError):
            load_step2_prompt_template()


class TestValidateLanguage:
    """Tests for language validation."""
    
    def test_validate_python(self):
        """Test validation of Python language."""
        assert validate_language("python") is True
        assert validate_language("Python") is True
        assert validate_language("PYTHON") is True
    
    def test_validate_javascript(self):
        """Test validation of JavaScript language."""
        assert validate_language("javascript") is True
        assert validate_language("JavaScript") is True
        assert validate_language("JAVASCRIPT") is True
    
    def test_validate_invalid_language(self):
        """Test validation of invalid language."""
        assert validate_language("java") is False
        assert validate_language("ruby") is False
        assert validate_language("") is False


class TestValidateCode:
    """Tests for code validation."""
    
    def test_validate_python_valid(self):
        """Test validation of valid Python code."""
        is_valid, error_msg = validate_code(SIMPLE_PYTHON_CODE, "python")
        assert is_valid is True
        assert error_msg is None
    
    def test_validate_python_invalid_syntax(self):
        """Test validation of invalid Python syntax."""
        is_valid, error_msg = validate_code(INVALID_PYTHON_CODE, "python")
        assert is_valid is False
        assert error_msg is not None
        assert "syntax" in error_msg.lower()
    
    def test_validate_python_empty(self):
        """Test validation of empty Python code."""
        is_valid, error_msg = validate_code("", "python")
        assert is_valid is False
        assert "empty" in error_msg.lower()
    
    def test_validate_javascript_valid(self):
        """Test validation of valid JavaScript code."""
        is_valid, error_msg = validate_code(SIMPLE_JAVASCRIPT_CODE, "javascript")
        assert is_valid is True
        assert error_msg is None
    
    def test_validate_javascript_unbalanced_braces(self):
        """Test validation of JavaScript with unbalanced braces."""
        is_valid, error_msg = validate_code("function test() {", "javascript")
        assert is_valid is False
        assert "brace" in error_msg.lower()
    
    def test_validate_javascript_unbalanced_parens(self):
        """Test validation of JavaScript with unbalanced parentheses."""
        is_valid, error_msg = validate_code("function test( {", "javascript")
        assert is_valid is False
        # This code has unbalanced braces, so it should detect braces first
        assert "brace" in error_msg.lower() or "parenthes" in error_msg.lower()
    
    def test_validate_javascript_empty(self):
        """Test validation of empty JavaScript code."""
        is_valid, error_msg = validate_code("", "javascript")
        assert is_valid is False
        assert "empty" in error_msg.lower()
    
    def test_validate_unsupported_language(self):
        """Test validation with unsupported language."""
        is_valid, error_msg = validate_code("code", "java")
        assert is_valid is False
        assert "unsupported" in error_msg.lower()


class TestParsePythonCode:
    """Tests for parsing Python code."""
    
    def test_parse_simple_python(self):
        """Test parsing simple Python code."""
        result = parse_python_code(SIMPLE_PYTHON_CODE)
        assert isinstance(result, dict)
        assert "functions" in result
        assert "classes" in result
        assert "dependencies" in result
        assert len(result["functions"]) == 1
    
    def test_parse_python_with_imports(self):
        """Test parsing Python code with imports."""
        result = parse_python_code(PYTHON_WITH_IMPORTS)
        assert len(result["imports"]) > 0
        # Check that third-party dependencies are extracted
        assert "requests" in result["dependencies"]
        assert "pandas" in result["dependencies"]
        # Check that stdlib modules are not in dependencies
        assert "os" not in result["dependencies"]
        assert "sys" not in result["dependencies"]
    
    def test_parse_python_with_functions(self):
        """Test parsing Python code with functions."""
        result = parse_python_code(SAMPLE_PYTHON_CODE)
        assert len(result["functions"]) >= 1
        func = result["functions"][0]
        assert "name" in func
        assert "args" in func
        assert func["name"] == "process_data"
    
    def test_parse_python_with_classes(self):
        """Test parsing Python code with classes."""
        result = parse_python_code(PYTHON_WITH_CLASSES)
        assert len(result["classes"]) >= 1
        cls = result["classes"][0]
        assert "name" in cls
        assert "methods" in cls
        assert cls["name"] == "MyClass"
        assert len(cls["methods"]) == 2
    
    def test_parse_python_invalid_syntax(self):
        """Test parsing invalid Python syntax."""
        result = parse_python_code(INVALID_PYTHON_CODE)
        assert "parse_error" in result
        assert result["functions"] == []
        assert result["classes"] == []
    
    def test_parse_python_empty(self):
        """Test parsing empty Python code."""
        result = parse_python_code("")
        assert result["functions"] == []
        assert result["classes"] == []
        assert result["dependencies"] == []


class TestParseJavaScriptCode:
    """Tests for parsing JavaScript code."""
    
    def test_parse_simple_javascript(self):
        """Test parsing simple JavaScript code."""
        result = parse_javascript_code(SIMPLE_JAVASCRIPT_CODE)
        assert isinstance(result, dict)
        assert "functions" in result
        assert "dependencies" in result
        assert len(result["functions"]) >= 1
    
    def test_parse_javascript_with_requires(self):
        """Test parsing JavaScript code with require statements."""
        result = parse_javascript_code(JAVASCRIPT_WITH_REQUIRES)
        assert len(result["imports"]) > 0
        assert "express" in result["dependencies"]
        assert "lodash" in result["dependencies"]
    
    def test_parse_javascript_with_functions(self):
        """Test parsing JavaScript code with various function types."""
        result = parse_javascript_code(JAVASCRIPT_WITH_FUNCTIONS)
        assert len(result["functions"]) >= 3
        # Check that different function types are captured
        func_names = [f["name"] for f in result["functions"]]
        assert "func1" in func_names
        assert "func2" in func_names
        assert "func3" in func_names
    
    def test_parse_javascript_with_arrow_functions(self):
        """Test parsing JavaScript code with arrow functions."""
        result = parse_javascript_code(SAMPLE_JAVASCRIPT_CODE)
        assert len(result["functions"]) >= 1
        # Check for arrow function
        arrow_funcs = [f for f in result["functions"] if f.get("type") == "arrow"]
        assert len(arrow_funcs) >= 1
    
    def test_parse_javascript_empty(self):
        """Test parsing empty JavaScript code."""
        result = parse_javascript_code("")
        assert result["functions"] == []
        assert result["dependencies"] == []


class TestExtractDependencies:
    """Tests for dependency extraction."""
    
    def test_extract_dependencies_python(self):
        """Test extracting dependencies from Python code."""
        deps = extract_dependencies(PYTHON_WITH_IMPORTS, "python")
        assert isinstance(deps, list)
        assert "requests" in deps
        assert "pandas" in deps
        # Stdlib should not be included
        assert "os" not in deps
        assert "sys" not in deps
    
    def test_extract_dependencies_javascript(self):
        """Test extracting dependencies from JavaScript code."""
        deps = extract_dependencies(JAVASCRIPT_WITH_REQUIRES, "javascript")
        assert isinstance(deps, list)
        assert "express" in deps
        assert "lodash" in deps
    
    def test_extract_dependencies_empty(self):
        """Test extracting dependencies from code with no dependencies."""
        deps = extract_dependencies(SIMPLE_PYTHON_CODE, "python")
        assert deps == []
    
    def test_extract_dependencies_unsupported_language(self):
        """Test extracting dependencies with unsupported language."""
        deps = extract_dependencies("code", "java")
        assert deps == []


class TestExtractFunctions:
    """Tests for function extraction."""
    
    def test_extract_functions_python(self):
        """Test extracting functions from Python code."""
        funcs = extract_functions(SAMPLE_PYTHON_CODE, "python")
        assert isinstance(funcs, list)
        assert len(funcs) >= 1
        assert funcs[0]["name"] == "process_data"
    
    def test_extract_functions_javascript(self):
        """Test extracting functions from JavaScript code."""
        funcs = extract_functions(JAVASCRIPT_WITH_FUNCTIONS, "javascript")
        assert isinstance(funcs, list)
        assert len(funcs) >= 3
    
    def test_extract_functions_empty(self):
        """Test extracting functions from code with no functions."""
        funcs = extract_functions("x = 1", "python")
        assert funcs == []
    
    def test_extract_functions_unsupported_language(self):
        """Test extracting functions with unsupported language."""
        funcs = extract_functions("code", "java")
        assert funcs == []


class TestExtractClasses:
    """Tests for class extraction."""
    
    def test_extract_classes_python(self):
        """Test extracting classes from Python code."""
        classes = extract_classes(PYTHON_WITH_CLASSES, "python")
        assert isinstance(classes, list)
        assert len(classes) >= 1
        assert classes[0]["name"] == "MyClass"
        assert len(classes[0]["methods"]) == 2
    
    def test_extract_classes_javascript(self):
        """Test extracting classes from JavaScript code (not supported)."""
        classes = extract_classes(SAMPLE_JAVASCRIPT_CODE, "javascript")
        assert classes == []
    
    def test_extract_classes_empty(self):
        """Test extracting classes from code with no classes."""
        classes = extract_classes(SIMPLE_PYTHON_CODE, "python")
        assert classes == []


class TestBuildExtractionPrompt:
    """Tests for building extraction prompt."""
    
    def test_build_prompt_with_code(self):
        """Test building prompt with code replacement."""
        prompt = build_extraction_prompt(SIMPLE_PYTHON_CODE, "python")
        assert isinstance(prompt, str)
        assert SIMPLE_PYTHON_CODE in prompt
        assert "[PASTE CUSTOM NODE CODE HERE]" not in prompt
    
    def test_build_prompt_with_node_name(self):
        """Test building prompt with node name context."""
        prompt = build_extraction_prompt(SIMPLE_PYTHON_CODE, "python", "TestNode")
        assert "TestNode" in prompt
        assert "python" in prompt.lower() or "Python" in prompt
    
    def test_build_prompt_javascript(self):
        """Test building prompt for JavaScript code."""
        prompt = build_extraction_prompt(SIMPLE_JAVASCRIPT_CODE, "javascript")
        assert SIMPLE_JAVASCRIPT_CODE in prompt
    
    @patch("src.mcp_server.tools.extract.load_step2_prompt_template")
    def test_build_prompt_template_error(self, mock_load):
        """Test handling of template loading error."""
        mock_load.side_effect = IOError("Template not found")
        with pytest.raises(IOError):
            build_extraction_prompt(SIMPLE_PYTHON_CODE, "python")


class TestGenerateSpecifications:
    """Tests for generating specifications using LLM."""
    
    def test_generate_specifications_success(self):
        """Test successful generation of specifications."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        mock_response = Mock()
        mock_response.content = "Generated specifications text"
        mock_provider.generate.return_value = mock_response
        
        result = generate_specifications(
            SIMPLE_PYTHON_CODE, "python", mock_provider, "TestNode"
        )
        
        assert isinstance(result, str)
        assert result == "Generated specifications text"
        mock_provider.generate.assert_called_once()
    
    def test_generate_specifications_auto_initialize(self):
        """Test auto-initialization of provider."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = False
        mock_response = Mock()
        mock_response.content = "Generated specifications"
        mock_provider.generate.return_value = mock_response
        
        result = generate_specifications(
            SIMPLE_PYTHON_CODE, "python", mock_provider
        )
        
        mock_provider.initialize.assert_called_once()
        assert result == "Generated specifications"
    
    def test_generate_specifications_error(self):
        """Test handling of LLM generation error."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        mock_provider.generate.side_effect = Exception("LLM API error")
        
        with pytest.raises(RuntimeError, match="Failed to generate specifications"):
            generate_specifications(SIMPLE_PYTHON_CODE, "python", mock_provider)


class TestFormatLlmResponse:
    """Tests for formatting LLM response."""
    
    def test_format_response_with_metadata(self):
        """Test formatting response with metadata."""
        metadata = {
            "node_name": "TestNode",
            "function_count": 2,
            "class_count": 1,
            "dependencies": ["requests"],
            "code_length": 100,
        }
        
        result = format_llm_response(
            "Generated specifications", metadata, "python", include_metadata=True
        )
        
        assert result["status"] == "success"
        assert result["specifications"] == "Generated specifications"
        assert "metadata" in result
        assert result["metadata"]["language"] == "python"
        assert result["metadata"]["node_name"] == "TestNode"
        assert result["metadata"]["function_count"] == 2
        assert result["metadata"]["class_count"] == 1
        assert "extraction_timestamp" in result
    
    def test_format_response_without_metadata(self):
        """Test formatting response without metadata."""
        metadata = {
            "node_name": "TestNode",
            "function_count": 0,
            "class_count": 0,
            "dependencies": [],
            "code_length": 50,
        }
        
        result = format_llm_response(
            "Generated specifications", metadata, "javascript", include_metadata=False
        )
        
        assert result["status"] == "success"
        assert "metadata" not in result
        assert result["specifications"] == "Generated specifications"


class TestExtractCustomLogicHandler:
    """Tests for extract_custom_logic handler integration."""
    
    @pytest.mark.asyncio
    async def test_handler_success_python(self):
        """Test successful handler execution with Python code."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        mock_response = Mock()
        mock_response.content = "Generated Python specifications"
        mock_provider.generate.return_value = mock_response
        
        mock_server = Mock()
        mock_server._get_llm_provider.return_value = mock_provider
        
        set_server_instance(mock_server)
        
        result = await extract_custom_logic(
            SIMPLE_PYTHON_CODE, "python", "TestNode"
        )
        
        assert result["status"] == "success"
        assert "specifications" in result
        assert "metadata" in result
        assert result["metadata"]["language"] == "python"
    
    @pytest.mark.asyncio
    async def test_handler_success_javascript(self):
        """Test successful handler execution with JavaScript code."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        mock_response = Mock()
        mock_response.content = "Generated JavaScript specifications"
        mock_provider.generate.return_value = mock_response
        
        mock_server = Mock()
        mock_server._get_llm_provider.return_value = mock_provider
        
        set_server_instance(mock_server)
        
        result = await extract_custom_logic(
            SIMPLE_JAVASCRIPT_CODE, "javascript"
        )
        
        assert result["status"] == "success"
        assert result["metadata"]["language"] == "javascript"
    
    @pytest.mark.asyncio
    async def test_handler_invalid_language(self):
        """Test handler with invalid language."""
        result = await extract_custom_logic(SIMPLE_PYTHON_CODE, "java")
        
        assert result["status"] == "error"
        assert "Unsupported language" in result["error"]
    
    @pytest.mark.asyncio
    async def test_handler_invalid_code(self):
        """Test handler with invalid code."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        
        mock_server = Mock()
        mock_server._get_llm_provider.return_value = mock_provider
        
        set_server_instance(mock_server)
        
        result = await extract_custom_logic(INVALID_PYTHON_CODE, "python")
        
        assert result["status"] == "error"
        assert "Invalid code" in result["error"]
    
    @pytest.mark.asyncio
    async def test_handler_empty_code(self):
        """Test handler with empty code."""
        result = await extract_custom_logic("", "python")
        
        assert result["status"] == "error"
        assert "empty" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_handler_provider_unavailable(self):
        """Test handler when provider is unavailable."""
        mock_server = Mock()
        mock_server._get_llm_provider.side_effect = Exception("Provider error")
        
        set_server_instance(mock_server)
        
        # Mock create_from_env to also fail
        with patch("src.mcp_server.tools.handlers.create_from_env", side_effect=Exception("No provider")):
            result = await extract_custom_logic(SIMPLE_PYTHON_CODE, "python")
            
            assert result["status"] == "error"
            assert "LLM provider unavailable" in result["error"]
    
    @pytest.mark.asyncio
    async def test_handler_llm_generation_error(self):
        """Test handler when LLM generation fails."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        mock_provider.generate.side_effect = Exception("LLM API error")
        
        mock_server = Mock()
        mock_server._get_llm_provider.return_value = mock_provider
        
        set_server_instance(mock_server)
        
        result = await extract_custom_logic(SIMPLE_PYTHON_CODE, "python")
        
        assert result["status"] == "error"
        assert "Failed to generate specifications" in result["error"]
    
    @pytest.mark.asyncio
    async def test_handler_metadata_extraction_error(self):
        """Test handler when metadata extraction fails (should continue)."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        mock_response = Mock()
        mock_response.content = "Generated specifications"
        mock_provider.generate.return_value = mock_response
        
        mock_server = Mock()
        mock_server._get_llm_provider.return_value = mock_provider
        
        set_server_instance(mock_server)
        
        # Mock extract_dependencies to fail
        with patch("src.mcp_server.tools.handlers.extract_dependencies", side_effect=Exception("Parse error")):
            result = await extract_custom_logic(SIMPLE_PYTHON_CODE, "python")
            
            # Should still succeed, but with default metadata
            assert result["status"] == "success"
            assert result["metadata"]["function_count"] == 0


class TestGetLlmProvider:
    """Tests for provider access."""
    
    def test_get_provider_from_server(self):
        """Test getting provider from server instance."""
        mock_provider = Mock()
        mock_server = Mock()
        mock_server._get_llm_provider.return_value = mock_provider
        
        set_server_instance(mock_server)
        
        provider = _get_llm_provider()
        assert provider == mock_provider
    
    def test_get_provider_fallback(self):
        """Test fallback to environment provider."""
        mock_provider = Mock()
        
        set_server_instance(None)
        
        with patch("src.mcp_server.tools.handlers.create_from_env") as mock_create:
            mock_create.return_value = mock_provider
            provider = _get_llm_provider()
            assert provider == mock_provider
            mock_create.assert_called_once_with(auto_initialize=True)
    
    def test_get_provider_error(self):
        """Test error handling when provider cannot be obtained."""
        mock_server = Mock()
        mock_server._get_llm_provider.side_effect = Exception("Server error")
        
        set_server_instance(mock_server)
        
        with patch("src.mcp_server.tools.handlers.create_from_env", side_effect=Exception("No provider")):
            with pytest.raises(RuntimeError, match="Failed to get or create LLM provider"):
                _get_llm_provider()
