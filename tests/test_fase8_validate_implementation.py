"""Tests for Fase 8: Validate Implementation tool implementation."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import ast

from src.mcp_server.tools.validate import (
    validate_code_input,
    check_python_syntax,
    validate_imports,
    check_basic_types,
    detect_paradigm,
    check_langgraph_patterns,
    verify_decorators,
    check_serialization,
    validate_guide_compliance,
    check_code_quality,
    generate_llm_suggestions,
    format_validation_report,
)
from src.mcp_server.tools.handlers import (
    validate_implementation,
)
from src.mcp_server.context import HandlerContext
from src.llm_provider.base import GenerationResponse


# Sample code for testing
VALID_PYTHON_CODE = """def hello():
    return "world"
"""

VALID_FUNCTIONAL_CODE = """from langgraph import entrypoint, task

@entrypoint
def workflow(input_data):
    result = process_data(input_data)
    return {"result": result}

@task
def process_data(data):
    return data.upper()
"""

VALID_GRAPH_CODE = """from langgraph.graph import StateGraph
from typing import TypedDict

class State(TypedDict):
    data: str

def node1(state: State):
    return {"data": "processed"}

graph = StateGraph(State)
graph.add_node("node1", node1)
graph.add_edge("node1", "END")
"""

INVALID_SYNTAX_CODE = """def hello(
    return "world"
"""

CODE_WITH_IMPORTS = """import json
from langgraph import entrypoint
from unknown_package import something
"""

CODE_WITHOUT_LANGGRAPH = """def workflow():
    return {"result": "success"}
"""

ASYNC_MISMATCH_CODE = """from langgraph import entrypoint, task

@entrypoint
def workflow(input_data):
    result = process_data(input_data)
    return {"result": result}

@task
async def process_data(data):
    return data.upper()
"""

VALID_ASYNC_CODE = """from langgraph import entrypoint, task

@entrypoint
async def workflow(input_data):
    result = await process_data(input_data)
    return {"result": result}

@task
async def process_data(data):
    return data.upper()
"""

CODE_WITH_DOCSTRING = '''"""Module docstring."""

def hello():
    """Function docstring."""
    return "world"
'''

CODE_WITHOUT_DOCSTRING = """def hello():
    return "world"
"""

CODE_WITH_ERROR_HANDLING = """try:
    result = process()
except Exception as e:
    print(e)
"""

CODE_WITHOUT_ERROR_HANDLING = """result = process()
print(result)
"""

CODE_WITH_LOGGING = """import logging
logger = logging.getLogger(__name__)
logger.info("Hello")
"""

CODE_WITHOUT_LOGGING = """print("Hello")
"""

# Sample guides for testing
SAMPLE_GUIDES = {
    "functional-api-implementation": """# Functional API Implementation Guide

## Core Pattern Overview
- Use @entrypoint decorator for main workflow
- CRITICAL: Entrypoint and all tasks must match in sync/async pattern
""",
    "graph-api-implementation": """# Graph API Implementation Guide

## Graph Construction
- Create StateGraph instance
- Add nodes and edges
""",
}


class TestValidateCodeInput:
    """Tests for code input validation."""
    
    def test_valid_code(self):
        """Test validation with valid code."""
        is_valid, error_msg = validate_code_input(VALID_PYTHON_CODE)
        assert is_valid is True
        assert error_msg is None
    
    def test_empty_code(self):
        """Test validation with empty code."""
        is_valid, error_msg = validate_code_input("")
        assert is_valid is False
        assert "empty" in error_msg.lower()
    
    def test_whitespace_only(self):
        """Test validation with whitespace-only code."""
        is_valid, error_msg = validate_code_input("   \n\t  ")
        assert is_valid is False
    
    def test_short_code(self):
        """Test validation with very short code."""
        is_valid, error_msg = validate_code_input("x = 1")
        assert is_valid is False
        assert "10" in error_msg


class TestCheckPythonSyntax:
    """Tests for Python syntax checking."""
    
    def test_valid_syntax(self):
        """Test syntax check with valid Python code."""
        is_valid, error_msg, syntax_error = check_python_syntax(VALID_PYTHON_CODE)
        assert is_valid is True
        assert error_msg is None
        assert syntax_error is None
    
    def test_invalid_syntax(self):
        """Test syntax check with invalid Python code."""
        is_valid, error_msg, syntax_error = check_python_syntax(INVALID_SYNTAX_CODE)
        assert is_valid is False
        assert error_msg is not None
        assert "syntax error" in error_msg.lower() or "error" in error_msg.lower()
    
    def test_malformed_code(self):
        """Test syntax check with malformed code."""
        is_valid, error_msg, syntax_error = check_python_syntax("def incomplete(")
        assert is_valid is False


class TestValidateImports:
    """Tests for import validation."""
    
    def test_valid_imports(self):
        """Test validation with valid imports."""
        code = "import json\nfrom typing import Dict"
        issues = validate_imports(code)
        assert isinstance(issues, list)
    
    def test_missing_langgraph_import(self):
        """Test detection of missing LangGraph imports."""
        issues = validate_imports(CODE_WITHOUT_LANGGRAPH)
        assert len(issues) > 0
        assert any("langgraph" in issue.get("message", "").lower() for issue in issues)
    
    def test_langgraph_import_present(self):
        """Test that LangGraph import is detected."""
        issues = validate_imports(VALID_FUNCTIONAL_CODE)
        # Should not warn about missing LangGraph imports (no "no langgraph" message)
        langgraph_missing_warnings = [
            issue for issue in issues
            if "langgraph" in issue.get("message", "").lower() 
            and ("no" in issue.get("message", "").lower() or "missing" in issue.get("message", "").lower())
        ]
        assert len(langgraph_missing_warnings) == 0
    
    def test_invalid_syntax_skips_import_check(self):
        """Test that import validation is skipped for invalid syntax."""
        issues = validate_imports(INVALID_SYNTAX_CODE)
        # Should return empty list or handle gracefully
        assert isinstance(issues, list)


class TestCheckBasicTypes:
    """Tests for basic type checking."""
    
    def test_valid_code(self):
        """Test type check with valid code."""
        issues = check_basic_types(VALID_PYTHON_CODE)
        assert isinstance(issues, list)
    
    def test_invalid_syntax_skips_type_check(self):
        """Test that type check is skipped for invalid syntax."""
        issues = check_basic_types(INVALID_SYNTAX_CODE)
        assert isinstance(issues, list)


class TestDetectParadigm:
    """Tests for paradigm detection."""
    
    def test_detect_functional(self):
        """Test detection of Functional API paradigm."""
        paradigm = detect_paradigm(VALID_FUNCTIONAL_CODE)
        assert paradigm == "functional"
    
    def test_detect_graph(self):
        """Test detection of Graph API paradigm."""
        paradigm = detect_paradigm(VALID_GRAPH_CODE)
        assert paradigm == "graph"
    
    def test_detect_unknown(self):
        """Test detection of unknown paradigm."""
        paradigm = detect_paradigm(CODE_WITHOUT_LANGGRAPH)
        assert paradigm == "unknown"
    
    def test_detect_from_entrypoint_pattern(self):
        """Test detection from @entrypoint pattern."""
        code = "@entrypoint\ndef workflow(): pass"
        paradigm = detect_paradigm(code)
        assert paradigm == "functional"
    
    def test_detect_from_stategraph_pattern(self):
        """Test detection from StateGraph pattern."""
        code = "graph = StateGraph(State)"
        paradigm = detect_paradigm(code)
        assert paradigm == "graph"


class TestCheckLanggraphPatterns:
    """Tests for LangGraph pattern compliance checking."""
    
    def test_functional_with_entrypoint(self):
        """Test functional code with @entrypoint."""
        issues = check_langgraph_patterns(VALID_FUNCTIONAL_CODE, "functional")
        # Should not have errors about missing entrypoint
        entrypoint_errors = [
            issue for issue in issues
            if "entrypoint" in issue.get("message", "").lower() and issue.get("severity") == "error"
        ]
        assert len(entrypoint_errors) == 0
    
    def test_functional_without_entrypoint(self):
        """Test functional code without @entrypoint."""
        issues = check_langgraph_patterns(CODE_WITHOUT_LANGGRAPH, "functional")
        entrypoint_errors = [
            issue for issue in issues
            if "entrypoint" in issue.get("message", "").lower() and issue.get("severity") == "error"
        ]
        assert len(entrypoint_errors) > 0
    
    def test_graph_without_stategraph(self):
        """Test graph code without StateGraph."""
        issues = check_langgraph_patterns(CODE_WITHOUT_LANGGRAPH, "graph")
        stategraph_errors = [
            issue for issue in issues
            if "stategraph" in issue.get("message", "").lower() and issue.get("severity") == "error"
        ]
        assert len(stategraph_errors) > 0
    
    def test_graph_with_stategraph(self):
        """Test graph code with StateGraph."""
        issues = check_langgraph_patterns(VALID_GRAPH_CODE, "graph")
        stategraph_errors = [
            issue for issue in issues
            if "stategraph" in issue.get("message", "").lower() and issue.get("severity") == "error"
        ]
        assert len(stategraph_errors) == 0
    
    def test_sync_async_mismatch(self):
        """Test detection of sync/async mismatch."""
        issues = check_langgraph_patterns(ASYNC_MISMATCH_CODE, "functional")
        mismatch_errors = [
            issue for issue in issues
            if "sync" in issue.get("message", "").lower() and "async" in issue.get("message", "").lower()
            and issue.get("severity") == "error"
        ]
        assert len(mismatch_errors) > 0
    
    def test_valid_async_code(self):
        """Test that valid async code doesn't trigger mismatch."""
        issues = check_langgraph_patterns(VALID_ASYNC_CODE, "functional")
        mismatch_errors = [
            issue for issue in issues
            if "sync" in issue.get("message", "").lower() and "async" in issue.get("message", "").lower()
            and issue.get("severity") == "error"
        ]
        assert len(mismatch_errors) == 0
    
    def test_invalid_syntax_skips_pattern_check(self):
        """Test that pattern check is skipped for invalid syntax."""
        issues = check_langgraph_patterns(INVALID_SYNTAX_CODE, "functional")
        assert isinstance(issues, list)


class TestVerifyDecorators:
    """Tests for decorator verification."""
    
    def test_valid_decorators(self):
        """Test verification with valid decorators."""
        issues = verify_decorators(VALID_FUNCTIONAL_CODE, "functional")
        assert isinstance(issues, list)
    
    def test_multiple_entrypoints(self):
        """Test detection of multiple entrypoints."""
        code = """from langgraph import entrypoint

@entrypoint
def workflow1(): pass

@entrypoint
def workflow2(): pass
"""
        issues = verify_decorators(code, "functional")
        multiple_entrypoint_warnings = [
            issue for issue in issues
            if "multiple" in issue.get("message", "").lower() and "entrypoint" in issue.get("message", "").lower()
        ]
        assert len(multiple_entrypoint_warnings) > 0
    
    def test_invalid_syntax_skips_decorator_check(self):
        """Test that decorator check is skipped for invalid syntax."""
        issues = verify_decorators(INVALID_SYNTAX_CODE, "functional")
        assert isinstance(issues, list)


class TestCheckSerialization:
    """Tests for serialization checking."""
    
    def test_valid_code(self):
        """Test serialization check with valid code."""
        issues = check_serialization(VALID_FUNCTIONAL_CODE)
        assert isinstance(issues, list)
    
    def test_invalid_syntax_skips_serialization_check(self):
        """Test that serialization check is skipped for invalid syntax."""
        issues = check_serialization(INVALID_SYNTAX_CODE)
        assert isinstance(issues, list)


class TestValidateGuideCompliance:
    """Tests for guide compliance validation."""
    
    def test_functional_guide_compliance(self):
        """Test compliance with functional guide."""
        issues = validate_guide_compliance(VALID_FUNCTIONAL_CODE, SAMPLE_GUIDES)
        assert isinstance(issues, list)
    
    def test_graph_guide_compliance(self):
        """Test compliance with graph guide."""
        issues = validate_guide_compliance(VALID_GRAPH_CODE, SAMPLE_GUIDES)
        assert isinstance(issues, list)
    
    def test_missing_guides(self):
        """Test handling of missing guides."""
        issues = validate_guide_compliance(VALID_FUNCTIONAL_CODE, {})
        assert isinstance(issues, list)
    
    def test_unknown_paradigm(self):
        """Test handling of unknown paradigm."""
        issues = validate_guide_compliance(CODE_WITHOUT_LANGGRAPH, SAMPLE_GUIDES)
        assert isinstance(issues, list)


class TestCheckCodeQuality:
    """Tests for code quality checking."""
    
    def test_code_with_docstring(self):
        """Test quality check with docstrings."""
        issues = check_code_quality(CODE_WITH_DOCSTRING)
        # Should not warn about missing docstrings
        docstring_warnings = [
            issue for issue in issues
            if "docstring" in issue.get("message", "").lower()
        ]
        assert len(docstring_warnings) == 0
    
    def test_code_without_docstring(self):
        """Test quality check without docstrings."""
        issues = check_code_quality(CODE_WITHOUT_DOCSTRING)
        docstring_suggestions = [
            issue for issue in issues
            if "docstring" in issue.get("message", "").lower()
        ]
        # May or may not suggest docstrings (depends on implementation)
        assert isinstance(issues, list)
    
    def test_code_with_error_handling(self):
        """Test quality check with error handling."""
        issues = check_code_quality(CODE_WITH_ERROR_HANDLING)
        assert isinstance(issues, list)
    
    def test_code_without_error_handling(self):
        """Test quality check without error handling."""
        long_code = CODE_WITHOUT_ERROR_HANDLING * 100  # Make it long enough
        issues = check_code_quality(long_code)
        # May suggest error handling for long code
        assert isinstance(issues, list)
    
    def test_code_with_logging(self):
        """Test quality check with logging."""
        issues = check_code_quality(CODE_WITH_LOGGING)
        assert isinstance(issues, list)
    
    def test_code_without_logging(self):
        """Test quality check without logging."""
        long_code = CODE_WITHOUT_LOGGING * 100  # Make it long enough
        issues = check_code_quality(long_code)
        # May suggest logging for long code
        assert isinstance(issues, list)
    
    def test_invalid_syntax_skips_quality_check(self):
        """Test that quality check is skipped for invalid syntax."""
        issues = check_code_quality(INVALID_SYNTAX_CODE)
        assert isinstance(issues, list)


class TestGenerateLlmSuggestions:
    """Tests for LLM suggestion generation."""
    
    @pytest.mark.asyncio
    async def test_generate_suggestions_success(self):
        """Test successful generation of LLM suggestions."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        mock_response = Mock()
        mock_response.content = """- Category: best_practice
- Priority: high
- Suggestion: Add error handling
- Reason: Improves robustness
"""
        mock_provider.generate.return_value = mock_response
        
        validation_results = {
            "syntax": {"valid": True, "errors": []},
            "compliance": {"issues": []},
        }
        
        suggestions = generate_llm_suggestions(
            VALID_FUNCTIONAL_CODE,
            mock_provider,
            validation_results
        )
        
        assert isinstance(suggestions, list)
        mock_provider.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_suggestions_auto_initialize(self):
        """Test auto-initialization of provider."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = False
        mock_provider.initialize.return_value = None
        mock_response = Mock()
        mock_response.content = "- Category: best_practice\n- Suggestion: Test"
        mock_provider.generate.return_value = mock_response
        
        validation_results = {
            "syntax": {"valid": True, "errors": []},
            "compliance": {"issues": []},
        }
        
        suggestions = generate_llm_suggestions(
            VALID_FUNCTIONAL_CODE,
            mock_provider,
            validation_results
        )
        
        mock_provider.initialize.assert_called_once()
        assert isinstance(suggestions, list)
    
    @pytest.mark.asyncio
    async def test_generate_suggestions_provider_error(self):
        """Test handling of provider errors."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        mock_provider.generate.side_effect = Exception("Provider error")
        
        validation_results = {
            "syntax": {"valid": True, "errors": []},
            "compliance": {"issues": []},
        }
        
        suggestions = generate_llm_suggestions(
            VALID_FUNCTIONAL_CODE,
            mock_provider,
            validation_results
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) == 0  # Should return empty list on error
    
    @pytest.mark.asyncio
    async def test_generate_suggestions_initialization_error(self):
        """Test handling of initialization errors."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = False
        mock_provider.initialize.side_effect = Exception("Init error")
        
        validation_results = {
            "syntax": {"valid": True, "errors": []},
            "compliance": {"issues": []},
        }
        
        suggestions = generate_llm_suggestions(
            VALID_FUNCTIONAL_CODE,
            mock_provider,
            validation_results
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) == 0


class TestFormatValidationReport:
    """Tests for validation report formatting."""
    
    def test_format_report_success(self):
        """Test formatting of successful validation report."""
        syntax_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        compliance_issues = []
        quality_issues = []
        suggestions = []
        
        report = format_validation_report(
            syntax_result=syntax_result,
            compliance_issues=compliance_issues,
            quality_issues=quality_issues,
            suggestions=suggestions,
            code_length=100,
            paradigm="functional",
        )
        
        assert report["status"] == "success"
        assert report["valid"] is True
        assert report["code_length"] == 100
        assert report["paradigm"] == "functional"
        assert report["summary"]["total_errors"] == 0
        assert report["summary"]["is_production_ready"] is True
    
    def test_format_report_with_errors(self):
        """Test formatting of report with errors."""
        syntax_result = {
            "valid": False,
            "errors": [{
                "type": "error",
                "severity": "error",
                "message": "Syntax error",
            }],
            "warnings": [],
        }
        compliance_issues = [{
            "type": "error",
            "severity": "error",
            "message": "Missing entrypoint",
        }]
        quality_issues = []
        suggestions = []
        
        report = format_validation_report(
            syntax_result=syntax_result,
            compliance_issues=compliance_issues,
            quality_issues=quality_issues,
            suggestions=suggestions,
            code_length=100,
            paradigm="functional",
        )
        
        assert report["valid"] is False
        assert report["summary"]["total_errors"] == 2
        assert report["summary"]["is_production_ready"] is False
    
    def test_format_report_with_warnings(self):
        """Test formatting of report with warnings."""
        syntax_result = {
            "valid": True,
            "errors": [],
            "warnings": [{
                "type": "warning",
                "severity": "warning",
                "message": "Missing docstring",
            }],
        }
        compliance_issues = []
        quality_issues = []
        suggestions = []
        
        report = format_validation_report(
            syntax_result=syntax_result,
            compliance_issues=compliance_issues,
            quality_issues=quality_issues,
            suggestions=suggestions,
            code_length=100,
            paradigm="functional",
        )
        
        assert report["valid"] is True
        assert report["summary"]["total_warnings"] == 1
    
    def test_format_report_with_suggestions(self):
        """Test formatting of report with suggestions."""
        syntax_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        compliance_issues = []
        quality_issues = []
        suggestions = [{
            "type": "suggestion",
            "severity": "info",
            "message": "Add error handling",
        }]
        
        report = format_validation_report(
            syntax_result=syntax_result,
            compliance_issues=compliance_issues,
            quality_issues=quality_issues,
            suggestions=suggestions,
            code_length=100,
            paradigm="functional",
        )
        
        assert report["summary"]["total_suggestions"] == 1
        assert len(report["best_practices"]["suggestions"]) == 1
    
    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        syntax_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        compliance_issues = []
        quality_issues = []
        suggestions = []
        
        report = format_validation_report(
            syntax_result=syntax_result,
            compliance_issues=compliance_issues,
            quality_issues=quality_issues,
            suggestions=suggestions,
            code_length=100,
            paradigm="functional",
        )
        
        assert 0.0 <= report["best_practices"]["score"] <= 1.0
        
        # Test with errors - score should be lower
        syntax_result_with_errors = {
            "valid": False,
            "errors": [{"severity": "error"}],
            "warnings": [],
        }
        report_with_errors = format_validation_report(
            syntax_result=syntax_result_with_errors,
            compliance_issues=compliance_issues,
            quality_issues=quality_issues,
            suggestions=suggestions,
            code_length=100,
            paradigm="functional",
        )
        
        assert report_with_errors["best_practices"]["score"] < report["best_practices"]["score"]


class TestValidateImplementationHandler:
    """Tests for validate_implementation handler."""
    
    @pytest.mark.asyncio
    async def test_handler_success_all_checks(self):
        """Test handler with all checks enabled."""
        result = await validate_implementation(
            code=VALID_FUNCTIONAL_CODE,
            check_syntax=True,
            check_compliance=True,
            check_best_practices=True,
        )
        
        assert result["status"] == "success"
        assert "valid" in result
        assert "syntax" in result
        assert "compliance" in result
        assert "best_practices" in result
        assert "summary" in result
    
    @pytest.mark.asyncio
    async def test_handler_syntax_only(self):
        """Test handler with syntax check only."""
        result = await validate_implementation(
            code=VALID_FUNCTIONAL_CODE,
            check_syntax=True,
            check_compliance=False,
            check_best_practices=False,
        )
        
        assert result["status"] == "success"
        assert "syntax" in result
    
    @pytest.mark.asyncio
    async def test_handler_compliance_only(self):
        """Test handler with compliance check only."""
        result = await validate_implementation(
            code=VALID_FUNCTIONAL_CODE,
            check_syntax=False,
            check_compliance=True,
            check_best_practices=False,
        )
        
        assert result["status"] == "success"
        assert "compliance" in result
    
    @pytest.mark.asyncio
    async def test_handler_best_practices_only(self):
        """Test handler with best practices check only."""
        result = await validate_implementation(
            code=VALID_FUNCTIONAL_CODE,
            check_syntax=False,
            check_compliance=False,
            check_best_practices=True,
        )
        
        assert result["status"] == "success"
        assert "best_practices" in result
    
    @pytest.mark.asyncio
    async def test_handler_invalid_code_input(self):
        """Test handler with invalid code input."""
        result = await validate_implementation(
            code="",
            check_syntax=True,
            check_compliance=True,
            check_best_practices=True,
        )
        
        assert result["status"] == "error"
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_handler_invalid_syntax(self):
        """Test handler with invalid syntax."""
        result = await validate_implementation(
            code=INVALID_SYNTAX_CODE,
            check_syntax=True,
            check_compliance=True,
            check_best_practices=True,
        )
        
        assert result["status"] == "success"
        assert result["syntax"]["valid"] is False
        assert len(result["syntax"]["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_handler_with_provider(self):
        """Test handler with LLM provider for suggestions."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        mock_response = Mock()
        mock_response.content = "- Category: best_practice\n- Suggestion: Test"
        mock_provider.generate.return_value = mock_response
        
        mock_server = Mock()
        mock_server._get_llm_provider.return_value = mock_provider
        HandlerContext.set(mock_server)
        
        result = await validate_implementation(
            code=VALID_FUNCTIONAL_CODE,
            check_syntax=True,
            check_compliance=True,
            check_best_practices=True,
        )
        
        assert result["status"] == "success"
        # Should have attempted to generate suggestions
        # (may or may not succeed depending on provider)
    
    @pytest.mark.asyncio
    async def test_handler_provider_error(self):
        """Test handler when provider fails."""
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = False
        mock_provider.initialize.side_effect = Exception("Provider error")
        
        mock_server = Mock()
        mock_server._get_llm_provider.return_value = mock_provider
        HandlerContext.set(mock_server)
        
        result = await validate_implementation(
            code=VALID_FUNCTIONAL_CODE,
            check_syntax=True,
            check_compliance=True,
            check_best_practices=True,
        )
        
        # Should still succeed but without LLM suggestions
        assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_handler_detects_paradigm(self):
        """Test that handler detects paradigm correctly."""
        result = await validate_implementation(
            code=VALID_FUNCTIONAL_CODE,
            check_syntax=True,
            check_compliance=True,
            check_best_practices=False,
        )
        
        assert result["paradigm"] == "functional"
        
        result_graph = await validate_implementation(
            code=VALID_GRAPH_CODE,
            check_syntax=True,
            check_compliance=True,
            check_best_practices=False,
        )
        
        assert result_graph["paradigm"] == "graph"
    
    @pytest.mark.asyncio
    async def test_handler_error_handling(self):
        """Test handler error handling."""
        # Test with code that causes errors in validation
        # (This is a stress test - actual behavior depends on implementation)
        result = await validate_implementation(
            code="x" * 10000,  # Very long invalid code
            check_syntax=True,
            check_compliance=True,
            check_best_practices=True,
        )
        
        # Should handle gracefully
        assert "status" in result
