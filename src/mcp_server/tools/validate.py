"""LangGraph code validation module.

This module provides functions to validate LangGraph implementation code
for syntax errors, compliance with LangGraph patterns, and best practices.
"""

import ast
import importlib.util
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...llm_provider.base import BaseLLMProvider, GenerationParams

logger = logging.getLogger(__name__)

# Path to guides directory
GUIDES_DIR = Path(__file__).parent.parent.parent.parent / "guides"

# LangGraph-specific imports that should be present
LANGGRAPH_IMPORTS = {
    "langgraph",
    "langgraph.checkpoint",
    "langgraph.checkpoint.memory",
    "langgraph.entrypoint",
    "langgraph.task",
}

# Known valid packages for import validation
KNOWN_PACKAGES = {
    "langgraph",
    "langchain",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google_genai",
    "typing",
    "collections",
    "dataclasses",
    "pydantic",
    "json",
    "os",
    "sys",
    "logging",
    "datetime",
    "pathlib",
}


def validate_code_input(code: str) -> Tuple[bool, Optional[str]]:
    """Validate code input.
    
    Args:
        code: Code string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not code or not code.strip():
        return False, "Code cannot be empty"
    
    if len(code.strip()) < 10:
        return False, "Code must be at least 10 characters long"
    
    return True, None


def check_python_syntax(code: str) -> Tuple[bool, Optional[str], Optional[SyntaxError]]:
    """Check Python syntax using AST parsing.
    
    Args:
        code: Python code string
        
    Returns:
        Tuple of (is_valid, error_message, syntax_error)
    """
    try:
        ast.parse(code)
        return True, None, None
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        if e.text:
            error_msg += f" (text: {e.text.strip()})"
        return False, error_msg, e
    except Exception as e:
        return False, f"Failed to parse code: {e}", None


def validate_imports(code: str) -> List[Dict[str, Any]]:
    """Validate imports in code.
    
    Args:
        code: Python code string
        
    Returns:
        List of import issues with severity
    """
    issues = []
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If code has syntax errors, skip import validation
        return issues
    
    # Extract all imports
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    
    # Check for LangGraph imports
    has_langgraph_import = False
    for imp in imports:
        if imp.startswith("langgraph"):
            has_langgraph_import = True
            break
    
    if not has_langgraph_import:
        issues.append({
            "type": "warning",
            "category": "syntax",
            "severity": "warning",
            "message": "No LangGraph imports found. Code should import from langgraph.",
            "line": None,
            "column": None,
            "code_snippet": None,
            "suggestion": "Add imports like 'from langgraph import entrypoint, task' or 'from langgraph.graph import StateGraph'",
            "guide_reference": None,
        })
    
    # Check for common invalid imports
    for imp in imports:
        # Check if import is from known packages or standard library
        base_package = imp.split(".")[0]
        if base_package not in KNOWN_PACKAGES and not importlib.util.find_spec(base_package):
            # Try to import to verify
            try:
                importlib.import_module(base_package)
            except ImportError:
                issues.append({
                    "type": "warning",
                    "category": "syntax",
                    "severity": "warning",
                    "message": f"Import '{imp}' may not be available. Verify package is installed.",
                    "line": None,
                    "column": None,
                    "code_snippet": None,
                    "suggestion": f"Ensure package '{base_package}' is installed and available",
                    "guide_reference": None,
                })
    
    return issues


def check_basic_types(code: str) -> List[Dict[str, Any]]:
    """Check basic type issues in code.
    
    Args:
        code: Python code string
        
    Returns:
        List of type issues
    """
    issues = []
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return issues
    
    # Check function signatures for common issues
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check for missing type hints (optional, just a suggestion)
            if not node.returns and len(node.args.args) > 0:
                # This is just informational, not an error
                pass
    
    return issues


def detect_paradigm(code: str) -> str:
    """Detect LangGraph paradigm from code.
    
    Args:
        code: Python code string
        
    Returns:
        Detected paradigm: "functional", "graph", or "unknown"
    """
    code_lower = code.lower()
    
    # Check for Functional API patterns
    functional_patterns = [
        r"@entrypoint",
        r"from langgraph import entrypoint",
        r"from langgraph.entrypoint import entrypoint",
        r"@task",
        r"from langgraph import task",
        r"from langgraph.task import task",
    ]
    
    # Check for Graph API patterns
    graph_patterns = [
        r"StateGraph",
        r"from langgraph.graph import StateGraph",
        r"from langgraph import StateGraph",
        r"\.add_node\(",
        r"\.add_edge\(",
        r"\.add_conditional_edges\(",
    ]
    
    functional_matches = sum(1 for pattern in functional_patterns if re.search(pattern, code))
    graph_matches = sum(1 for pattern in graph_patterns if re.search(pattern, code))
    
    if graph_matches > 0 and graph_matches >= functional_matches:
        return "graph"
    elif functional_matches > 0:
        return "functional"
    else:
        return "unknown"


def check_langgraph_patterns(code: str, paradigm: str) -> List[Dict[str, Any]]:
    """Check LangGraph pattern compliance.
    
    Args:
        code: Python code string
        paradigm: Detected or expected paradigm ("functional" or "graph")
        
    Returns:
        List of compliance issues
    """
    issues = []
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return issues
    
    if paradigm == "functional":
        # Check for @entrypoint decorator
        has_entrypoint = False
        entrypoint_line = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "entrypoint":
                        has_entrypoint = True
                        entrypoint_line = node.lineno
                    elif isinstance(decorator, ast.Attribute) and decorator.attr == "entrypoint":
                        has_entrypoint = True
                        entrypoint_line = node.lineno
        
        if not has_entrypoint:
            issues.append({
                "type": "error",
                "category": "compliance",
                "severity": "error",
                "message": "Functional API requires @entrypoint decorator on main workflow function",
                "line": None,
                "column": None,
                "code_snippet": None,
                "suggestion": "Add @entrypoint decorator to your main workflow function",
                "guide_reference": "functional-api-implementation.md#core-pattern-overview",
            })
        
        # Check for sync/async consistency
        has_async_entrypoint = False
        has_async_task = False
        has_sync_entrypoint = False
        has_sync_task = False
        
        for node in ast.walk(tree):
            is_entrypoint = False
            is_task = False
            
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                is_entrypoint = any(
                    (isinstance(d, ast.Name) and d.id == "entrypoint") or
                    (isinstance(d, ast.Attribute) and d.attr == "entrypoint")
                    for d in node.decorator_list
                )
                is_task = any(
                    (isinstance(d, ast.Name) and d.id == "task") or
                    (isinstance(d, ast.Attribute) and d.attr == "task")
                    for d in node.decorator_list
                )
                
                if is_entrypoint:
                    if isinstance(node, ast.AsyncFunctionDef):
                        has_async_entrypoint = True
                    else:
                        has_sync_entrypoint = True
                        
                if is_task:
                    if isinstance(node, ast.AsyncFunctionDef):
                        has_async_task = True
                    else:
                        has_sync_task = True
        
        # Check for sync/async mismatch
        if has_sync_entrypoint and has_async_task:
            issues.append({
                "type": "error",
                "category": "compliance",
                "severity": "error",
                "message": "CRITICAL: Sync/Async mismatch - synchronous entrypoint cannot call async tasks",
                "line": None,
                "column": None,
                "code_snippet": None,
                "suggestion": "Either make entrypoint async (with async keyword) or make all tasks synchronous",
                "guide_reference": "functional-api-implementation.md#syncasync-execution-patterns",
            })
        elif has_async_entrypoint and has_sync_task and has_async_task:
            # Mixed async/sync tasks with async entrypoint - warn but not error
            issues.append({
                "type": "warning",
                "category": "compliance",
                "severity": "warning",
                "message": "Async entrypoint has both sync and async tasks. Consider making all tasks async for consistency",
                "line": None,
                "column": None,
                "code_snippet": None,
                "suggestion": "Make all tasks async if entrypoint is async",
                "guide_reference": "functional-api-implementation.md#syncasync-execution-patterns",
            })
    
    elif paradigm == "graph":
        # Check for StateGraph
        has_stategraph = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "StateGraph":
                    has_stategraph = True
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "StateGraph":
                    has_stategraph = True
        
        if not has_stategraph:
            issues.append({
                "type": "error",
                "category": "compliance",
                "severity": "error",
                "message": "Graph API requires StateGraph initialization",
                "line": None,
                "column": None,
                "code_snippet": None,
                "suggestion": "Create StateGraph instance: 'graph = StateGraph(State)'",
                "guide_reference": "graph-api-implementation.md#graph-construction",
            })
    
    return issues


def verify_decorators(code: str, paradigm: str) -> List[Dict[str, Any]]:
    """Verify decorator usage.
    
    Args:
        code: Python code string
        paradigm: Detected paradigm
        
    Returns:
        List of decorator issues
    """
    issues = []
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return issues
    
    if paradigm == "functional":
        # Check that entrypoint decorator is used correctly
        entrypoint_functions = []
        task_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "entrypoint":
                        entrypoint_functions.append((node.name, node.lineno))
                    elif isinstance(decorator, ast.Attribute) and decorator.attr == "entrypoint":
                        entrypoint_functions.append((node.name, node.lineno))
                    elif isinstance(decorator, ast.Name) and decorator.id == "task":
                        task_functions.append((node.name, node.lineno))
                    elif isinstance(decorator, ast.Attribute) and decorator.attr == "task":
                        task_functions.append((node.name, node.lineno))
        
        if len(entrypoint_functions) > 1:
            issues.append({
                "type": "warning",
                "category": "compliance",
                "severity": "warning",
                "message": f"Multiple @entrypoint functions found. Only one entrypoint should exist.",
                "line": None,
                "column": None,
                "code_snippet": None,
                "suggestion": "Ensure only one function has @entrypoint decorator",
                "guide_reference": "functional-api-implementation.md#core-pattern-overview",
            })
    
    return issues


def check_serialization(code: str) -> List[Dict[str, Any]]:
    """Check serialization patterns.
    
    Args:
        code: Python code string
        
    Returns:
        List of serialization issues
    """
    issues = []
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return issues
    
    # Check for non-serializable patterns in entrypoint parameters
    # This is a simplified check - look for common non-serializable types
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if this is an entrypoint function
            is_entrypoint = any(
                (isinstance(d, ast.Name) and d.id == "entrypoint") or
                (isinstance(d, ast.Attribute) and d.attr == "entrypoint")
                for d in node.decorator_list
            )
            
            if is_entrypoint:
                # Check parameters for suspicious patterns
                for arg in node.args.args:
                    # This is a basic check - in reality, we'd need more sophisticated analysis
                    pass
    
    return issues


def validate_guide_compliance(code: str, guides: Dict[str, str]) -> List[Dict[str, Any]]:
    """Validate code against guide requirements.
    
    Args:
        code: Python code string
        guides: Dictionary of guide contents
        
    Returns:
        List of compliance issues with guide references
    """
    issues = []
    
    paradigm = detect_paradigm(code)
    
    if paradigm == "functional" and "functional-api-implementation" in guides:
        guide_content = guides["functional-api-implementation"]
        
        # Check for sync/async consistency requirement
        if "CRITICAL" in guide_content and "sync/async" in guide_content.lower():
            # Check code for sync/async consistency using AST
            try:
                tree = ast.parse(code)
                has_async_entrypoint = False
                has_async_task = False
                has_sync_task = False
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                        is_entrypoint = any(
                            (isinstance(d, ast.Name) and d.id == "entrypoint") or
                            (isinstance(d, ast.Attribute) and d.attr == "entrypoint")
                            for d in node.decorator_list
                        )
                        is_task = any(
                            (isinstance(d, ast.Name) and d.id == "task") or
                            (isinstance(d, ast.Attribute) and d.attr == "task")
                            for d in node.decorator_list
                        )
                        
                        if is_entrypoint and isinstance(node, ast.AsyncFunctionDef):
                            has_async_entrypoint = True
                        if is_task:
                            if isinstance(node, ast.AsyncFunctionDef):
                                has_async_task = True
                            else:
                                has_sync_task = True
                
                if has_async_entrypoint and has_sync_task and not has_async_task:
                    # Async entrypoint with only sync tasks
                    issues.append({
                        "type": "warning",
                        "category": "compliance",
                        "severity": "warning",
                        "message": "Async entrypoint should have async tasks for consistency",
                        "line": None,
                        "column": None,
                        "code_snippet": None,
                        "suggestion": "Ensure all tasks are async if entrypoint is async",
                        "guide_reference": "functional-api-implementation.md#syncasync-execution-patterns",
                    })
            except SyntaxError:
                # Skip if code has syntax errors
                pass
    
    elif paradigm == "graph" and "graph-api-implementation" in guides:
        guide_content = guides["graph-api-implementation"]
        
        # Check for StateGraph requirement
        if "StateGraph" not in code and "StateGraph" in guide_content:
            issues.append({
                "type": "error",
                "category": "compliance",
                "severity": "error",
                "message": "Graph API requires StateGraph initialization",
                "line": None,
                "column": None,
                "code_snippet": None,
                "suggestion": "Create StateGraph instance as shown in graph-api-implementation guide",
                "guide_reference": "graph-api-implementation.md#graph-construction",
            })
    
    return issues


def check_code_quality(code: str) -> List[Dict[str, Any]]:
    """Check code quality and style.
    
    Args:
        code: Python code string
        
    Returns:
        List of quality issues
    """
    issues = []
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return issues
    
    # Check for docstrings
    has_docstring = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            if ast.get_docstring(node):
                has_docstring = True
                break
    
    if not has_docstring:
        issues.append({
            "type": "suggestion",
            "category": "quality",
            "severity": "info",
            "message": "Consider adding docstrings to functions and classes",
            "line": None,
            "column": None,
            "code_snippet": None,
            "suggestion": "Add docstrings following PEP 257 conventions",
            "guide_reference": None,
        })
    
    # Check for error handling
    has_try_except = "try:" in code or "except" in code
    
    if not has_try_except and len(code) > 500:
        issues.append({
            "type": "suggestion",
            "category": "quality",
            "severity": "info",
            "message": "Consider adding error handling for robustness",
            "line": None,
            "column": None,
            "code_snippet": None,
            "suggestion": "Add try/except blocks for error handling",
            "guide_reference": None,
        })
    
    # Check for logging
    has_logging = "logging" in code or "logger" in code.lower()
    
    if not has_logging and len(code) > 300:
        issues.append({
            "type": "suggestion",
            "category": "quality",
            "severity": "info",
            "message": "Consider adding logging for debugging and monitoring",
            "line": None,
            "column": None,
            "code_snippet": None,
            "suggestion": "Add logging statements for important operations",
            "guide_reference": None,
        })
    
    return issues


def generate_llm_suggestions(
    code: str,
    provider: BaseLLMProvider,
    validation_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Generate LLM-based improvement suggestions.
    
    Args:
        code: Python code string
        provider: LLM provider instance
        validation_results: Results from syntax and compliance checks
        
    Returns:
        List of LLM-generated suggestions
    """
    suggestions = []
    
    if not provider.is_initialized():
        try:
            provider.initialize()
        except Exception as e:
            logger.warning(f"Failed to initialize provider for suggestions: {e}")
            return suggestions
    
    try:
        # Build prompt for LLM code review
        prompt_parts = [
            "You are a code reviewer for LangGraph implementations. ",
            "Review the following code and provide actionable improvement suggestions.\n\n",
            "## Code to Review\n",
            "```python\n",
            code,
            "\n```\n\n",
        ]
        
        # Add validation context
        if validation_results.get("syntax", {}).get("errors"):
            prompt_parts.append("## Known Issues\n")
            for error in validation_results["syntax"]["errors"]:
                prompt_parts.append(f"- {error.get('message', 'Unknown error')}\n")
            prompt_parts.append("\n")
        
        if validation_results.get("compliance", {}).get("issues"):
            prompt_parts.append("## Compliance Issues\n")
            for issue in validation_results["compliance"]["issues"]:
                prompt_parts.append(f"- {issue.get('message', 'Unknown issue')}\n")
            prompt_parts.append("\n")
        
        prompt_parts.append(
            "## Task\n"
            "Provide 3-5 specific, actionable suggestions to improve this code. "
            "Focus on:\n"
            "- LangGraph best practices\n"
            "- Code quality and maintainability\n"
            "- Error handling and robustness\n"
            "- Performance optimizations\n"
            "\n"
            "Format each suggestion as:\n"
            "- Category: [category]\n"
            "- Priority: [high/medium/low]\n"
            "- Suggestion: [specific suggestion]\n"
            "- Reason: [why this helps]\n"
        )
        
        prompt = "".join(prompt_parts)
        
        logger.info("Generating LLM suggestions for code review...")
        params = GenerationParams(
            temperature=0.3,  # Lower temperature for focused suggestions
            max_tokens=1000,
        )
        
        response = provider.generate(prompt, params=params)
        llm_output = response.content
        
        # Parse LLM output into structured suggestions
        # Simple parsing - look for suggestion patterns
        lines = llm_output.split("\n")
        current_suggestion = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("- Category:"):
                if current_suggestion:
                    suggestions.append(current_suggestion)
                current_suggestion = {"category": line.replace("- Category:", "").strip()}
            elif line.startswith("- Priority:"):
                if current_suggestion:
                    current_suggestion["priority"] = line.replace("- Priority:", "").strip()
            elif line.startswith("- Suggestion:"):
                if current_suggestion:
                    current_suggestion["suggestion"] = line.replace("- Suggestion:", "").strip()
            elif line.startswith("- Reason:"):
                if current_suggestion:
                    current_suggestion["reason"] = line.replace("- Reason:", "").strip()
        
        if current_suggestion:
            suggestions.append(current_suggestion)
        
        # Format suggestions into standard format
        formatted_suggestions = []
        for sug in suggestions:
            formatted_suggestions.append({
                "type": "suggestion",
                "category": "best_practice",
                "severity": sug.get("priority", "medium").lower(),
                "message": sug.get("suggestion", "Improvement suggestion"),
                "line": None,
                "column": None,
                "code_snippet": None,
                "suggestion": sug.get("suggestion", ""),
                "guide_reference": None,
                "reason": sug.get("reason", ""),
            })
        
        logger.info(f"Generated {len(formatted_suggestions)} LLM suggestions")
        return formatted_suggestions
        
    except Exception as e:
        logger.warning(f"Failed to generate LLM suggestions: {e}")
        return suggestions


def format_validation_report(
    syntax_result: Dict[str, Any],
    compliance_issues: List[Dict[str, Any]],
    quality_issues: List[Dict[str, Any]],
    suggestions: List[Dict[str, Any]],
    code_length: int,
    paradigm: str,
) -> Dict[str, Any]:
    """Format validation report.
    
    Args:
        syntax_result: Syntax validation result
        compliance_issues: List of compliance issues
        quality_issues: List of quality issues
        suggestions: List of LLM suggestions
        code_length: Length of code
        paradigm: Detected paradigm
        
    Returns:
        Formatted validation report dictionary
    """
    # Count issues by type
    syntax_errors = syntax_result.get("errors", [])
    syntax_warnings = syntax_result.get("warnings", [])
    
    total_errors = len(syntax_errors) + len([
        i for i in compliance_issues + quality_issues
        if i.get("severity") == "error"
    ])
    
    total_warnings = len(syntax_warnings) + len([
        i for i in compliance_issues + quality_issues
        if i.get("severity") == "warning"
    ])
    
    total_suggestions = len(suggestions) + len([
        i for i in quality_issues
        if i.get("type") == "suggestion" or i.get("severity") == "info"
    ])
    
    # Determine if code is production ready
    is_production_ready = (
        total_errors == 0 and
        total_warnings == 0 and
        len(compliance_issues) == 0
    )
    
    # Calculate quality score (0.0-1.0)
    # Base score starts at 1.0, deduct points for issues
    quality_score = 1.0
    quality_score -= total_errors * 0.3  # Errors are severe
    quality_score -= total_warnings * 0.1  # Warnings are moderate
    quality_score = max(0.0, min(1.0, quality_score))  # Clamp to [0.0, 1.0]
    
    report = {
        "status": "success",
        "valid": total_errors == 0,
        "code_length": code_length,
        "paradigm": paradigm,
        "syntax": {
            "valid": syntax_result.get("valid", False),
            "errors": syntax_errors,
            "warnings": syntax_warnings,
        },
        "compliance": {
            "valid": len(compliance_issues) == 0,
            "issues": compliance_issues,
            "paradigm_detected": paradigm,
        },
        "best_practices": {
            "score": quality_score,
            "issues": quality_issues,
            "suggestions": suggestions,
        },
        "summary": {
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "total_suggestions": total_suggestions,
            "is_production_ready": is_production_ready,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    return report
