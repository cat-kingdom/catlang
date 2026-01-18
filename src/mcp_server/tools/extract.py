"""Custom logic extraction module.

This module provides functions to extract custom logic from n8n custom nodes
(Python or JavaScript code) and generate technical requirements specifications
using LLM providers.
"""

import ast
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...llm_provider.base import BaseLLMProvider, GenerationParams

logger = logging.getLogger(__name__)

# Path to Step2 prompt template
STEP2_PROMPT_PATH = Path(__file__).parent.parent.parent.parent / "OrchestrationPrompts" / "Step2-CustomLogic.md"

# Python standard library modules (not considered external dependencies)
PYTHON_STDLIB_MODULES = {
    "sys", "os", "json", "datetime", "time", "random", "math", "re", "collections",
    "itertools", "functools", "operator", "pathlib", "urllib", "http", "socket",
    "ssl", "email", "base64", "hashlib", "uuid", "logging", "typing", "dataclasses",
    "enum", "abc", "contextlib", "copy", "pickle", "io", "csv", "xml", "html",
    "textwrap", "string", "unicodedata", "locale", "calendar", "zoneinfo",
}


def load_step2_prompt_template() -> str:
    """Load Step2 prompt template from file.
    
    Returns:
        Prompt template as string
        
    Raises:
        FileNotFoundError: If prompt template file doesn't exist
        IOError: If file cannot be read
    """
    if not STEP2_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Step2 prompt template not found at {STEP2_PROMPT_PATH}"
        )
    
    try:
        with open(STEP2_PROMPT_PATH, "r", encoding="utf-8") as f:
            template = f.read()
        logger.debug(f"Loaded Step2 prompt template ({len(template)} chars)")
        return template
    except Exception as e:
        raise IOError(f"Failed to read prompt template: {e}") from e


def validate_language(language: str) -> bool:
    """Validate that language is supported.
    
    Args:
        language: Programming language string
        
    Returns:
        True if language is supported, False otherwise
    """
    supported_languages = {"python", "javascript"}
    is_valid = language.lower() in supported_languages
    if not is_valid:
        logger.warning(f"Unsupported language: {language}")
    return is_valid


def validate_code(code: str, language: str) -> Tuple[bool, Optional[str]]:
    """Validate code input.
    
    Args:
        code: Code string to validate
        language: Programming language
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not code or not code.strip():
        return False, "Code cannot be empty"
    
    if language.lower() == "python":
        # Basic Python syntax check using AST
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Invalid Python syntax: {e}"
        except Exception as e:
            return False, f"Failed to parse Python code: {e}"
    elif language.lower() == "javascript":
        # Basic JavaScript validation (check for balanced braces)
        open_braces = code.count("{")
        close_braces = code.count("}")
        open_parens = code.count("(")
        close_parens = code.count(")")
        
        if open_braces != close_braces:
            return False, "Unbalanced braces in JavaScript code"
        if open_parens != close_parens:
            return False, "Unbalanced parentheses in JavaScript code"
        
        return True, None
    else:
        return False, f"Unsupported language: {language}"
    
    return True, None


def parse_python_code(code: str) -> Dict[str, Any]:
    """Parse Python code using AST module.
    
    Args:
        code: Python code string
        
    Returns:
        Dictionary containing parsed code information:
        - imports: List of import statements
        - functions: List of function definitions
        - classes: List of class definitions
        - dependencies: List of third-party dependencies
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.warning(f"Failed to parse Python code: {e}")
        return {
            "imports": [],
            "functions": [],
            "classes": [],
            "dependencies": [],
            "parse_error": str(e),
        }
    
    imports = []
    functions = []
    classes = []
    dependencies = []
    
    for node in ast.walk(tree):
        # Extract imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                # Check if it's a third-party dependency
                module_name = alias.name.split(".")[0]
                if module_name not in PYTHON_STDLIB_MODULES:
                    if module_name not in dependencies:
                        dependencies.append(module_name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
                module_name = node.module.split(".")[0]
                if module_name not in PYTHON_STDLIB_MODULES:
                    if module_name not in dependencies:
                        dependencies.append(module_name)
        
        # Extract functions
        elif isinstance(node, ast.FunctionDef):
            func_info = {
                "name": node.name,
                "args": [arg.arg for arg in node.args.args],
                "lineno": node.lineno,
            }
            # Extract docstring
            docstring = ast.get_docstring(node)
            if docstring:
                func_info["docstring"] = docstring
            functions.append(func_info)
        
        # Extract classes
        elif isinstance(node, ast.ClassDef):
            class_info = {
                "name": node.name,
                "methods": [],
                "lineno": node.lineno,
            }
            # Extract methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_info = {
                        "name": item.name,
                        "args": [arg.arg for arg in item.args.args],
                    }
                    docstring = ast.get_docstring(item)
                    if docstring:
                        method_info["docstring"] = docstring
                    class_info["methods"].append(method_info)
            
            # Extract class docstring
            docstring = ast.get_docstring(node)
            if docstring:
                class_info["docstring"] = docstring
            
            classes.append(class_info)
    
    logger.debug(
        f"Parsed Python code: {len(imports)} imports, "
        f"{len(functions)} functions, {len(classes)} classes, "
        f"{len(dependencies)} dependencies"
    )
    
    return {
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "dependencies": dependencies,
    }


def parse_javascript_code(code: str) -> Dict[str, Any]:
    """Parse JavaScript code using regex patterns.
    
    Note: This is a simplified parser for MVP. Full AST parsing would require
    external libraries like esprima or acorn.
    
    Args:
        code: JavaScript code string
        
    Returns:
        Dictionary containing parsed code information:
        - imports: List of import/require statements
        - functions: List of function definitions
        - dependencies: List of dependencies
    """
    imports = []
    functions = []
    dependencies = []
    
    # Extract require() statements
    require_pattern = r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
    require_matches = re.findall(require_pattern, code)
    for match in require_matches:
        imports.append(match)
        # Extract package name (before /)
        package_name = match.split("/")[0]
        if package_name not in dependencies:
            dependencies.append(package_name)
    
    # Extract import statements (ES6 modules)
    import_pattern = r"import\s+(?:(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)\s+from\s+)?['\"]([^'\"]+)['\"]"
    import_matches = re.findall(import_pattern, code)
    for match in import_matches:
        imports.append(match)
        package_name = match.split("/")[0]
        if package_name not in dependencies:
            dependencies.append(package_name)
    
    # Extract function declarations: function name() {}
    func_decl_pattern = r"function\s+(\w+)\s*\([^)]*\)\s*\{"
    func_decl_matches = re.finditer(func_decl_pattern, code)
    for match in func_decl_matches:
        functions.append({
            "name": match.group(1),
            "type": "declaration",
            "position": match.start(),
        })
    
    # Extract arrow functions: const name = () => {}
    arrow_func_pattern = r"(?:const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>"
    arrow_func_matches = re.finditer(arrow_func_pattern, code)
    for match in arrow_func_matches:
        functions.append({
            "name": match.group(1),
            "type": "arrow",
            "position": match.start(),
        })
    
    logger.debug(
        f"Parsed JavaScript code: {len(imports)} imports, "
        f"{len(functions)} functions, {len(dependencies)} dependencies"
    )
    
    return {
        "imports": imports,
        "functions": functions,
        "dependencies": dependencies,
    }


def extract_dependencies(code: str, language: str) -> List[str]:
    """Extract dependencies from code.
    
    Args:
        code: Code string
        language: Programming language
        
    Returns:
        List of dependency names
    """
    if language.lower() == "python":
        parsed = parse_python_code(code)
        return parsed.get("dependencies", [])
    elif language.lower() == "javascript":
        parsed = parse_javascript_code(code)
        return parsed.get("dependencies", [])
    else:
        logger.warning(f"Unsupported language for dependency extraction: {language}")
        return []


def extract_functions(code: str, language: str) -> List[Dict[str, Any]]:
    """Extract function definitions from code.
    
    Args:
        code: Code string
        language: Programming language
        
    Returns:
        List of function information dictionaries
    """
    if language.lower() == "python":
        parsed = parse_python_code(code)
        return parsed.get("functions", [])
    elif language.lower() == "javascript":
        parsed = parse_javascript_code(code)
        return parsed.get("functions", [])
    else:
        logger.warning(f"Unsupported language for function extraction: {language}")
        return []


def extract_classes(code: str, language: str) -> List[Dict[str, Any]]:
    """Extract class definitions from code.
    
    Note: Only supported for Python. JavaScript classes would require
    more sophisticated parsing.
    
    Args:
        code: Code string
        language: Programming language
        
    Returns:
        List of class information dictionaries (empty for JavaScript)
    """
    if language.lower() == "python":
        parsed = parse_python_code(code)
        return parsed.get("classes", [])
    else:
        # JavaScript classes not extracted in MVP
        return []


def build_extraction_prompt(
    code: str,
    language: str,
    node_name: Optional[str] = None,
) -> str:
    """Build LLM prompt for custom logic extraction.
    
    Args:
        code: Custom node code
        language: Programming language
        node_name: Optional node name for context
        
    Returns:
        Complete prompt string
    """
    try:
        template = load_step2_prompt_template()
        
        # Replace placeholder with actual code
        prompt = template.replace("[PASTE CUSTOM NODE CODE HERE]", code)
        
        # Add language context if provided
        if node_name:
            # Add node name context at the beginning
            context = f"Node Name: {node_name}\nLanguage: {language.capitalize()}\n\n"
            prompt = context + prompt
        
        logger.debug(f"Built extraction prompt ({len(prompt)} chars)")
        return prompt
    except Exception as e:
        logger.error(f"Failed to build extraction prompt: {e}")
        raise


def generate_specifications(
    code: str,
    language: str,
    provider: BaseLLMProvider,
    node_name: Optional[str] = None,
) -> str:
    """Generate custom logic specifications using LLM provider.
    
    Args:
        code: Custom node code
        language: Programming language
        provider: LLM provider instance
        node_name: Optional node name for context
        
    Returns:
        Generated specifications text
        
    Raises:
        RuntimeError: If LLM generation fails
    """
    if not provider.is_initialized():
        provider.initialize()
    
    try:
        # Build prompt
        prompt = build_extraction_prompt(code, language, node_name)
        
        # Generate specifications using LLM
        logger.info(f"Generating specifications for {language} code...")
        params = GenerationParams(
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=4000,  # Allow for detailed specifications
        )
        
        response = provider.generate(prompt, params=params)
        specifications = response.content
        
        logger.info(f"Generated specifications ({len(specifications)} chars)")
        return specifications
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise RuntimeError(f"Failed to generate specifications: {e}") from e


def format_llm_response(
    llm_output: str,
    metadata: Dict[str, Any],
    language: str,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """Format LLM output into structured response.
    
    Args:
        llm_output: Raw LLM output text
        metadata: Code metadata dictionary
        language: Programming language
        include_metadata: Whether to include metadata in response
        
    Returns:
        Structured response dictionary
    """
    response = {
        "status": "success",
        "specifications": llm_output,  # LLM output is already formatted text
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    if include_metadata:
        response["metadata"] = {
            "language": language,
            "node_name": metadata.get("node_name"),
            "function_count": metadata.get("function_count", 0),
            "class_count": metadata.get("class_count", 0),
            "dependencies": metadata.get("dependencies", []),
            "code_length": metadata.get("code_length", 0),
        }
    
    return response
