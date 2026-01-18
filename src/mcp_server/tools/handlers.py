"""Tool handlers for MCP Server.

This module contains the actual implementation of tool handlers.
These are placeholder implementations that will be fully implemented
in later phases (Fase 5-8).
"""

import logging
from typing import Any

from ..context import get_llm_provider, get_resource_manager
from ..decorators import handle_errors
from ..constants import ResponseStatus, ErrorCode, ErrorMessage
from .analyze import (
    parse_n8n_workflow,
    validate_n8n_schema,
    extract_workflow_metadata,
    generate_requirements,
    format_llm_response,
)
from .extract import (
    validate_language,
    validate_code,
    extract_dependencies,
    extract_functions,
    extract_classes,
    generate_specifications,
    format_llm_response as format_extraction_response,
)
from .generate import (
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
from .validate import (
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

logger = logging.getLogger(__name__)


@handle_errors
async def analyze_n8n_workflow(
    workflow_json: str,
    include_metadata: bool = True,
) -> dict[str, Any]:
    """Analyze n8n workflow and generate production requirements.
    
    Args:
        workflow_json: n8n workflow JSON as string
        include_metadata: Whether to include metadata
        
    Returns:
        Dictionary containing production requirements with status, requirements text,
        metadata (if requested), and analysis timestamp
        
    Raises:
        ValueError: If workflow JSON is invalid
        RuntimeError: If LLM generation fails
    """
    logger.info(f"analyze_n8n_workflow called (include_metadata={include_metadata})")
    logger.debug(f"Workflow JSON length: {len(workflow_json)} characters")
    
    # Parse and validate workflow JSON
    workflow_data = parse_n8n_workflow(workflow_json)
    is_valid, error_msg = validate_n8n_schema(workflow_data)
    
    if not is_valid:
        return {
            "status": ResponseStatus.ERROR.value,
            "error": f"{ErrorMessage.INVALID_WORKFLOW_SCHEMA}: {error_msg}",
            "error_code": ErrorCode.INVALID_WORKFLOW_JSON.value,
            "workflow_length": len(workflow_json),
        }
    
    # Extract metadata
    metadata = extract_workflow_metadata(workflow_data)
    
    # Get LLM provider from context
    provider = get_llm_provider()
    
    # Generate requirements using LLM
    requirements_text = generate_requirements(workflow_json, provider)
    
    # Format response
    response = format_llm_response(requirements_text, metadata, include_metadata)
    
    logger.info("Successfully analyzed n8n workflow")
    return response


@handle_errors
async def extract_custom_logic(
    code: str,
    language: str,
    node_name: str | None = None,
) -> dict[str, Any]:
    """Extract custom logic from custom node code.
    
    Args:
        code: Custom node code (Python or JavaScript)
        language: Programming language ("python" or "javascript")
        node_name: Optional node name for context
        
    Returns:
        Dictionary containing custom logic specifications with status,
        specifications text, metadata (if requested), and extraction timestamp
        
    Raises:
        ValueError: If code or language is invalid
        RuntimeError: If LLM generation fails
    """
    logger.info(
        f"extract_custom_logic called "
        f"(language={language}, node_name={node_name})"
    )
    logger.debug(f"Code length: {len(code)} characters")
    
    # Validate language
    if not validate_language(language):
        return {
            "status": ResponseStatus.ERROR.value,
            "error": f"{ErrorMessage.INVALID_LANGUAGE}: {language}",
            "error_code": ErrorCode.INVALID_LANGUAGE.value,
            "code_length": len(code),
        }
    
    # Validate code
    is_valid, error_msg = validate_code(code, language)
    if not is_valid:
        return {
            "status": ResponseStatus.ERROR.value,
            "error": f"{ErrorMessage.INVALID_CODE}: {error_msg}",
            "error_code": ErrorCode.INVALID_CODE.value,
            "language": language,
            "code_length": len(code),
        }
    
    # Extract metadata (optional, for response)
    try:
        dependencies = extract_dependencies(code, language)
        functions = extract_functions(code, language)
        classes = extract_classes(code, language)
        
        metadata = {
            "node_name": node_name,
            "function_count": len(functions),
            "class_count": len(classes),
            "dependencies": dependencies,
            "code_length": len(code),
        }
    except Exception as e:
        logger.warning(f"Failed to extract metadata: {e}. Continuing without metadata.")
        metadata = {
            "node_name": node_name,
            "function_count": 0,
            "class_count": 0,
            "dependencies": [],
            "code_length": len(code),
        }
    
    # Get LLM provider from context
    provider = get_llm_provider()
    
    # Generate specifications using LLM
    specifications_text = generate_specifications(
        code, language, provider, node_name
    )
    
    # Format response
    response = format_extraction_response(
        specifications_text, metadata, language, include_metadata=True
    )
    
    logger.info("Successfully extracted custom logic")
    return response


@handle_errors
async def generate_langgraph_implementation(
    requirements: str,
    custom_logic_specs: str | None = None,
    paradigm: str = "auto",
    output_format: str = "code",
) -> dict[str, Any]:
    """Generate LangGraph implementation from requirements.
    
    Args:
        requirements: Production requirements from analyze_n8n_workflow
        custom_logic_specs: Custom logic specifications (optional)
        paradigm: LangGraph paradigm ("functional", "graph", or "auto")
        output_format: Output format ("code" or "file")
        
    Returns:
        Dictionary containing generated code or file path
        
    Raises:
        ValueError: If requirements or parameters are invalid
        RuntimeError: If LLM generation fails
    """
    logger.info(
        f"generate_langgraph_implementation called "
        f"(paradigm={paradigm}, output_format={output_format})"
    )
    logger.debug(f"Requirements length: {len(requirements)} characters")
    
    # Validate inputs
    is_valid, error_msg = validate_requirements(requirements)
    if not is_valid:
        return {
            "status": ResponseStatus.ERROR.value,
            "error": f"{ErrorMessage.INVALID_REQUIREMENTS}: {error_msg}",
            "error_code": ErrorCode.INVALID_REQUIREMENTS.value,
            "requirements_length": len(requirements),
            "paradigm": paradigm,
            "output_format": output_format,
        }
    
    if not validate_paradigm(paradigm):
        return {
            "status": ResponseStatus.ERROR.value,
            "error": f"{ErrorMessage.INVALID_PARADIGM}: {paradigm}",
            "error_code": ErrorCode.INVALID_PARADIGM.value,
            "requirements_length": len(requirements),
            "paradigm": paradigm,
            "output_format": output_format,
        }
    
    if not validate_output_format(output_format):
        return {
            "status": ResponseStatus.ERROR.value,
            "error": f"{ErrorMessage.INVALID_OUTPUT_FORMAT}: {output_format}",
            "error_code": ErrorCode.INVALID_OUTPUT_FORMAT.value,
            "requirements_length": len(requirements),
            "paradigm": paradigm,
            "output_format": output_format,
        }
    
    # Load Step3 prompt template
    step3_template = load_step3_prompt_template()
    
    # Load all guides
    try:
        guides = load_all_guides()
        guides_context = build_guides_context(guides)
    except Exception as e:
        logger.warning(f"Failed to load guides: {e}. Continuing without guides.")
        guides = {}
        guides_context = ""
    
    # Determine paradigm
    selected_paradigm = determine_paradigm(requirements, paradigm)
    
    # Get paradigm-specific guide
    try:
        paradigm_guide = get_paradigm_guide(selected_paradigm, guides)
    except Exception as e:
        logger.warning(f"Failed to get paradigm guide: {e}. Continuing without paradigm guide.")
        paradigm_guide = ""
    
    # Build complete generation prompt
    prompt = build_generation_prompt(
        requirements=requirements,
        custom_logic_specs=custom_logic_specs,
        guides_context=guides_context,
        paradigm=selected_paradigm,
        paradigm_guide=paradigm_guide,
    )
    
    # Get LLM provider from context
    provider = get_llm_provider()
    
    # Generate code using LLM
    llm_output = generate_code(prompt, provider)
    
    # Extract code from LLM response
    code = extract_code_from_response(llm_output)
    
    # Handle output format
    file_path = None
    if output_format == "file":
        file_path = save_code_to_file(code)
    
    # Format and return response
    response = format_response(
        code=code,
        paradigm=selected_paradigm,
        output_format=output_format,
        file_path=file_path,
    )
    
    logger.info("Successfully generated LangGraph implementation")
    return response


@handle_errors
async def validate_implementation(
    code: str,
    check_syntax: bool = True,
    check_compliance: bool = True,
    check_best_practices: bool = True,
) -> dict[str, Any]:
    """Validate LangGraph implementation code.
    
    Args:
        code: LangGraph implementation code to validate
        check_syntax: Check Python syntax errors
        check_compliance: Check LangGraph pattern compliance
        check_best_practices: Check best practices and generate suggestions
        
    Returns:
        Dictionary containing validation results with status, syntax results,
        compliance issues, quality issues, and suggestions
    """
    logger.info(
        f"validate_implementation called "
        f"(syntax={check_syntax}, compliance={check_compliance}, "
        f"best_practices={check_best_practices})"
    )
    logger.debug(f"Code length: {len(code)} characters")
    
    # Validate code input
    is_valid, error_msg = validate_code_input(code)
    if not is_valid:
        return {
            "status": ResponseStatus.ERROR.value,
            "error": f"{ErrorMessage.INVALID_CODE}: {error_msg}",
            "error_code": ErrorCode.INVALID_CODE.value,
            "code_length": len(code),
        }
    
    # Initialize results
    syntax_result = {"valid": True, "errors": [], "warnings": []}
    compliance_issues = []
    quality_issues = []
    suggestions = []
    detected_paradigm = "unknown"
    
    # 1. Syntax Validation
    if check_syntax:
        try:
            is_syntax_valid, syntax_error_msg, syntax_error = check_python_syntax(code)
            syntax_result["valid"] = is_syntax_valid
            
            if not is_syntax_valid:
                syntax_result["errors"].append({
                    "type": "error",
                    "category": "syntax",
                    "severity": "error",
                    "message": syntax_error_msg or "Syntax error",
                    "line": syntax_error.lineno if syntax_error else None,
                    "column": syntax_error.offset if syntax_error else None,
                    "code_snippet": syntax_error.text.strip() if syntax_error and syntax_error.text else None,
                    "suggestion": "Fix syntax errors before proceeding",
                    "guide_reference": None,
                })
            else:
                # Check imports and basic types only if syntax is valid
                import_issues = validate_imports(code)
                syntax_result["warnings"].extend(import_issues)
                
                type_issues = check_basic_types(code)
                if type_issues:
                    syntax_result["warnings"].extend(type_issues)
                    
        except Exception as e:
            logger.error(f"Syntax validation failed: {e}")
            syntax_result["valid"] = False
            syntax_result["errors"].append({
                "type": "error",
                "category": "syntax",
                "severity": "error",
                "message": f"Syntax validation error: {e}",
                "line": None,
                "column": None,
                "code_snippet": None,
                "suggestion": "Check code for syntax errors",
                "guide_reference": None,
            })
    
    # 2. Compliance Checking (only if syntax is valid or check_compliance is True)
    if check_compliance:
            try:
                # Detect paradigm
                detected_paradigm = detect_paradigm(code)
                
                # Check LangGraph patterns
                pattern_issues = check_langgraph_patterns(code, detected_paradigm)
                compliance_issues.extend(pattern_issues)
                
                # Verify decorators
                decorator_issues = verify_decorators(code, detected_paradigm)
                compliance_issues.extend(decorator_issues)
                
                # Check serialization
                serialization_issues = check_serialization(code)
                compliance_issues.extend(serialization_issues)
                
                # Validate guide compliance
                try:
                    guides = load_all_guides()
                    guide_compliance_issues = validate_guide_compliance(code, guides)
                    compliance_issues.extend(guide_compliance_issues)
                except Exception as e:
                    logger.warning(f"Failed to load guides for compliance checking: {e}")
                    
            except Exception as e:
                logger.error(f"Compliance checking failed: {e}")
                compliance_issues.append({
                    "type": "error",
                    "category": "compliance",
                    "severity": "error",
                    "message": f"Compliance checking error: {e}",
                    "line": None,
                    "column": None,
                    "code_snippet": None,
                    "suggestion": "Review compliance checking error",
                    "guide_reference": None,
                })
    
    # 3. Best Practices Checking
    if check_best_practices:
            try:
                # Check code quality
                quality_issues_list = check_code_quality(code)
                quality_issues.extend(quality_issues_list)
                
                # Generate LLM suggestions
                try:
                    provider = get_llm_provider()
                    validation_results = {
                        "syntax": syntax_result,
                        "compliance": {
                            "issues": compliance_issues,
                            "paradigm_detected": detected_paradigm,
                        },
                    }
                    llm_suggestions = generate_llm_suggestions(code, provider, validation_results)
                    suggestions.extend(llm_suggestions)
                except Exception as e:
                    logger.warning(f"Failed to generate LLM suggestions: {e}")
                    # Continue without LLM suggestions
                    
            except Exception as e:
                logger.error(f"Best practices checking failed: {e}")
                quality_issues.append({
                    "type": "error",
                    "category": "quality",
                    "severity": "error",
                    "message": f"Best practices checking error: {e}",
                    "line": None,
                    "column": None,
                    "code_snippet": None,
                    "suggestion": "Review best practices checking error",
                    "guide_reference": None,
                })
    
    # Format validation report
    report = format_validation_report(
        syntax_result=syntax_result,
        compliance_issues=compliance_issues,
        quality_issues=quality_issues,
        suggestions=suggestions,
        code_length=len(code),
        paradigm=detected_paradigm,
    )
    
    logger.info(
        f"Validation complete: "
        f"valid={report['valid']}, "
        f"errors={report['summary']['total_errors']}, "
        f"warnings={report['summary']['total_warnings']}, "
        f"suggestions={report['summary']['total_suggestions']}"
    )
    
    return report


@handle_errors
async def list_guides(
    category: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """List all available implementation guides.
    
    Args:
        category: Filter by category (optional)
        tags: Filter by tags (optional)
        
    Returns:
        Dictionary containing list of guides with metadata
    """
    logger.info(
        f"list_guides called "
        f"(category={category}, tags={tags})"
    )
    
    # Get resource manager from context
    resource_manager = get_resource_manager()
    
    # Ensure resource manager is initialized
    if not resource_manager._initialized:
        resource_manager.initialize()
    
    # List guides with filters
    guides = resource_manager.indexer.list_guides(category=category, tags=tags)
    
    # Convert to response format
    guides_list = [guide.to_dict() for guide in guides]
    
    logger.info(f"Listed {len(guides_list)} guides")
    return {
        "status": ResponseStatus.SUCCESS.value,
        "guides": guides_list,
        "count": len(guides_list),
        "filters": {
            "category": category,
            "tags": tags,
        },
    }


@handle_errors
async def query_guide(
    guide_name: str,
    category: str | None = None,
) -> dict[str, Any]:
    """Query a specific implementation guide.
    
    Args:
        guide_name: Name of the guide to query
        category: Category of the guide (optional)
        
    Returns:
        Dictionary containing guide content and metadata
    """
    logger.info(
        f"query_guide called "
        f"(guide_name={guide_name}, category={category})"
    )
    
    # Validate input
    if not guide_name or not guide_name.strip():
        return {
            "status": ResponseStatus.ERROR.value,
            "error": ErrorMessage.GUIDE_NAME_EMPTY,
            "error_code": ErrorCode.INVALID_INPUT.value,
            "guide_name": guide_name,
        }
    
    # Get resource manager from context
    resource_manager = get_resource_manager()
    
    # Ensure resource manager is initialized
    if not resource_manager._initialized:
        resource_manager.initialize()
    
    # Get guide metadata
    guide = resource_manager.indexer.get_guide(guide_name, category=category)
    
    if guide is None:
        logger.warning(f"Guide not found: {guide_name} (category={category})")
        return {
            "status": ResponseStatus.ERROR.value,
            "error": f"{ErrorMessage.GUIDE_NOT_FOUND}: '{guide_name}'",
            "error_code": ErrorCode.GUIDE_NOT_FOUND.value,
            "guide_name": guide_name,
            "category": category,
        }
    
    # Get guide content
    content = resource_manager.get_guide_content(guide_name, category=category)
    
    if content is None:
        logger.warning(f"Guide content not found: {guide_name}")
        return {
            "status": ResponseStatus.ERROR.value,
            "error": f"{ErrorMessage.GUIDE_NOT_FOUND}: '{guide_name}' (content)",
            "error_code": ErrorCode.GUIDE_NOT_FOUND.value,
            "guide_name": guide_name,
        }
    
    logger.info(f"Retrieved guide: {guide_name}")
    return {
        "status": ResponseStatus.SUCCESS.value,
        "guide_name": guide_name,
        "category": guide.category,
        "title": guide.title,
        "description": guide.description,
        "tags": guide.tags,
        "uri": guide.uri,
        "content": content,
        "last_modified": guide.last_modified.isoformat() if guide.last_modified else None,
    }
