"""LangGraph code generation module.

This module provides functions to generate LangGraph implementation code
from production requirements and custom logic specifications using LLM providers.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ...llm_provider.base import BaseLLMProvider, GenerationParams

logger = logging.getLogger(__name__)

# Path to Step3 prompt template
STEP3_PROMPT_PATH = Path(__file__).parent.parent.parent.parent / "OrchestrationPrompts" / "Step3-MainOrchestorPrompt.md"

# Path to guides directory
GUIDES_DIR = Path(__file__).parent.parent.parent.parent / "guides"

# Expected guide files
EXPECTED_GUIDES = [
    "paradigm-selection.md",
    "functional-api-implementation.md",
    "graph-api-implementation.md",
    "authentication-setup.md",
    "api-integration.md",
    "project-structure.md",
    "testing-and-troubleshooting.md",
    "output-requirements.md",
]


def load_step3_prompt_template() -> str:
    """Load Step3 prompt template from file.
    
    Returns:
        Prompt template as string
        
    Raises:
        FileNotFoundError: If prompt template file doesn't exist
        IOError: If file cannot be read
    """
    if not STEP3_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Step3 prompt template not found at {STEP3_PROMPT_PATH}"
        )
    
    try:
        with open(STEP3_PROMPT_PATH, "r", encoding="utf-8") as f:
            template = f.read()
        logger.debug(f"Loaded Step3 prompt template ({len(template)} chars)")
        return template
    except Exception as e:
        raise IOError(f"Failed to read prompt template: {e}") from e


def load_all_guides() -> Dict[str, str]:
    """Load all guide files from guides directory.
    
    Returns:
        Dictionary mapping guide filename (without .md) to content
        
    Raises:
        IOError: If guides directory cannot be accessed
    """
    guides = {}
    
    if not GUIDES_DIR.exists():
        logger.warning(f"Guides directory not found at {GUIDES_DIR}")
        return guides
    
    if not GUIDES_DIR.is_dir():
        logger.warning(f"Guides path is not a directory: {GUIDES_DIR}")
        return guides
    
    # Load all .md files in guides directory
    for guide_file in GUIDES_DIR.glob("*.md"):
        guide_name = guide_file.stem  # filename without .md extension
        try:
            with open(guide_file, "r", encoding="utf-8") as f:
                content = f.read()
            guides[guide_name] = content
            logger.debug(f"Loaded guide: {guide_name} ({len(content)} chars)")
        except Exception as e:
            logger.warning(f"Failed to load guide {guide_name}: {e}")
    
    # Warn about missing expected guides
    missing_guides = []
    for expected_guide in EXPECTED_GUIDES:
        guide_name = Path(expected_guide).stem
        if guide_name not in guides:
            missing_guides.append(expected_guide)
    
    if missing_guides:
        logger.warning(f"Missing expected guides: {', '.join(missing_guides)}")
    
    logger.info(f"Loaded {len(guides)} guides")
    return guides


def build_guides_context(guides: Dict[str, str]) -> str:
    """Format all guides into a single context string.
    
    Args:
        guides: Dictionary mapping guide name to content
        
    Returns:
        Formatted guides context string
    """
    if not guides:
        return ""
    
    context_parts = []
    for guide_name, content in sorted(guides.items()):
        context_parts.append(f"## Guide: {guide_name}\n{content}\n\n")
    
    context = "\n".join(context_parts)
    logger.debug(f"Built guides context ({len(context)} chars)")
    return context


def validate_requirements(requirements: str) -> Tuple[bool, Optional[str]]:
    """Validate requirements input.
    
    Args:
        requirements: Requirements string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not requirements or not requirements.strip():
        return False, "Requirements cannot be empty"
    
    if len(requirements.strip()) < 100:
        return False, "Requirements must be at least 100 characters long"
    
    return True, None


def validate_paradigm(paradigm: str) -> bool:
    """Validate paradigm value.
    
    Args:
        paradigm: Paradigm string
        
    Returns:
        True if paradigm is valid, False otherwise
    """
    valid_paradigms = {"functional", "graph", "auto"}
    is_valid = paradigm.lower() in valid_paradigms
    if not is_valid:
        logger.warning(f"Invalid paradigm: {paradigm}")
    return is_valid


def validate_output_format(output_format: str) -> bool:
    """Validate output format value.
    
    Args:
        output_format: Output format string
        
    Returns:
        True if output format is valid, False otherwise
    """
    valid_formats = {"code", "file"}
    is_valid = output_format.lower() in valid_formats
    if not is_valid:
        logger.warning(f"Invalid output format: {output_format}")
    return is_valid


def determine_paradigm(requirements: str, paradigm: str) -> str:
    """Determine the appropriate paradigm for code generation.
    
    If paradigm is "auto", analyzes requirements to determine best paradigm.
    Otherwise, returns the provided paradigm.
    
    Args:
        requirements: Production requirements text
        paradigm: Requested paradigm ("functional", "graph", or "auto")
        
    Returns:
        Determined paradigm ("functional" or "graph")
    """
    if paradigm.lower() != "auto":
        return paradigm.lower()
    
    # Auto-detection logic (simplified for MVP)
    # Default to "functional" per paradigm-selection.md guidance
    logger.info("Auto-detecting paradigm from requirements...")
    
    requirements_lower = requirements.lower()
    
    # Indicators for Graph API
    graph_indicators = [
        "parallel execution",
        "multiple parallel paths",
        "complex state management",
        "stategraph",
        "node-level checkpointing",
        "workflow visualization",
        "time-travel debugging",
        "multi-agent",
        "reducer",
        "conditional edges",
        "graph loop",
    ]
    
    # Indicators for Functional API
    functional_indicators = [
        "sequential",
        "simple conditional",
        "entrypoint",
        "@entrypoint",
        "simple workflow",
        "moderate complexity",
    ]
    
    # Count indicators
    graph_count = sum(1 for indicator in graph_indicators if indicator in requirements_lower)
    functional_count = sum(1 for indicator in functional_indicators if indicator in requirements_lower)
    
    # Determine paradigm
    if graph_count > functional_count and graph_count > 0:
        selected = "graph"
    else:
        # Default to functional per paradigm-selection.md
        selected = "functional"
    
    logger.info(f"Auto-detected paradigm: {selected} (graph indicators: {graph_count}, functional indicators: {functional_count})")
    return selected


def get_paradigm_guide(paradigm: str, guides: Dict[str, str]) -> str:
    """Get paradigm-specific guide content.
    
    Args:
        paradigm: Selected paradigm ("functional" or "graph")
        guides: Dictionary of all guides
        
    Returns:
        Formatted paradigm guide content
    """
    paradigm_guide_name = None
    if paradigm == "functional":
        paradigm_guide_name = "functional-api-implementation"
    elif paradigm == "graph":
        paradigm_guide_name = "graph-api-implementation"
    
    context_parts = []
    
    # Always include paradigm-selection guide for context
    if "paradigm-selection" in guides:
        context_parts.append(f"## Paradigm Selection Guide\n{guides['paradigm-selection']}\n\n")
    
    # Include paradigm-specific guide
    if paradigm_guide_name and paradigm_guide_name in guides:
        context_parts.append(f"## {paradigm.capitalize()} API Implementation Guide\n{guides[paradigm_guide_name]}\n\n")
    elif paradigm_guide_name:
        logger.warning(f"Paradigm guide not found: {paradigm_guide_name}")
    
    return "\n".join(context_parts)


def build_generation_prompt(
    requirements: str,
    custom_logic_specs: Optional[str],
    guides_context: str,
    paradigm: str,
    paradigm_guide: str,
) -> str:
    """Build complete prompt for LLM code generation.
    
    Args:
        requirements: Production requirements text
        custom_logic_specs: Custom logic specifications (optional)
        guides_context: Formatted guides context
        paradigm: Selected paradigm
        paradigm_guide: Paradigm-specific guide content
        
    Returns:
        Complete generation prompt
    """
    try:
        template = load_step3_prompt_template()
        
        # Build prompt sections
        prompt_parts = [template]
        
        # Add guides context
        if guides_context:
            prompt_parts.append("## Guides Context\n")
            prompt_parts.append(guides_context)
        
        # Add paradigm selection
        prompt_parts.append(f"\n## Paradigm Selection\n")
        prompt_parts.append(f"Selected Paradigm: {paradigm}\n")
        prompt_parts.append(paradigm_guide)
        
        # Add production requirements
        prompt_parts.append("\n## Production Requirements\n")
        prompt_parts.append(requirements)
        
        # Add custom logic specifications if provided
        if custom_logic_specs:
            prompt_parts.append("\n## Custom Logic Specifications\n")
            prompt_parts.append(custom_logic_specs)
        
        # Add task instruction
        prompt_parts.append(
            "\n## Task\n"
            "Generate complete LangGraph implementation code following the guides and requirements above. "
            "The code must be complete, runnable, and follow the selected paradigm. "
            "Include all necessary imports, decorators, and patterns as specified in the guides."
        )
        
        prompt = "\n".join(prompt_parts)
        
        logger.debug(f"Built generation prompt ({len(prompt)} chars)")
        return prompt
    except Exception as e:
        logger.error(f"Failed to build generation prompt: {e}")
        raise


def generate_code(
    prompt: str,
    provider: BaseLLMProvider,
) -> str:
    """Generate LangGraph code using LLM provider.
    
    Args:
        prompt: Complete generation prompt
        provider: LLM provider instance
        
    Returns:
        Raw LLM output text
        
    Raises:
        RuntimeError: If LLM generation fails
    """
    if not provider.is_initialized():
        provider.initialize()
    
    try:
        logger.info("Generating LangGraph code using LLM...")
        params = GenerationParams(
            temperature=0.2,  # Lower temperature for consistent code generation
            max_tokens=8000,  # Allow for complete implementation
        )
        
        response = provider.generate(prompt, params=params)
        code_output = response.content
        
        logger.info(f"Generated code output ({len(code_output)} chars)")
        return code_output
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise RuntimeError(f"Failed to generate code: {e}") from e


def extract_code_from_response(llm_output: str) -> str:
    """Extract Python code from LLM response.
    
    Handles cases where LLM includes markdown code blocks or plain text.
    
    Args:
        llm_output: Raw LLM output text
        
    Returns:
        Extracted Python code string
        
    Raises:
        ValueError: If no code can be extracted
    """
    # Try to extract from markdown code blocks first
    # Pattern: ```python ... ``` or ``` ... ```
    code_block_pattern = r"```(?:python)?\s*\n?(.*?)```"
    matches = re.findall(code_block_pattern, llm_output, re.DOTALL)
    
    if matches:
        # Use the first (and usually only) code block
        code = matches[0].strip()
        logger.debug(f"Extracted code from markdown code block ({len(code)} chars)")
        return code
    
    # If no code blocks found, check if the entire output looks like code
    # Simple heuristic: if it starts with import or from, it's likely code
    stripped_output = llm_output.strip()
    if stripped_output.startswith(("import ", "from ", "#", '"""', "'''")):
        logger.debug(f"Extracted code from plain text output ({len(stripped_output)} chars)")
        return stripped_output
    
    # If we can't extract code, raise error with snippet
    snippet = llm_output[:200] + "..." if len(llm_output) > 200 else llm_output
    raise ValueError(
        f"Could not extract Python code from LLM output. "
        f"Output snippet: {snippet}"
    )


def save_code_to_file(code: str, base_path: Optional[str] = None) -> str:
    """Save generated code to a file.
    
    Args:
        code: Generated Python code
        base_path: Base directory path (defaults to workspace root)
        
    Returns:
        Path to saved file
        
    Raises:
        IOError: If file cannot be written
    """
    if base_path is None:
        # Default to workspace root (parent of src/)
        base_path = Path(__file__).parent.parent.parent.parent
    
    base_path = Path(base_path)
    
    # Create filename with timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"generated_workflow_{timestamp}.py"
    file_path = base_path / filename
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"Saved generated code to {file_path}")
        return str(file_path)
    except Exception as e:
        raise IOError(f"Failed to write code to file: {e}") from e


def format_response(
    code: str,
    paradigm: str,
    output_format: str,
    file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Format response dictionary.
    
    Args:
        code: Generated Python code
        paradigm: Selected paradigm
        output_format: Output format ("code" or "file")
        file_path: File path if output_format is "file"
        
    Returns:
        Formatted response dictionary
    """
    response = {
        "status": "success",
        "code": code,  # Always include code
        "paradigm": paradigm,
        "output_format": output_format,
        "code_length": len(code),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    if output_format == "file" and file_path:
        response["file_path"] = file_path
    
    return response
