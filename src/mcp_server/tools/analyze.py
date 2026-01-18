"""n8n workflow analysis module.

This module provides functions to analyze n8n workflow JSON and generate
production requirements using LLM providers.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...llm_provider.base import BaseLLMProvider, GenerationParams

logger = logging.getLogger(__name__)

# Path to Step1 prompt template
STEP1_PROMPT_PATH = Path(__file__).parent.parent.parent.parent / "OrchestrationPrompts" / "Step1-ProductionRequirements.md"


def load_step1_prompt_template() -> str:
    """Load Step1 prompt template from file.
    
    Returns:
        Prompt template as string
        
    Raises:
        FileNotFoundError: If prompt template file doesn't exist
        IOError: If file cannot be read
    """
    if not STEP1_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Step1 prompt template not found at {STEP1_PROMPT_PATH}"
        )
    
    try:
        with open(STEP1_PROMPT_PATH, "r", encoding="utf-8") as f:
            template = f.read()
        logger.debug(f"Loaded Step1 prompt template ({len(template)} chars)")
        return template
    except Exception as e:
        raise IOError(f"Failed to read prompt template: {e}") from e


def parse_n8n_workflow(workflow_json: str) -> Dict[str, Any]:
    """Parse n8n workflow JSON string.
    
    Args:
        workflow_json: n8n workflow JSON as string
        
    Returns:
        Parsed workflow dictionary
        
    Raises:
        ValueError: If JSON is invalid or malformed
    """
    try:
        workflow_data = json.loads(workflow_json)
        logger.debug(f"Parsed n8n workflow JSON ({len(str(workflow_data))} chars)")
        return workflow_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to parse workflow JSON: {e}") from e


def validate_n8n_schema(workflow_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate n8n workflow schema structure.
    
    Args:
        workflow_data: Parsed workflow dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(workflow_data, dict):
        return False, "Workflow data must be a dictionary"
    
    # Check for required top-level fields
    if "nodes" not in workflow_data:
        return False, "Missing required field: 'nodes'"
    
    if "connections" not in workflow_data:
        return False, "Missing required field: 'connections'"
    
    # Validate nodes structure
    nodes = workflow_data.get("nodes", [])
    if not isinstance(nodes, list):
        return False, "'nodes' must be a list"
    
    # Validate connections structure
    connections = workflow_data.get("connections", {})
    if not isinstance(connections, dict):
        return False, "'connections' must be a dictionary"
    
    # Basic validation passed
    return True, None


def extract_nodes(workflow_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract node information from workflow data.
    
    Args:
        workflow_data: Parsed workflow dictionary
        
    Returns:
        List of node dictionaries
    """
    nodes = workflow_data.get("nodes", [])
    if not isinstance(nodes, list):
        return []
    
    logger.debug(f"Extracted {len(nodes)} nodes from workflow")
    return nodes


def extract_connections(workflow_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract connection graph from workflow data.
    
    Args:
        workflow_data: Parsed workflow dictionary
        
    Returns:
        Connection dictionary
    """
    connections = workflow_data.get("connections", {})
    if not isinstance(connections, dict):
        return {}
    
    logger.debug(f"Extracted connections from workflow")
    return connections


def identify_custom_nodes(nodes: List[Dict[str, Any]]) -> List[str]:
    """Identify custom nodes in the workflow.
    
    Custom nodes are typically Function nodes, Code nodes, or nodes
    that are not part of the standard n8n library.
    
    Args:
        nodes: List of node dictionaries
        
    Returns:
        List of custom node names
    """
    custom_node_types = {
        "n8n-nodes-base.function",
        "n8n-nodes-base.code",
        "n8n-nodes-base.executeCommand",
    }
    
    custom_nodes = []
    for node in nodes:
        node_type = node.get("type", "")
        node_name = node.get("name", "")
        
        # Check if it's a known custom node type
        if node_type in custom_node_types:
            custom_nodes.append(node_name or node_type)
        # Also check for nodes with custom code/function content
        elif node.get("parameters", {}).get("functionCode") or node.get("parameters", {}).get("jsCode"):
            custom_nodes.append(node_name or node_type)
    
    logger.debug(f"Identified {len(custom_nodes)} custom nodes: {custom_nodes}")
    return custom_nodes


def extract_workflow_metadata(workflow_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from workflow.
    
    Args:
        workflow_data: Parsed workflow dictionary
        
    Returns:
        Metadata dictionary
    """
    nodes = extract_nodes(workflow_data)
    custom_nodes = identify_custom_nodes(nodes)
    
    metadata = {
        "node_count": len(nodes),
        "custom_node_count": len(custom_nodes),
        "custom_nodes": custom_nodes,
        "workflow_name": workflow_data.get("name", "Unnamed Workflow"),
        "workflow_id": workflow_data.get("id"),
        "workflow_version": workflow_data.get("version"),
    }
    
    # Determine complexity based on node count and custom nodes
    if metadata["node_count"] <= 5:
        metadata["complexity"] = "simple"
    elif metadata["node_count"] <= 15:
        metadata["complexity"] = "moderate"
    else:
        metadata["complexity"] = "complex"
    
    logger.debug(f"Extracted workflow metadata: {metadata}")
    return metadata


def build_analysis_prompt(workflow_json: str) -> str:
    """Build LLM prompt for workflow analysis.
    
    Args:
        workflow_json: n8n workflow JSON as string
        
    Returns:
        Complete prompt string
    """
    try:
        template = load_step1_prompt_template()
        
        # Replace placeholder with actual workflow JSON
        prompt = template.replace("[PASTE YOUR N8N JSON CODE HERE]", workflow_json)
        
        logger.debug(f"Built analysis prompt ({len(prompt)} chars)")
        return prompt
    except Exception as e:
        logger.error(f"Failed to build analysis prompt: {e}")
        raise


def generate_requirements(
    workflow_json: str,
    provider: BaseLLMProvider,
) -> str:
    """Generate production requirements using LLM provider.
    
    Args:
        workflow_json: n8n workflow JSON as string
        provider: LLM provider instance
        
    Returns:
        Generated requirements text
        
    Raises:
        RuntimeError: If LLM generation fails
    """
    if not provider.is_initialized():
        provider.initialize()
    
    try:
        # Build prompt
        prompt = build_analysis_prompt(workflow_json)
        
        # Generate requirements using LLM
        logger.info("Generating requirements using LLM...")
        params = GenerationParams(
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=4000,  # Allow for detailed analysis
        )
        
        response = provider.generate(prompt, params=params)
        requirements = response.content
        
        logger.info(f"Generated requirements ({len(requirements)} chars)")
        return requirements
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise RuntimeError(f"Failed to generate requirements: {e}") from e


def format_llm_response(
    llm_output: str,
    metadata: Dict[str, Any],
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """Format LLM output into structured response.
    
    Args:
        llm_output: Raw LLM output text
        metadata: Workflow metadata dictionary
        include_metadata: Whether to include metadata in response
        
    Returns:
        Structured response dictionary
    """
    response = {
        "status": "success",
        "requirements": llm_output,  # LLM output is already formatted text
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    if include_metadata:
        response["metadata"] = {
            "node_count": metadata.get("node_count", 0),
            "custom_node_count": metadata.get("custom_node_count", 0),
            "custom_nodes": metadata.get("custom_nodes", []),
            "complexity": metadata.get("complexity", "unknown"),
            "workflow_name": metadata.get("workflow_name", "Unknown"),
        }
    
    return response
