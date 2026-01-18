"""Tool schema definitions for MCP Server.

This module defines JSON Schema schemas for all MCP tools.
"""

from typing import Any

# Tool schema definitions following JSON Schema specification
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "analyze_n8n_workflow": {
        "name": "analyze_n8n_workflow",
        "description": "Analyze an n8n workflow JSON and generate production requirements for LangGraph implementation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_json": {
                    "type": "string",
                    "description": "The n8n workflow JSON as a string",
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Whether to include workflow metadata in the analysis",
                    "default": True,
                },
            },
            "required": ["workflow_json"],
        },
    },
    "extract_custom_logic": {
        "name": "extract_custom_logic",
        "description": "Extract custom logic from n8n custom nodes (Python/JavaScript code) and generate requirements specification.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The custom node code (Python or JavaScript)",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript"],
                    "description": "Programming language of the code",
                },
                "node_name": {
                    "type": "string",
                    "description": "Name of the custom node (optional, for context)",
                },
            },
            "required": ["code", "language"],
        },
    },
    "generate_langgraph_implementation": {
        "name": "generate_langgraph_implementation",
        "description": "Generate LangGraph implementation code from production requirements and custom logic specifications.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "requirements": {
                    "type": "string",
                    "description": "Production requirements from analyze_n8n_workflow",
                },
                "custom_logic_specs": {
                    "type": "string",
                    "description": "Custom logic specifications from extract_custom_logic (optional)",
                },
                "paradigm": {
                    "type": "string",
                    "enum": ["functional", "graph", "auto"],
                    "description": "LangGraph paradigm to use (functional/graph/auto)",
                    "default": "auto",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["code", "file"],
                    "description": "Output format (code string or file path)",
                    "default": "code",
                },
            },
            "required": ["requirements"],
        },
    },
    "validate_implementation": {
        "name": "validate_implementation",
        "description": "Validate generated LangGraph implementation code for syntax errors, compliance, and best practices.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The LangGraph implementation code to validate",
                },
                "check_syntax": {
                    "type": "boolean",
                    "description": "Check Python syntax errors",
                    "default": True,
                },
                "check_compliance": {
                    "type": "boolean",
                    "description": "Check LangGraph pattern compliance",
                    "default": True,
                },
                "check_best_practices": {
                    "type": "boolean",
                    "description": "Check best practices and generate suggestions",
                    "default": True,
                },
            },
            "required": ["code"],
        },
    },
    "list_guides": {
        "name": "list_guides",
        "description": "List all available implementation guides with metadata (category, tags, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter guides by category (optional)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter guides by tags (optional)",
                },
            },
            "required": [],
        },
    },
    "query_guide": {
        "name": "query_guide",
        "description": "Query a specific implementation guide by name or URI.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "guide_name": {
                    "type": "string",
                    "description": "Name of the guide to query (e.g., 'functional-paradigm')",
                },
                "category": {
                    "type": "string",
                    "description": "Category of the guide (optional, helps with lookup)",
                },
            },
            "required": ["guide_name"],
        },
    },
}


class ToolSchema:
    """Tool schema wrapper for easier access."""
    
    def __init__(self, name: str, schema: dict[str, Any]):
        """Initialize tool schema.
        
        Args:
            name: Tool name
            schema: JSON Schema dictionary
        """
        self.name = name
        self.schema = schema
        self.description = schema.get("description", "")
        self.input_schema = schema.get("inputSchema", {})
    
    def get_required_params(self) -> list[str]:
        """Get list of required parameter names.
        
        Returns:
            List of required parameter names
        """
        return self.input_schema.get("required", [])
    
    def get_properties(self) -> dict[str, Any]:
        """Get parameter properties.
        
        Returns:
            Dictionary of parameter properties
        """
        return self.input_schema.get("properties", {})
    
    def validate_required(self, params: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate that all required parameters are present.
        
        Args:
            params: Parameters dictionary
            
        Returns:
            Tuple of (is_valid, missing_params)
        """
        required = self.get_required_params()
        missing = [param for param in required if param not in params]
        return len(missing) == 0, missing


def get_tool_schema(name: str) -> ToolSchema | None:
    """Get tool schema by name.
    
    Args:
        name: Tool name
        
    Returns:
        ToolSchema instance if found, None otherwise
    """
    if name not in TOOL_SCHEMAS:
        return None
    return ToolSchema(name, TOOL_SCHEMAS[name])


def get_tool_schemas() -> dict[str, ToolSchema]:
    """Get all tool schemas.
    
    Returns:
        Dictionary mapping tool names to ToolSchema instances
    """
    return {
        name: ToolSchema(name, schema)
        for name, schema in TOOL_SCHEMAS.items()
    }
