"""Constants for MCP Server.

This module defines all constants, enums, and error messages used throughout
the MCP server to avoid magic strings and improve maintainability.
"""

from enum import Enum


class ResponseStatus(str, Enum):
    """Response status values."""
    
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


class ErrorCode(str, Enum):
    """Error code identifiers for error categorization."""
    
    # Input validation errors
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_WORKFLOW_JSON = "INVALID_WORKFLOW_JSON"
    INVALID_CODE = "INVALID_CODE"
    INVALID_LANGUAGE = "INVALID_LANGUAGE"
    INVALID_REQUIREMENTS = "INVALID_REQUIREMENTS"
    INVALID_PARADIGM = "INVALID_PARADIGM"
    INVALID_OUTPUT_FORMAT = "INVALID_OUTPUT_FORMAT"
    
    # Provider errors
    LLM_PROVIDER_UNAVAILABLE = "LLM_PROVIDER_UNAVAILABLE"
    LLM_GENERATION_FAILED = "LLM_GENERATION_FAILED"
    
    # Resource errors
    RESOURCE_MANAGER_UNAVAILABLE = "RESOURCE_MANAGER_UNAVAILABLE"
    GUIDE_NOT_FOUND = "GUIDE_NOT_FOUND"
    
    # Validation errors
    VALIDATION_FAILED = "VALIDATION_FAILED"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    
    # Generic errors
    RUNTIME_ERROR = "RUNTIME_ERROR"
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"


class ErrorMessage:
    """Error message templates."""
    
    # Input validation
    INVALID_WORKFLOW_JSON = "Invalid n8n workflow JSON"
    INVALID_WORKFLOW_SCHEMA = "Invalid n8n workflow schema"
    INVALID_CODE = "Invalid code"
    INVALID_LANGUAGE = "Unsupported language. Supported: python, javascript"
    INVALID_REQUIREMENTS = "Invalid requirements"
    INVALID_PARADIGM = "Invalid paradigm. Must be 'functional', 'graph', or 'auto'"
    INVALID_OUTPUT_FORMAT = "Invalid output format. Must be 'code' or 'file'"
    
    # Provider errors
    LLM_PROVIDER_UNAVAILABLE = "LLM provider unavailable"
    LLM_GENERATION_FAILED = "Failed to generate content using LLM"
    
    # Resource errors
    RESOURCE_MANAGER_UNAVAILABLE = "Resource manager not initialized"
    GUIDE_NOT_FOUND = "Guide not found"
    GUIDE_NAME_EMPTY = "Guide name cannot be empty"
    
    # Context errors
    SERVER_CONTEXT_NOT_SET = "Server context not set"
    
    # Generic
    UNEXPECTED_ERROR = "Unexpected error occurred"
