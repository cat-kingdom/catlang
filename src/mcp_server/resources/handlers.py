"""Resource handlers for MCP Server.

This module provides resource handler functions for FastMCP registration.
"""

import logging
from typing import Any

from ..context import get_resource_manager
from ..decorators import handle_errors

logger = logging.getLogger(__name__)


@handle_errors
async def list_all_guide_resources() -> list[dict[str, Any]]:
    """List all guide resources.
    
    Returns:
        List of resource dictionaries in MCP format
    """
    resource_manager = get_resource_manager()
    resources = resource_manager.list_resources()
    logger.debug(f"Listed {len(resources)} guide resources")
    return resources


@handle_errors
async def get_guide_resource(category: str, name: str) -> dict[str, Any]:
    """Get a specific guide resource by category and name.
    
    Args:
        category: Guide category
        name: Guide name (filename without .md)
        
    Returns:
        Resource dictionary in MCP format, or empty dict if not found
    """
    resource_manager = get_resource_manager()
    uri = f"guide://docs/{category}/{name}"
    resource = resource_manager.get_resource(uri)
    if resource:
        logger.debug(f"Retrieved guide resource: {uri}")
        return resource
    else:
        logger.warning(f"Guide resource not found: {uri}")
        return {}
