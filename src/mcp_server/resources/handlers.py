"""Resource handlers for MCP Server.

This module provides resource handler functions for FastMCP registration.
"""

import logging
from typing import Any, Dict, List, Optional

from .guides import GuideResourceManager

logger = logging.getLogger(__name__)

# Global resource manager instance (set by server)
_resource_manager: Optional[GuideResourceManager] = None


def set_resource_manager(manager: GuideResourceManager) -> None:
    """Set resource manager for handler access.
    
    Args:
        manager: GuideResourceManager instance
    """
    global _resource_manager
    _resource_manager = manager
    logger.debug("Resource manager set for handlers")


async def list_all_guide_resources() -> List[Dict[str, Any]]:
    """List all guide resources.
    
    Returns:
        List of resource dictionaries in MCP format
    """
    if _resource_manager is None:
        logger.error("Resource manager not initialized")
        return []
    
    try:
        resources = _resource_manager.list_resources()
        logger.debug(f"Listed {len(resources)} guide resources")
        return resources
    except Exception as e:
        logger.error(f"Failed to list guide resources: {e}", exc_info=True)
        return []


async def get_guide_resource(category: str, name: str) -> Dict[str, Any]:
    """Get a specific guide resource by category and name.
    
    Args:
        category: Guide category
        name: Guide name (filename without .md)
        
    Returns:
        Resource dictionary in MCP format, or empty dict if not found
    """
    if _resource_manager is None:
        logger.error("Resource manager not initialized")
        return {}
    
    try:
        uri = f"guide://docs/{category}/{name}"
        resource = _resource_manager.get_resource(uri)
        if resource:
            logger.debug(f"Retrieved guide resource: {uri}")
            return resource
        else:
            logger.warning(f"Guide resource not found: {uri}")
            return {}
    except Exception as e:
        logger.error(f"Failed to get guide resource: {e}", exc_info=True)
        return {}
