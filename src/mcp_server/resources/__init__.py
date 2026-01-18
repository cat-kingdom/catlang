"""Resource management for MCP Server.

This package provides resource management functionality for exposing
implementation guides as MCP resources.
"""

from .guides import (
    GuideMetadata,
    GuideIndexer,
    GuideResourceManager,
    parse_frontmatter,
    infer_category,
)

__all__ = [
    "GuideMetadata",
    "GuideIndexer",
    "GuideResourceManager",
    "parse_frontmatter",
    "infer_category",
]
