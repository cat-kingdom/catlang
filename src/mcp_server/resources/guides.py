"""Guide resource management for MCP Server.

This module provides functionality to index, manage, and expose
implementation guides as MCP resources.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class GuideMetadata:
    """Metadata for an implementation guide."""
    
    name: str  # Filename without .md extension
    category: str  # Category (e.g., "implementation", "paradigm", "setup")
    tags: List[str] = field(default_factory=list)  # Tags for categorization
    title: str = ""  # Display title
    description: Optional[str] = None  # Guide description
    file_path: Path = field(default_factory=Path)  # Full path to guide file
    content: str = ""  # Full markdown content
    last_modified: Optional[datetime] = None  # Last modification time
    uri: str = ""  # MCP resource URI (guide://docs/{category}/{name})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "tags": self.tags,
            "title": self.title,
            "description": self.description,
            "uri": self.uri,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
        }
    
    def to_resource_dict(self) -> Dict[str, Any]:
        """Convert to MCP resource format."""
        return {
            "uri": self.uri,
            "name": self.title or self.name,
            "description": self.description or f"Guide: {self.name}",
            "mimeType": "text/markdown",
            "metadata": {
                "category": self.category,
                "tags": self.tags,
                "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            },
        }


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Extract YAML frontmatter from markdown content.
    
    Args:
        content: Markdown content with optional YAML frontmatter
        
    Returns:
        Tuple of (frontmatter_dict, content_without_frontmatter)
    """
    frontmatter = {}
    content_clean = content
    
    # Check for frontmatter delimiters
    if content.startswith("---"):
        # Find the closing delimiter
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter_str = parts[1].strip()
            content_clean = parts[2].lstrip()
            
            # Parse YAML frontmatter
            if frontmatter_str:
                try:
                    frontmatter = yaml.safe_load(frontmatter_str) or {}
                except yaml.YAMLError as e:
                    logger.warning(f"Failed to parse frontmatter YAML: {e}")
                    frontmatter = {}
    
    return frontmatter, content_clean


def infer_category(file_path: Path, frontmatter: Dict[str, Any]) -> str:
    """Infer category from filename or frontmatter.
    
    Args:
        file_path: Path to guide file
        frontmatter: Parsed frontmatter dictionary
        
    Returns:
        Inferred category string
    """
    # Check frontmatter first
    if "category" in frontmatter:
        return str(frontmatter["category"])
    
    # Infer from filename patterns
    filename = file_path.stem.lower()
    
    # Common patterns
    if "paradigm" in filename or "selection" in filename:
        return "paradigm"
    elif "implementation" in filename or "api" in filename:
        return "implementation"
    elif "setup" in filename or "authentication" in filename:
        return "setup"
    elif "testing" in filename or "troubleshooting" in filename:
        return "testing"
    elif "integration" in filename:
        return "integration"
    elif "structure" in filename:
        return "structure"
    elif "output" in filename or "requirements" in filename:
        return "requirements"
    
    # Default category
    return "general"


def extract_title_from_content(content: str, filename: str) -> str:
    """Extract title from markdown content or filename.
    
    Args:
        content: Markdown content
        filename: Guide filename (without extension)
        
    Returns:
        Title string
    """
    # Try to find first H1 heading
    h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    
    # Fallback: capitalize filename
    return filename.replace("-", " ").replace("_", " ").title()


class GuideIndexer:
    """Index and manage implementation guides."""
    
    def __init__(self, guides_path: Path):
        """Initialize guide indexer.
        
        Args:
            guides_path: Path to guides directory
        """
        self.guides_path = Path(guides_path)
        self._index: Dict[str, GuideMetadata] = {}
        self._initialized = False
    
    def scan_guides_directory(self) -> Dict[str, GuideMetadata]:
        """Scan guides directory and build index.
        
        Returns:
            Dictionary mapping guide name to GuideMetadata
        """
        index = {}
        
        if not self.guides_path.exists():
            logger.warning(f"Guides directory not found: {self.guides_path}")
            return index
        
        if not self.guides_path.is_dir():
            logger.warning(f"Guides path is not a directory: {self.guides_path}")
            return index
        
        # Scan for .md files
        for guide_file in self.guides_path.glob("*.md"):
            try:
                metadata = self._index_guide_file(guide_file)
                if metadata:
                    index[metadata.name] = metadata
                    logger.debug(f"Indexed guide: {metadata.name} ({metadata.category})")
            except Exception as e:
                logger.error(f"Failed to index guide {guide_file}: {e}", exc_info=True)
        
        logger.info(f"Indexed {len(index)} guides from {self.guides_path}")
        return index
    
    def _index_guide_file(self, file_path: Path) -> Optional[GuideMetadata]:
        """Index a single guide file.
        
        Args:
            file_path: Path to guide file
            
        Returns:
            GuideMetadata if successful, None otherwise
        """
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Parse frontmatter
            frontmatter, content_clean = parse_frontmatter(content)
            
            # Extract metadata
            name = file_path.stem
            category = infer_category(file_path, frontmatter)
            tags = frontmatter.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            elif not isinstance(tags, list):
                tags = []
            
            title = frontmatter.get("title") or extract_title_from_content(content_clean, name)
            description = frontmatter.get("description")
            
            # Get last modified time
            try:
                last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            except OSError:
                last_modified = None
            
            # Build URI
            uri = f"guide://docs/{category}/{name}"
            
            # Create metadata
            metadata = GuideMetadata(
                name=name,
                category=category,
                tags=tags,
                title=title,
                description=description,
                file_path=file_path,
                content=content_clean,  # Content without frontmatter
                last_modified=last_modified,
                uri=uri,
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to index guide file {file_path}: {e}", exc_info=True)
            return None
    
    def build_index(self) -> Dict[str, GuideMetadata]:
        """Build guide index.
        
        Returns:
            Dictionary mapping guide name to GuideMetadata
        """
        self._index = self.scan_guides_directory()
        self._initialized = True
        return self._index
    
    def get_guide(self, guide_name: str, category: Optional[str] = None) -> Optional[GuideMetadata]:
        """Get guide by name.
        
        Args:
            guide_name: Guide name (filename without .md)
            category: Optional category filter
            
        Returns:
            GuideMetadata if found, None otherwise
        """
        if not self._initialized:
            self.build_index()
        
        guide = self._index.get(guide_name)
        if guide and (category is None or guide.category == category):
            return guide
        return None
    
    def list_guides(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[GuideMetadata]:
        """List guides with optional filters.
        
        Args:
            category: Filter by category (optional)
            tags: Filter by tags (any tag match, optional)
            
        Returns:
            List of GuideMetadata matching filters
        """
        if not self._initialized:
            self.build_index()
        
        guides = list(self._index.values())
        
        # Filter by category
        if category:
            guides = [g for g in guides if g.category == category]
        
        # Filter by tags (any tag match)
        if tags:
            tag_set = set(t.lower() for t in tags)
            guides = [
                g for g in guides
                if any(tag.lower() in tag_set for tag in g.tags)
            ]
        
        # Sort by name
        guides.sort(key=lambda g: g.name)
        return guides
    
    def search_guides(
        self,
        query: str,
        category: Optional[str] = None,
    ) -> List[GuideMetadata]:
        """Search guides by text query.
        
        Args:
            query: Search query (searches in title and description)
            category: Optional category filter
            
        Returns:
            List of matching GuideMetadata
        """
        if not self._initialized:
            self.build_index()
        
        query_lower = query.lower()
        guides = self.list_guides(category=category)
        
        # Simple text search in title and description
        matches = []
        for guide in guides:
            if (
                query_lower in guide.title.lower()
                or (guide.description and query_lower in guide.description.lower())
                or query_lower in guide.name.lower()
            ):
                matches.append(guide)
        
        return matches


class GuideResourceManager:
    """Manage guide resources for MCP server."""
    
    def __init__(self, guides_path: Path):
        """Initialize resource manager.
        
        Args:
            guides_path: Path to guides directory
        """
        self.guides_path = Path(guides_path)
        self.indexer = GuideIndexer(guides_path)
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize resource manager by building index."""
        if not self._initialized:
            self.indexer.build_index()
            self._initialized = True
            logger.info("Guide resource manager initialized")
    
    def get_resource(self, uri: str) -> Optional[Dict[str, Any]]:
        """Get resource by URI.
        
        Args:
            uri: Resource URI (e.g., guide://docs/{category}/{name})
            
        Returns:
            Resource dictionary in MCP format, or None if not found
        """
        if not self._initialized:
            self.initialize()
        
        # Parse URI: guide://docs/{category}/{name}
        match = re.match(r"guide://docs/([^/]+)/(.+)", uri)
        if not match:
            logger.warning(f"Invalid resource URI format: {uri}")
            return None
        
        category, name = match.groups()
        guide = self.indexer.get_guide(name, category=category)
        
        if not guide:
            logger.debug(f"Guide not found: {uri}")
            return None
        
        # Return resource with content
        resource = guide.to_resource_dict()
        resource["contents"] = [
            {
                "uri": guide.uri,
                "mimeType": "text/markdown",
                "text": guide.content,
            }
        ]
        return resource
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List all resources.
        
        Returns:
            List of resource dictionaries in MCP format
        """
        if not self._initialized:
            self.initialize()
        
        guides = self.indexer.list_guides()
        return [guide.to_resource_dict() for guide in guides]
    
    def get_guide_content(
        self,
        guide_name: str,
        category: Optional[str] = None,
    ) -> Optional[str]:
        """Get guide content by name.
        
        Args:
            guide_name: Guide name
            category: Optional category filter
            
        Returns:
            Guide content (markdown), or None if not found
        """
        if not self._initialized:
            self.initialize()
        
        guide = self.indexer.get_guide(guide_name, category=category)
        if guide:
            return guide.content
        return None
