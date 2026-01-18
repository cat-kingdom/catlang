"""Tests for MCP Server Fase 9 - Resource Management - Guides.

Tests cover:
- Guide indexing and metadata extraction
- Frontmatter parsing
- Resource management
- Resource handlers
- Tool handlers integration
- Search functionality
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
from typing import Any

from src.mcp_server.resources.guides import (
    GuideMetadata,
    GuideIndexer,
    GuideResourceManager,
    parse_frontmatter,
    infer_category,
    extract_title_from_content,
)
from src.mcp_server.resources.handlers import (
    set_resource_manager,
    list_all_guide_resources,
    get_guide_resource,
)
from src.mcp_server.tools.handlers import (
    list_guides,
    query_guide,
    set_server_instance,
    _resource_manager_tool,
)
from src.mcp_server.server import MCPServer


@pytest.fixture
def temp_guides_dir():
    """Create temporary directory with test guides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        guides_path = Path(tmpdir) / "guides"
        guides_path.mkdir()
        
        # Create test guide files
        # Guide 1: With frontmatter
        guide1_content = """---
title: "Functional API Guide"
category: "implementation"
tags: ["functional", "api"]
description: "Guide for functional API"
---
# Functional API Guide

This is a guide about functional API.
"""
        (guides_path / "functional-api.md").write_text(guide1_content)
        
        # Guide 2: Without frontmatter
        guide2_content = """# Paradigm Selection Guide

This guide helps you select the right paradigm.
"""
        (guides_path / "paradigm-selection.md").write_text(guide2_content)
        
        # Guide 3: With frontmatter but no category
        guide3_content = """---
title: "Testing Guide"
tags: ["testing", "debugging"]
---
# Testing Guide

This is about testing.
"""
        (guides_path / "testing-guide.md").write_text(guide3_content)
        
        yield guides_path


@pytest.mark.unit
class TestParseFrontmatter:
    """Test frontmatter parsing."""
    
    def test_parse_frontmatter_with_yaml(self):
        """Test parsing frontmatter with YAML."""
        content = """---
title: "Test Guide"
category: "test"
tags: ["tag1", "tag2"]
---
# Content

This is the content.
"""
        frontmatter, content_clean = parse_frontmatter(content)
        
        assert frontmatter["title"] == "Test Guide"
        assert frontmatter["category"] == "test"
        assert frontmatter["tags"] == ["tag1", "tag2"]
        assert "# Content" in content_clean
        assert "---" not in content_clean
    
    def test_parse_frontmatter_without_yaml(self):
        """Test parsing content without frontmatter."""
        content = """# Test Guide

This is the content.
"""
        frontmatter, content_clean = parse_frontmatter(content)
        
        assert frontmatter == {}
        assert content_clean == content
    
    def test_parse_frontmatter_invalid_yaml(self):
        """Test parsing with invalid YAML."""
        content = """---
title: "Test Guide"
invalid: yaml: syntax
---
# Content
"""
        frontmatter, content_clean = parse_frontmatter(content)
        
        # Should handle gracefully
        assert isinstance(frontmatter, dict)
        assert "# Content" in content_clean


@pytest.mark.unit
class TestInferCategory:
    """Test category inference."""
    
    def test_infer_from_frontmatter(self):
        """Test category inference from frontmatter."""
        frontmatter = {"category": "implementation"}
        category = infer_category(Path("test.md"), frontmatter)
        assert category == "implementation"
    
    def test_infer_from_filename_paradigm(self):
        """Test category inference from filename - paradigm."""
        frontmatter = {}
        category = infer_category(Path("paradigm-selection.md"), frontmatter)
        assert category == "paradigm"
    
    def test_infer_from_filename_implementation(self):
        """Test category inference from filename - implementation."""
        frontmatter = {}
        category = infer_category(Path("functional-api-implementation.md"), frontmatter)
        assert category == "implementation"
    
    def test_infer_from_filename_setup(self):
        """Test category inference from filename - setup."""
        frontmatter = {}
        category = infer_category(Path("authentication-setup.md"), frontmatter)
        assert category == "setup"
    
    def test_infer_default(self):
        """Test default category inference."""
        frontmatter = {}
        category = infer_category(Path("unknown-guide.md"), frontmatter)
        assert category == "general"


@pytest.mark.unit
class TestExtractTitleFromContent:
    """Test title extraction from content."""
    
    def test_extract_from_h1(self):
        """Test extracting title from H1 heading."""
        content = """# My Guide Title

Some content here.
"""
        title = extract_title_from_content(content, "my-guide")
        assert title == "My Guide Title"
    
    def test_extract_from_filename(self):
        """Test extracting title from filename."""
        content = """Some content without H1.
"""
        title = extract_title_from_content(content, "my-guide")
        assert title == "My Guide"


@pytest.mark.unit
class TestGuideMetadata:
    """Test GuideMetadata dataclass."""
    
    def test_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = GuideMetadata(
            name="test-guide",
            category="test",
            tags=["tag1"],
            title="Test Guide",
            description="A test guide",
            file_path=Path("test.md"),
            content="# Test",
            last_modified=datetime(2025, 1, 18),
            uri="guide://docs/test/test-guide",
        )
        
        result = metadata.to_dict()
        assert result["name"] == "test-guide"
        assert result["category"] == "test"
        assert result["tags"] == ["tag1"]
        assert result["title"] == "Test Guide"
        assert result["uri"] == "guide://docs/test/test-guide"
    
    def test_to_resource_dict(self):
        """Test converting to MCP resource format."""
        metadata = GuideMetadata(
            name="test-guide",
            category="test",
            tags=["tag1"],
            title="Test Guide",
            description="A test guide",
            file_path=Path("test.md"),
            content="# Test",
            uri="guide://docs/test/test-guide",
        )
        
        result = metadata.to_resource_dict()
        assert result["uri"] == "guide://docs/test/test-guide"
        assert result["name"] == "Test Guide"
        assert result["mimeType"] == "text/markdown"
        assert "metadata" in result


@pytest.mark.unit
class TestGuideIndexer:
    """Test GuideIndexer class."""
    
    def test_scan_guides_directory(self, temp_guides_dir):
        """Test scanning guides directory."""
        indexer = GuideIndexer(temp_guides_dir)
        index = indexer.scan_guides_directory()
        
        assert len(index) == 3
        assert "functional-api" in index
        assert "paradigm-selection" in index
        assert "testing-guide" in index
    
    def test_index_guide_with_frontmatter(self, temp_guides_dir):
        """Test indexing guide with frontmatter."""
        indexer = GuideIndexer(temp_guides_dir)
        guide_file = temp_guides_dir / "functional-api.md"
        metadata = indexer._index_guide_file(guide_file)
        
        assert metadata is not None
        assert metadata.name == "functional-api"
        assert metadata.category == "implementation"
        assert metadata.title == "Functional API Guide"
        assert "functional" in metadata.tags
        assert metadata.uri == "guide://docs/implementation/functional-api"
    
    def test_index_guide_without_frontmatter(self, temp_guides_dir):
        """Test indexing guide without frontmatter."""
        indexer = GuideIndexer(temp_guides_dir)
        guide_file = temp_guides_dir / "paradigm-selection.md"
        metadata = indexer._index_guide_file(guide_file)
        
        assert metadata is not None
        assert metadata.name == "paradigm-selection"
        assert metadata.category == "paradigm"
        assert metadata.title == "Paradigm Selection Guide"
        assert metadata.tags == []
    
    def test_build_index(self, temp_guides_dir):
        """Test building index."""
        indexer = GuideIndexer(temp_guides_dir)
        index = indexer.build_index()
        
        assert len(index) == 3
        assert indexer._initialized is True
    
    def test_get_guide(self, temp_guides_dir):
        """Test getting guide by name."""
        indexer = GuideIndexer(temp_guides_dir)
        indexer.build_index()
        
        guide = indexer.get_guide("functional-api")
        assert guide is not None
        assert guide.name == "functional-api"
        
        guide = indexer.get_guide("nonexistent")
        assert guide is None
    
    def test_get_guide_with_category(self, temp_guides_dir):
        """Test getting guide with category filter."""
        indexer = GuideIndexer(temp_guides_dir)
        indexer.build_index()
        
        guide = indexer.get_guide("functional-api", category="implementation")
        assert guide is not None
        
        guide = indexer.get_guide("functional-api", category="wrong")
        assert guide is None
    
    def test_list_guides_all(self, temp_guides_dir):
        """Test listing all guides."""
        indexer = GuideIndexer(temp_guides_dir)
        indexer.build_index()
        
        guides = indexer.list_guides()
        assert len(guides) == 3
    
    def test_list_guides_by_category(self, temp_guides_dir):
        """Test listing guides by category."""
        indexer = GuideIndexer(temp_guides_dir)
        indexer.build_index()
        
        guides = indexer.list_guides(category="implementation")
        assert len(guides) == 1
        assert guides[0].name == "functional-api"
    
    def test_list_guides_by_tags(self, temp_guides_dir):
        """Test listing guides by tags."""
        indexer = GuideIndexer(temp_guides_dir)
        indexer.build_index()
        
        guides = indexer.list_guides(tags=["functional"])
        assert len(guides) == 1
        assert guides[0].name == "functional-api"
    
    def test_list_guides_by_category_and_tags(self, temp_guides_dir):
        """Test listing guides with multiple filters."""
        indexer = GuideIndexer(temp_guides_dir)
        indexer.build_index()
        
        guides = indexer.list_guides(category="implementation", tags=["functional"])
        assert len(guides) == 1
    
    def test_search_guides(self, temp_guides_dir):
        """Test searching guides."""
        indexer = GuideIndexer(temp_guides_dir)
        indexer.build_index()
        
        results = indexer.search_guides("functional")
        assert len(results) == 1
        assert results[0].name == "functional-api"
        
        results = indexer.search_guides("paradigm")
        assert len(results) == 1
        assert results[0].name == "paradigm-selection"
    
    def test_search_guides_with_category(self, temp_guides_dir):
        """Test searching guides with category filter."""
        indexer = GuideIndexer(temp_guides_dir)
        indexer.build_index()
        
        results = indexer.search_guides("guide", category="implementation")
        assert len(results) == 1


@pytest.mark.unit
class TestGuideResourceManager:
    """Test GuideResourceManager class."""
    
    def test_initialization(self, temp_guides_dir):
        """Test resource manager initialization."""
        manager = GuideResourceManager(temp_guides_dir)
        assert not manager._initialized
        
        manager.initialize()
        assert manager._initialized
        assert len(manager.indexer._index) == 3
    
    def test_get_resource(self, temp_guides_dir):
        """Test getting resource by URI."""
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        
        resource = manager.get_resource("guide://docs/implementation/functional-api")
        assert resource is not None
        assert resource["uri"] == "guide://docs/implementation/functional-api"
        assert "contents" in resource
        assert len(resource["contents"]) == 1
        assert resource["contents"][0]["mimeType"] == "text/markdown"
    
    def test_get_resource_invalid_uri(self, temp_guides_dir):
        """Test getting resource with invalid URI."""
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        
        resource = manager.get_resource("invalid://uri")
        assert resource is None
    
    def test_get_resource_not_found(self, temp_guides_dir):
        """Test getting non-existent resource."""
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        
        resource = manager.get_resource("guide://docs/implementation/nonexistent")
        assert resource is None
    
    def test_list_resources(self, temp_guides_dir):
        """Test listing all resources."""
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        
        resources = manager.list_resources()
        assert len(resources) == 3
        assert all("uri" in r for r in resources)
        assert all("name" in r for r in resources)
    
    def test_get_guide_content(self, temp_guides_dir):
        """Test getting guide content."""
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        
        content = manager.get_guide_content("functional-api")
        assert content is not None
        assert "# Functional API Guide" in content
        
        content = manager.get_guide_content("nonexistent")
        assert content is None


@pytest.mark.unit
class TestResourceHandlers:
    """Test resource handlers."""
    
    def test_set_resource_manager(self, temp_guides_dir):
        """Test setting resource manager."""
        manager = GuideResourceManager(temp_guides_dir)
        set_resource_manager(manager)
        # Handler should be set (no exception)
    
    @pytest.mark.asyncio
    async def test_list_all_guide_resources(self, temp_guides_dir):
        """Test listing all guide resources."""
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        set_resource_manager(manager)
        
        resources = await list_all_guide_resources()
        assert len(resources) == 3
    
    @pytest.mark.asyncio
    async def test_list_all_guide_resources_no_manager(self):
        """Test listing resources without manager."""
        set_resource_manager(None)
        resources = await list_all_guide_resources()
        assert resources == []
    
    @pytest.mark.asyncio
    async def test_get_guide_resource(self, temp_guides_dir):
        """Test getting guide resource."""
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        set_resource_manager(manager)
        
        resource = await get_guide_resource("implementation", "functional-api")
        assert resource is not None
        assert resource["uri"] == "guide://docs/implementation/functional-api"
    
    @pytest.mark.asyncio
    async def test_get_guide_resource_not_found(self, temp_guides_dir):
        """Test getting non-existent resource."""
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        set_resource_manager(manager)
        
        resource = await get_guide_resource("implementation", "nonexistent")
        assert resource == {}


@pytest.mark.integration
class TestToolHandlersIntegration:
    """Test tool handlers integration with resource manager."""
    
    @pytest.mark.asyncio
    async def test_list_guides_tool(self, temp_guides_dir):
        """Test list_guides tool handler."""
        # Create mock server with resource manager
        mock_server = MagicMock()
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        mock_server.resource_manager = manager
        
        set_server_instance(mock_server)
        
        result = await list_guides()
        assert result["status"] == "success"
        assert result["count"] == 3
        assert len(result["guides"]) == 3
    
    @pytest.mark.asyncio
    async def test_list_guides_with_category_filter(self, temp_guides_dir):
        """Test list_guides with category filter."""
        mock_server = MagicMock()
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        mock_server.resource_manager = manager
        
        set_server_instance(mock_server)
        
        result = await list_guides(category="implementation")
        assert result["status"] == "success"
        assert result["count"] == 1
        assert result["guides"][0]["category"] == "implementation"
    
    @pytest.mark.asyncio
    async def test_list_guides_with_tags_filter(self, temp_guides_dir):
        """Test list_guides with tags filter."""
        mock_server = MagicMock()
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        mock_server.resource_manager = manager
        
        set_server_instance(mock_server)
        
        result = await list_guides(tags=["functional"])
        assert result["status"] == "success"
        assert result["count"] == 1
    
    @pytest.mark.asyncio
    async def test_query_guide_tool(self, temp_guides_dir):
        """Test query_guide tool handler."""
        mock_server = MagicMock()
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        mock_server.resource_manager = manager
        
        set_server_instance(mock_server)
        
        result = await query_guide("functional-api")
        assert result["status"] == "success"
        assert result["guide_name"] == "functional-api"
        assert "content" in result
        assert result["content"] is not None
    
    @pytest.mark.asyncio
    async def test_query_guide_with_category(self, temp_guides_dir):
        """Test query_guide with category."""
        mock_server = MagicMock()
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        mock_server.resource_manager = manager
        
        set_server_instance(mock_server)
        
        result = await query_guide("functional-api", category="implementation")
        assert result["status"] == "success"
        assert result["category"] == "implementation"
    
    @pytest.mark.asyncio
    async def test_query_guide_not_found(self, temp_guides_dir):
        """Test query_guide with non-existent guide."""
        mock_server = MagicMock()
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        mock_server.resource_manager = manager
        
        set_server_instance(mock_server)
        
        result = await query_guide("nonexistent")
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_query_guide_empty_name(self, temp_guides_dir):
        """Test query_guide with empty name."""
        mock_server = MagicMock()
        manager = GuideResourceManager(temp_guides_dir)
        manager.initialize()
        mock_server.resource_manager = manager
        
        set_server_instance(mock_server)
        
        result = await query_guide("")
        assert result["status"] == "error"
        assert "empty" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_list_guides_no_resource_manager(self):
        """Test list_guides without resource manager."""
        # Clear global state
        set_server_instance(None)
        import src.mcp_server.tools.handlers as handlers_module
        handlers_module._resource_manager_tool = None
        
        result = await list_guides()
        assert result["status"] == "error"
        assert "not initialized" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_query_guide_no_resource_manager(self):
        """Test query_guide without resource manager."""
        # Clear global state
        set_server_instance(None)
        import src.mcp_server.tools.handlers as handlers_module
        handlers_module._resource_manager_tool = None
        
        result = await query_guide("test")
        assert result["status"] == "error"
        assert "not initialized" in result["error"].lower()


@pytest.mark.integration
class TestServerIntegration:
    """Test server integration with resources."""
    
    def test_server_initializes_resource_manager(self, temp_guides_dir):
        """Test that server initializes resource manager."""
        config = {
            "workspace": {
                "path": str(temp_guides_dir.parent),
                "guides_path": "guides",
            }
        }
        
        server = MCPServer(config)
        assert server.resource_manager is not None
        assert server.resource_manager.guides_path == temp_guides_dir
    
    def test_server_registers_resources(self, temp_guides_dir):
        """Test that server registers resources."""
        config = {
            "workspace": {
                "path": str(temp_guides_dir.parent),
                "guides_path": "guides",
            }
        }
        
        server = MCPServer(config)
        server._register_resources()
        
        # Resource manager should be initialized
        assert server.resource_manager._initialized is True
        
        # Resources should be registered (check by trying to list)
        resources = server.resource_manager.list_resources()
        assert len(resources) == 3
