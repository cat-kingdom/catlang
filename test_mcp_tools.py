#!/usr/bin/env python3
"""Test script to verify MCP tools are registered correctly."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.mcp_server.server import create_server


async def test_tools():
    """Test if tools are registered in FastMCP."""
    print("Creating MCP server...")
    server = create_server()
    
    print("\nRegistering capabilities...")
    server._register_capabilities()
    
    print("\nChecking FastMCP tools...")
    # Check if tools are registered in FastMCP
    if hasattr(server.mcp, '_tool_manager'):
        tool_manager = server.mcp._tool_manager
        if hasattr(tool_manager, '_tools'):
            tools = tool_manager._tools
            print(f"✓ Found {len(tools)} tools in FastMCP:")
            for tool_name in tools.keys():
                print(f"  - {tool_name}")
        else:
            print("⚠ Tool manager doesn't have _tools attribute")
    else:
        print("⚠ FastMCP doesn't have _tool_manager attribute")
    
    # Try to list tools using FastMCP's list_tools method (async)
    print("\nUsing FastMCP.list_tools() (async)...")
    try:
        tools_list = await server.mcp.list_tools()
        print(f"✓ FastMCP.list_tools() returned {len(tools_list)} tools:")
        for tool in tools_list:
            print(f"  - {tool.name}: {tool.description[:50]}...")
    except Exception as e:
        print(f"❌ Error calling list_tools(): {e}")
    
    # Check internal registry
    print("\nChecking internal registry...")
    registry_tools = server.tool_registry.list_tools()
    print(f"✓ Internal registry has {len(registry_tools)} tools:")
    for tool_name in registry_tools:
        print(f"  - {tool_name}")
    
    print("\n" + "="*50)
    print("Summary:")
    print(f"  - FastMCP tools: {len(tools_list) if 'tools_list' in locals() else 'N/A'}")
    print(f"  - Internal registry: {len(registry_tools)}")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(test_tools())
