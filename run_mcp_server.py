#!/usr/bin/env python3
"""Convenience script to run the MCP server.

Usage:
    python run_mcp_server.py

Or make it executable:
    chmod +x run_mcp_server.py
    ./run_mcp_server.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.mcp_server.server import create_server


async def main() -> None:
    """Main entry point for running the MCP server."""
    try:
        server = create_server()
        await server.start()
    except KeyboardInterrupt:
        print("\nServer interrupted by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Failed to start server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
