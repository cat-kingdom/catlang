"""Main entry point for MCP Server.

This script can be used to run the MCP server directly.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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
