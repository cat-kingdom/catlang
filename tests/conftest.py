"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Set test environment variables
os.environ.setdefault("TESTING", "true")
