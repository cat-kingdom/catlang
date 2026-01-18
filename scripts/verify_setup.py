#!/usr/bin/env python3
"""Script to verify development environment setup."""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version >= 3.10."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (need 3.10+)")
        return False


def check_package(package_name):
    """Check if a package is installed."""
    try:
        __import__(package_name)
        print(f"✓ {package_name} installed")
        return True
    except ImportError:
        print(f"✗ {package_name} not installed")
        return False


def check_directory(path):
    """Check if directory exists."""
    if Path(path).exists():
        print(f"✓ {path} exists")
        return True
    else:
        print(f"✗ {path} missing")
        return False


def check_file(path):
    """Check if file exists."""
    if Path(path).exists():
        print(f"✓ {path} exists")
        return True
    else:
        print(f"✗ {path} missing")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("CatLang Development Environment Verification")
    print("=" * 60)
    print()

    checks = []
    
    # Python version
    print("Checking Python version...")
    checks.append(check_python_version())
    print()

    # Required packages
    print("Checking required packages...")
    required_packages = [
        "langgraph",
        "langchain_openai",
        "mcp",
    ]
    for package in required_packages:
        checks.append(check_package(package))
    print()

    # Development packages
    print("Checking development packages...")
    dev_packages = [
        "pytest",
        "mypy",
        "ruff",
    ]
    for package in dev_packages:
        checks.append(check_package(package))
    print()

    # Project structure
    print("Checking project structure...")
    directories = [
        "src/mcp_server",
        "src/llm_provider",
        "src/workflow_engine",
        "config",
        "tests",
    ]
    for directory in directories:
        checks.append(check_directory(directory))
    print()

    # Configuration files
    print("Checking configuration files...")
    config_files = [
        "pyproject.toml",
        "config/server.yaml",
        "config/providers.yaml",
    ]
    for config_file in config_files:
        checks.append(check_file(config_file))
    print()

    # Summary
    print("=" * 60)
    passed = sum(checks)
    total = len(checks)
    print(f"Results: {passed}/{total} checks passed")
    print("=" * 60)

    if passed == total:
        print("✓ All checks passed! Environment is ready.")
        return 0
    else:
        print("✗ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
