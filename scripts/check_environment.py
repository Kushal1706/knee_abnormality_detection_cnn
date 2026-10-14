"""Check the local development environment."""

import platform
import sys
from pathlib import Path


def main() -> None:
    """Print basic project environment information."""
    project_root = Path(__file__).resolve().parents[1]

    print("Knee Abnormality Detection - Environment Check")
    print("-" * 48)
    print(f"Python version : {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"Operating system: {platform.system()}")
    print(f"Project root   : {project_root}")
    print(f"Virtual environment: {sys.prefix != sys.base_prefix}")


if __name__ == "__main__":
    main()
