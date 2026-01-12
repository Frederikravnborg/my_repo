"""Pytest configuration file."""

import os
import sys
from pathlib import Path

# Add src directory to Python path so pytest can find my_project
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Ensure matplotlib uses a headless backend in CI / local test runs.
os.environ.setdefault("MPLBACKEND", "Agg")

