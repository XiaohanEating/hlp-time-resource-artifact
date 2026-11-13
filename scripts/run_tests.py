"""Run scoped structural tests with the selected interpreter."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
sys.path.insert(0, str(root / 'vendor/ShuttlebusMemory/src/circuit'))
import pytest
raise SystemExit(pytest.main([str(root / 'src'), '-q']))
