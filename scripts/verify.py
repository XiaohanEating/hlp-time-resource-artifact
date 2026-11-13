"""Verify the distributed archive and derive exact finite-domain outputs."""
from pathlib import Path
import json
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
from hlp_artifact.verify import verify
print(json.dumps(verify(root), indent=2))
