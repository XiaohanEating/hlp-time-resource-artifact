"""Paths and provenance shared by the capacity validation experiments."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CIRCUIT_SOURCE = ROOT / 'vendor/ShuttlebusMemory/src/circuit'
OUTPUT = ROOT / 'results/capacity_validation'
if str(CIRCUIT_SOURCE) not in sys.path:
    sys.path.insert(0, str(CIRCUIT_SOURCE))


def save_result(name, result):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result['python'] = sys.version
    result['executable'] = sys.executable
    paths = sorted(Path(__file__).parent.glob('*.py')) + [
        CIRCUIT_SOURCE / 'surface_code.py', CIRCUIT_SOURCE / 'lv1_codes.py',
        CIRCUIT_SOURCE / 'hierarchical_circuit.py', CIRCUIT_SOURCE / 'high_level_utils.py']
    result['source_sha256'] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    target = OUTPUT / (name + '.json')
    pending = target.with_suffix('.json.pending')
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending.replace(target)
    print(target)

