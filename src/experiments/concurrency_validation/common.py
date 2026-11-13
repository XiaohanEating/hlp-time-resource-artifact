import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'results/concurrency_validation'
SOURCE = ROOT / 'vendor/ShuttlebusMemory/src/circuit'
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20), b''):
            value.update(block)
    return value.hexdigest()


def save(name, result):
    OUTPUT.mkdir(exist_ok=True,parents=True)
    result['python']=sys.executable
    result['version']=sys.version
    result['source_hashes']={str(p.relative_to(ROOT)):digest(p)
                             for p in Path(__file__).parent.glob('*.py')}
    target=OUTPUT/(name+'.json')
    pending=target.with_suffix('.json.pending')
    pending.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pending.replace(target)
    print(target,flush=True)
