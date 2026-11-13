"""Verify and decompress archived text; never overwrite differing files."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
root = Path(__file__).resolve().parents[1]
records = json.loads((root / 'provenance/assembly.json').read_text())['records']
for r in records:
    if not r['published'].endswith('.gz'):
        continue
    source = root / r['published']
    if hashlib.sha256(source.read_bytes()).hexdigest() != r['published_sha256']:
        raise ValueError(f'Compressed input changed: {source.name}')
    target = source.with_suffix('')
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != r.get('uncompressed_sha256', r['original_sha256']):
            raise ValueError(f'Refusing to overwrite {target}')
        continue
    pending = target.with_suffix(target.suffix + '.pending')
    with gzip.open(source, 'rb') as inp, pending.open('wb') as out:
        shutil.copyfileobj(inp, out, 1 << 20)
    h = hashlib.sha256()
    with pending.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    if h.hexdigest() != r.get('uncompressed_sha256', r['original_sha256']):
        raise ValueError(f'Uncompressed input changed: {source.name}')
    pending.replace(target)
    print(target.relative_to(root))
