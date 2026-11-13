"""Independent checks against archived endpoint and integrity records."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from .model import Evidence, ceildiv, cost

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def identity(bundle):
    return bundle['bus'], bundle['port'], tuple(bundle['target'])

def check_certificate(proof, dependencies):
    """Check exact coverage, endpoint exclusion, cap and per-batch lower bound."""
    expected = {(r['group'], r['batch']): r['bundles'] for r in dependencies['batches']}
    assert len(expected) == len(dependencies['batches'])
    seen_batches = set()
    total = 0
    for row in proof['batches']:
        key = row['group'], row['batch']
        assert key not in seen_batches
        seen_batches.add(key)
        bundles = expected[key]
        lookup = {identity(b): b for b in bundles}
        assert len(lookup) == len(bundles)
        observed = Counter()
        for layer in row['sublayers']:
            assert layer
            occupied = set()
            gate_count = 0
            for bundle in layer:
                identifier = identity(bundle)
                observed[identifier] += 1
                pairs = lookup[identifier]['pairs']
                assert len(pairs) == 25
                for pair in pairs:
                    a, b = map(tuple, pair)
                    assert a != b and a not in occupied and b not in occupied
                    occupied.update((a, b))
                    gate_count += 1
            assert gate_count <= proof['cap']
        assert observed == Counter(lookup.keys())
        assert len(row['sublayers']) == row['lower_bound'] == ceildiv(25 * len(bundles), proof['cap'])
        total += len(row['sublayers'])
    assert seen_batches == set(expected)
    assert total == proof['cross_gate_layers'] == cost(proof['ports'], 16, proof['cap']).layers
    return total

def verify(root):
    root = Path(root)
    manifest = json.loads((root / 'provenance/release_manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        assert sha256(root / row['path']) == row['sha256'], row['path']
    evidence = Evidence()
    certificates = 0
    physical_gates = 0
    for q in (4, 6, 8):
        with gzip.open(root / f'results/concurrency_validation/n16_q{q}_dependency_batches.json.gz', 'rt', encoding='utf-8') as f:
            dependencies = json.load(f)
        for cap in (800, 1600, 2400, 3200, 6400):
            proof = json.loads((root / f'results/concurrency_validation/n16_q{q}_cap{cap}_certificate.json').read_text())
            check_certificate(proof, dependencies)
            certificates += 1
        physical_gates += sum(len(b['pairs']) for r in dependencies['batches'] for b in r['bundles'])
    evidence.export()
    evidence.assert_sources_unchanged()
    result = dict(integrity_files=len(manifest['files']),
                  combinations_recomputed=evidence.checks['candidate_cap_pairs_recomputed'],
                  pareto_sets_matched=evidence.checks['pareto_sets_matched'],
                  physical_certificates_verified=certificates,
                  physical_gate_instances_in_three_dependency_maps=physical_gates,
                  new_noisy_samples=0, decoder_runs=0)
    (root / 'output/verification.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
