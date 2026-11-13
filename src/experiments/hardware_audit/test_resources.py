"""Check resource classification independently of the large cached circuits."""
import pytest

from .resources import local_resources, stored_circuit_counts
from surface_code import Rotated_surface_code
import json
from .resources import ROOT, static_resources


@pytest.mark.parametrize('ports', [4, 6, 8])
def test_regenerated_resources_match_archived_fields(ports):
    saved = json.loads((ROOT / 'results/hardware_audit/resources.json').read_text())
    expected = next(r for r in saved['records'] if r['ports'] == ports)
    actual = static_resources(16, ports)
    for key, value in actual.items():
        if key not in ('source', 'source_sha256'):
            assert value == expected[key], key


def test_distance_three_patch_resource_counts():
    counts = local_resources(Rotated_surface_code(3))
    assert (counts['qubits'], counts['data'], counts['ancillas']) == (17, 9, 8)
    assert counts['local_edges'] == 24
    assert list(counts['local_layer_counts'].values()) == [6, 6, 6, 6]


def test_flat_source_counts_local_cross_and_separate_readout_slots(tmp_path):
    path = tmp_path / 'sample.stim'
    path.write_text('''QUBIT_COORDS(0,-2) 0
QUBIT_COORDS(1,-3) 1
QUBIT_COORDS(0,0) 2
QUBIT_COORDS(1,-1) 3
CX 0 1 2 3
DEPOLARIZE2(0.001) 0 1 2 3
TICK
CX[tag] 0 2
TICK
M 0 1
TICK
MX 2 3
''', encoding='utf-8')
    # The production function records paths relative to ROOT.
    from . import resources as module
    previous = module.ROOT
    try:
        module.ROOT = tmp_path
        data = stored_circuit_counts(path)
    finally:
        module.ROOT = previous
    assert data['local_edge_count'] == 2
    assert data['cross_edge_count'] == 1
    assert data['max_local_gates'] == 2
    assert data['max_cross_gates'] == 1
    assert data['measurement_layer_histogram'] == {2: 2}


def test_repeated_circuit_is_rejected_instead_of_silently_undercounted(tmp_path):
    path = tmp_path / 'repeat.stim'
    path.write_text('REPEAT 2 {\nTICK\n}\n', encoding='utf-8')
    with pytest.raises(AssertionError, match='flat cached circuit'):
        stored_circuit_counts(path)
