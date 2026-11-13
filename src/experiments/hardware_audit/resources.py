"""Count requested hardware resources without assuming a fabricated layout.

Static-edge counts assume fixed endpoint identities and a dedicated pairwise
connection for each requested pair. They do not count shared mediators, physical
microwave channels, routed length, or manufacturing layers.
"""
from collections import Counter
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

from ..concurrency_validation.common import ROOT, digest
from surface_code import Rotated_surface_code, ShuttleBus_code, QubitType
from gate_types import Pauli

OUTPUT = ROOT / 'results/hardware_audit'


def local_resources(code):
    counts = Counter(q.qubit_type.value for q in code.qubits)
    layers = Counter()
    degree = Counter()
    edges = set()
    for ancilla in code.qubits:
        if ancilla.qubit_type == QubitType.Data:
            continue
        for index, data in ancilla.SEIndex_2_neighbor_qubit.items():
            edge = tuple(sorted((ancilla.position, data.position)))
            assert edge not in edges
            edges.add(edge)
            layers[index] += 1
            degree[ancilla.position] += 1
            degree[data.position] += 1
    assert len(edges) == sum(layers.values())
    return dict(qubits=len(code.qubits), data=counts['Data'],
                ancillas=len(code.qubits)-counts['Data'],
                local_edges=len(edges), local_layer_counts=dict(sorted(layers.items())),
                degree=degree)


def static_resources(n, ports):
    path = ROOT / f'results/capacity_validation/connectivity_n{n}_q{ports}.json'
    edges = [tuple(map(tuple, pair)) for pair in json.loads(path.read_text())]
    assert len(set(edges)) == len(edges)
    assert all(a[0] == 'bus' and b[0] == 'core' and a != b for a, b in edges)
    degree = Counter(vertex for edge in edges for vertex in edge)
    core = local_resources(Rotated_surface_code(5))
    bus = local_resources(ShuttleBus_code(5, ports, Pauli.Z))
    bus_x = local_resources(ShuttleBus_code(5, ports, Pauli.X))
    assert bus == bus_x, 'X/Z relabelling must preserve the physical local graph'
    combined = Counter()
    for v, cross_degree in degree.items():
        local = bus if v[0] == 'bus' else core
        combined[v] = cross_degree + local['degree'][tuple(v[-2:])]
    vertex_count = len(degree)
    planar_limit = 2*vertex_count-4
    local_edges = n*n*core['local_edges'] + n*bus['local_edges']
    local_layers = [n*n*core['local_layer_counts'][i] + n*bus['local_layer_counts'][i]
                    for i in (1, 2, 3, 4)]
    return dict(n=n, ports=ports, buses=n, source=str(path.relative_to(ROOT)),
                source_sha256=digest(path),
                qubits=n*n*core['qubits'] + n*bus['qubits'],
                core_data=n*n*core['data'], bus_data=n*bus['data'],
                core_ancillas=n*n*core['ancillas'], bus_ancillas=n*bus['ancillas'],
                cross_edges=len(edges), cross_vertices=vertex_count,
                cross_operations_per_cycle=4*n*n*25,
                peak_cross_gates=n*ports*25,
                max_bus_cross_degree=max(v for k, v in degree.items() if k[0] == 'bus'),
                max_core_cross_degree=max(v for k, v in degree.items() if k[0] == 'core'),
                max_bus_total_degree=max(v for k, v in combined.items() if k[0] == 'bus'),
                max_core_total_degree=max(v for k, v in combined.items() if k[0] == 'core'),
                bipartite_planar_edge_limit=planar_limit,
                edges_exceeding_single_plane_bound=max(0, len(edges)-planar_limit),
                abstract_planar_edge_partition_lower_bound=math.ceil(len(edges)/planar_limit),
                local_edges=local_edges, total_requested_pairwise_edges=local_edges+len(edges),
                synchronized_local_gate_layers=local_layers,
                local_measurements_per_synchronized_round=n*n*core['ancillas']+n*bus['ancillas'],
                bus_data_measurements_per_group=n*bus['data'])


def stored_circuit_counts(path):
    """Stream the existing flat Stim source and check physical layer counts."""
    coords = {}
    local_edges, cross_edges = set(), set()
    layer_local, layer_cross, measurement_count = 0, 0, 0
    local_hist, cross_hist, measurement_hist = Counter(), Counter(), Counter()
    occupied = set()
    def close_layer():
        if layer_local:
            local_hist[layer_local] += 1
        if layer_cross:
            cross_hist[layer_cross] += 1
        if measurement_count:
            measurement_hist[measurement_count] += 1
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            assert not line.startswith('REPEAT'), 'This audit requires a flat cached circuit'
            if line.startswith('QUBIT_COORDS('):
                match = re.fullmatch(r'QUBIT_COORDS\(([^)]+)\) (\d+)', line)
                assert match
                coords[int(match[2])] = [float(x) for x in match[1].split(',')]
                continue
            name = line.split(' ', 1)[0].split('[', 1)[0].split('(', 1)[0]
            if name == 'TICK':
                close_layer()
                layer_local = layer_cross = measurement_count = 0
                occupied.clear()
            elif name == 'CX':
                targets = [int(x) for x in line.split(' ', 1)[1].split()]
                assert len(targets) % 2 == 0
                assert len(set(targets)) == len(targets)
                assert not occupied.intersection(targets)
                occupied.update(targets)
                for a, b in zip(targets[::2], targets[1::2]):
                    pair = tuple(sorted((a, b)))
                    cross = (coords[a][1] < -1) != (coords[b][1] < -1)
                    if cross:
                        cross_edges.add(pair)
                        layer_cross += 1
                    else:
                        local_edges.add(pair)
                        layer_local += 1
            elif name in ('M', 'MX', 'MY', 'MR', 'MRX', 'MRY'):
                measurement_count += len(line.split(' ', 1)[1].split())
    close_layer()
    return dict(source=str(path.relative_to(ROOT)), sha256=digest(path),
                local_edge_count=len(local_edges), cross_edge_count=len(cross_edges),
                local_layer_histogram=dict(sorted(local_hist.items())),
                cross_layer_histogram=dict(sorted(cross_hist.items())),
                measurement_layer_histogram=dict(sorted(measurement_hist.items())),
                max_local_gates=max(local_hist), max_cross_gates=max(cross_hist))


def main():
    started = time.perf_counter()
    records = [static_resources(16, q) for q in (4, 6, 8)]
    checks = []
    for n, q, name in [
        (16, 4, 'circuit_cache/squareberg_n16_pool16_alpha_1_d5_r1.stim'),
        (8, 8, 'capacity_validation/n8_q8_d5_r2.stim'),
    ]:
        measured = stored_circuit_counts(ROOT / 'results' / name)
        model = static_resources(n, q)
        assert measured['local_edge_count'] == model['local_edges']
        assert measured['cross_edge_count'] == model['cross_edges']
        assert measured['max_local_gates'] == max(model['synchronized_local_gate_layers'])
        assert measured['max_cross_gates'] == model['peak_cross_gates']
        checks.append(measured)
    result = dict(scope=__doc__, records=records, cached_circuit_cross_checks=checks,
                  python=sys.executable, python_version=platform.python_version(),
                  elapsed_seconds=time.perf_counter()-started,
                  source_hashes={str(p.relative_to(ROOT)): digest(p) for p in [
                      Path(__file__),
                      ROOT/'vendor/ShuttlebusMemory/src/circuit/surface_code.py',
                      ROOT/'src/experiments/concurrency_validation/common.py']})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pending = OUTPUT / 'resources.json.pending'
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    pending.replace(OUTPUT / 'resources.json')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
