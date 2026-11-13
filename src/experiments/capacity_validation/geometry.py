"""Exact CSS checks and graph distance for existing rectangular bus geometry.

The distance computation uses the degree-at-most-two check incidence graph,
lifted by logical parity. It is exact for these static CSS matrices, not a
circuit fault-distance or decoder-performance calculation.
"""
from collections import deque
import time

from .common import save_result
from surface_code import ShuttleBus_code, Rotated_surface_code, QubitType
from gate_types import Pauli


def rank(rows):
    pivots = {}
    for value in rows:
        while value:
            lead = value.bit_length() - 1
            if lead in pivots:
                value ^= pivots[lead]
            else:
                pivots[lead] = value
                break
    return len(pivots)


def matrix(code):
    data = [q for q in code.qubits if q.qubit_type == QubitType.Data]
    indices = {q.position: i for i, q in enumerate(data)}
    def mask(qubits):
        return sum(1 << indices[q.position] for q in qubits)
    checks = {}
    for basis, kind in [('X', QubitType.X_ancilla), ('Z', QubitType.Z_ancilla)]:
        checks[basis] = [mask(q.SEIndex_2_neighbor_qubit.values())
                         for q in code.qubits if q.qubit_type == kind]
    logical = {'X': mask(code.logical_X_operator), 'Z': mask(code.logical_Z_operator)}
    return data, checks, logical


def graph_distance(checks, logical_opposite, n):
    boundary = len(checks)
    adjacency = [[] for _ in range(boundary + 1)]
    for i in range(n):
        ends = [j for j, row in enumerate(checks) if (row >> i) & 1]
        if len(ends) > 2:
            raise ValueError('Static check matrix is not graphlike')
        ends += [boundary] * (2 - len(ends))
        u, v = ends
        parity = (logical_opposite >> i) & 1
        adjacency[u].append((v, parity, i))
        adjacency[v].append((u, parity, i))
    best = n + 1
    witness = None
    for start in range(len(adjacency)):
        queue = deque([(start, 0, 0, 0)])
        visited = {(start, 0)}
        while queue:
            vertex, parity, length, error = queue.popleft()
            if length >= best:
                continue
            for neighbor, flip, index in adjacency[vertex]:
                state = (neighbor, parity ^ flip)
                if state == (start, 1):
                    best, witness = length + 1, error ^ (1 << index)
                    queue.clear()
                    break
                if state not in visited:
                    visited.add(state)
                    queue.append((neighbor, parity ^ flip, length + 1, error ^ (1 << index)))
    assert witness is not None
    assert witness.bit_count() == best
    assert all((row & witness).bit_count() % 2 == 0 for row in checks)
    assert (logical_opposite & witness).bit_count() % 2 == 1
    return best, [i for i in range(n) if (witness >> i) & 1]


def inspect(code):
    data, checks, logical = matrix(code)
    assert all((x & z).bit_count() % 2 == 0 for x in checks['X'] for z in checks['Z'])
    assert (logical['X'] & logical['Z']).bit_count() % 2 == 1
    k = len(data) - rank(checks['X']) - rank(checks['Z'])
    assert k == 1
    distances = {}
    for basis, opposite in [('X', 'Z'), ('Z', 'X')]:
        assert all((logical[basis] & row).bit_count() % 2 == 0 for row in checks[opposite])
        distance, witness = graph_distance(checks[opposite], logical[opposite], len(data))
        distances[basis] = dict(distance=distance,
                                witness_data_positions=[data[i].position for i in witness])
    # Verify that each existing SE gate layer uses every physical qubit at most once.
    for layer in range(1, 5):
        occupied = set()
        for anc in code.qubits:
            if anc.qubit_type == QubitType.Data or layer not in anc.SEIndex_2_neighbor_qubit:
                continue
            pair = (anc.position, anc.SEIndex_2_neighbor_qubit[layer].position)
            assert not occupied.intersection(pair)
            occupied.update(pair)
    return dict(data_qubits=len(data), total_qubits=len(code.qubits), logical_qubits=k,
                ranks={b: rank(rows) for b, rows in checks.items()},
                commute=True, se_layers_conflict_free=True, static_distances=distances)


def main():
    started = time.perf_counter()
    core = Rotated_surface_code(5)
    results = []
    for ports in (4, 6, 8):
        for basis in (Pauli.X, Pauli.Z):
            bus = ShuttleBus_code(5, ports, basis)
            result = inspect(bus)
            occupied = set()
            per_port = []
            for port in bus.ports:
                mapping = bus.dock(core, port)
                bus_data = {p for p in mapping if p in bus.position_2_qubit
                            and bus.position_2_qubit[p].qubit_type == QubitType.Data}
                assert len(bus_data) == 25
                assert not occupied.intersection(bus_data)
                occupied.update(bus_data)
                assert all(core.position_2_qubit[mapping[p]].qubit_type == QubitType.Data
                           for p in bus_data)
                per_port.append(len(bus_data))
            result.update(ports=ports, long_basis=basis.basis,
                          parallel_port_data_pairs=per_port,
                          inter_gate_single_fault_max_new_data_blocks=1,
                          note='Disjoint transversal layer only; subsequent SE/decoder propagation untested')
            results.append(result)
    save_result('geometry', dict(scope='Static matrices and existing physical gate mappings',
                                core=inspect(core), buses=results,
                                elapsed_seconds=time.perf_counter() - started))


if __name__ == '__main__':
    main()
