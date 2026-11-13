"""Checks for independent failure modes in the experiment auditors."""
import itertools
import stim
from .geometry import graph_distance, inspect
from .allocation import costs_at_deadline
from .circuit import fault_signature_audit
from surface_code import Rotated_surface_code


def test_static_graph_distance_matches_exhaustive_small_css():
    # Three-bit repetition: Z-checks, X logical distance three.
    checks=[0b011,0b110]
    distance,_=graph_distance(checks,0b001,3)
    brute=min(e.bit_count() for e in range(1,8)
              if all((e&r).bit_count()%2==0 for r in checks) and e&1)
    assert distance==brute==3
    # Rotated surface code gives an independent known symmetric distance.
    result=inspect(Rotated_surface_code(3))
    assert [v['distance'] for v in result['static_distances'].values()]==[3,3]


def test_signature_audit_detects_single_and_pair_logical_ambiguity():
    assert fault_signature_audit(stim.DetectorErrorModel('error(0.01) L0'))[
        'has_undetected_at_most_two_signature_candidate']
    result=fault_signature_audit(stim.DetectorErrorModel('error(0.01) D0\nerror(0.02) D0 L0'))
    assert result['two_signature_collision_examples']
    assert not fault_signature_audit(stim.DetectorErrorModel('error(0.01) D0 L0\nerror(0.02) D1'))[
        'has_undetected_at_most_two_signature_candidate']


def test_allocation_dp_matches_exhaustive_bus_multisets():
    n=4
    for deadline in (12,17,24,27):
        candidate=costs_at_deadline(n,deadline)
        brute=10**12
        for count in range(1,n+1):
            for ports in itertools.combinations_with_replacement(range(1,n+1),count):
                durations=[12+5*((n+q-1)//q-1) for q in ports]
                if sum(deadline//d for d in durations)>=n:
                    brute=min(brute,sum(50*q-1 for q in ports))
        assert candidate['joint_min_bus_qubits']==brute
