import pytest
import stim
from .frontier import cost,physical_batches,certificate
from .rebuild import serialize,normalized_operations,is_cross


def test_workload_bound_is_attained_and_every_physical_target_is_retained():
    batches=physical_batches(8,8)
    for cap in (400,800,1200,1600):
        proof=certificate(8,8,cap,batches)
        for row in proof['batches']:
            original=batches[row['group'],row['batch']]
            expected=sorted((x['bus'],x['port'],x['target']) for x in original)
            observed=sorted((x['bus'],x['port'],x['target']) for layer in row['sublayers'] for x in layer)
            assert expected==observed
            assert all(25*len(layer)<=cap for layer in row['sublayers'])
        assert proof['lower_bound_attained']


def test_reference_times_and_capacity_price():
    assert [cost(16,q,16,6400) for q in (4,6,8)]==[(108,16),(88,12),(68,8)]
    assert [cost(16,q,16,1600) for q in (4,6,8)]==[(108,16),(88,20),(68,16)]
    assert cost(16,8,16,2400)==(68,16)


@pytest.mark.skipif(tuple(map(int, stim.__version__.split('.')[:2])) < (1, 15),
                    reason='Tagged circuit reconstruction requires Stim 1.15; no environment changes made')
def test_serialization_preserves_ideal_logic_and_checks_bus_boundary_ancilla():
    c=stim.Circuit('''
        QUBIT_COORDS(0,0) 0
        QUBIT_COORDS(2,0) 1
        QUBIT_COORDS(0,-4) 2
        QUBIT_COORDS(2,-4) 3
        R 0 1 2 3
        TICK
        CX 0 2 1 3
        TICK
        M 0 1 2 3
        DETECTOR rec[-4] rec[-2]
        DETECTOR rec[-3] rec[-1]
    ''')
    transformed,info=serialize(c,1)
    assert info['extra_cross_layers']==1
    assert info['peak_cross_gates']==1
    assert normalized_operations(c)==normalized_operations(transformed)
    transformed.detector_error_model(allow_gauge_detectors=False)
    assert not transformed.compile_detector_sampler(seed=1).sample(8).any()
    assert not is_cross(stim.CircuitInstruction('CX',[0,1]),{0:[3,-1],1:[2,0]})


def test_shared_qubit_cross_gate_layer_is_rejected():
    c=stim.Circuit('QUBIT_COORDS(0,0) 0\nQUBIT_COORDS(0,-4) 1\nQUBIT_COORDS(2,-4) 2\nCX 0 1 0 2')
    with pytest.raises(AssertionError):serialize(c,1)
