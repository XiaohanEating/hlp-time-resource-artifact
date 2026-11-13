"""Build small full HLP memory circuits and audit deterministic fault signatures."""
import argparse
import hashlib
import time
from collections import Counter
import stim

from .common import OUTPUT, save_result
from .schedule import CapacitySquareBerg
from hierarchical_circuit import HierarchicalCircuit, NoiseConfig, Noise
from lv1_circ_gen import AbstractCircuit


def fault_signature_audit(dem):
    """Check all undecomposed DEM signatures for undetected weight <= 2.

    A collision is a candidate requiring source-location verification. No
    collision rules out these low-order signature combinations, but does not
    certify correction of two faults or distance five.
    """
    syndrome_to_obs = {}
    singles, pairs = [], []
    sizes = Counter()
    count = 0
    for instruction in dem.flattened():
        if instruction.type != 'error':
            continue
        detectors, observables = set(), set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors.symmetric_difference_update([target.val])
            elif target.is_logical_observable_id():
                observables.symmetric_difference_update([target.val])
            else:
                raise ValueError('Expected undecomposed DEM without separators')
        key, obs = tuple(sorted(detectors)), tuple(sorted(observables))
        count += 1
        sizes[len(key)] += 1
        if not key and obs:
            singles.append(dict(detectors=key, observables=obs))
        if key in syndrome_to_obs and syndrome_to_obs[key] != obs:
            if len(pairs) < 10:
                pairs.append(dict(detectors=key, first=syndrome_to_obs[key], second=obs))
        else:
            syndrome_to_obs[key] = obs
    return dict(error_signatures=count, detector_weight_histogram=dict(sizes),
                undetected_single_signatures=singles,
                two_signature_collision_examples=pairs,
                has_undetected_at_most_two_signature_candidate=bool(singles or pairs),
                scope='All undecomposed DEM terms; no decoder correction or distance-five certificate')


def run(n, ports, rounds):
    label = f'n{n}_q{ports}_d5_r{rounds}'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = dict(n=n, ports=ports, distance=5, memory_rounds=rounds,
                  noise_probability=0.001, stim_version=stim.__version__, seed=41819,
                  status='building', stages={})
    start = time.perf_counter()
    try:
        config = NoiseConfig()
        for noise in (Noise.IDLE, Noise.SQGATE, Noise.TQGATE,
                      Noise.Z_RESET, Noise.X_RESET, Noise.Z_MEAS, Noise.X_MEAS):
            config.add_noise((noise, 0.001))
        physical = HierarchicalCircuit(CapacitySquareBerg(n,n,ports),5,ports,noise_config=config)
        abstract = AbstractCircuit(CapacitySquareBerg(n,n,ports),5,ports)
        physical.perfect_reset()
        abstract.perf_reset()
        cycles = []
        for cycle in range(rounds):
            before, abstract_before = physical.time, abstract.clock_time
            physical.logical_SE_round()
            abstract.logical_se_round()
            delta, abstract_delta = physical.time-before, abstract.clock_time-abstract_before
            cycles.append(dict(cycle=cycle, physical_half_round_units=delta,
                               abstract_half_round_units=abstract_delta,
                               physical_rounds=delta/2))
            assert delta == abstract_delta
            print(label, 'cycle', cycle, cycles[-1], flush=True)
        physical.perfect_meas()
        abstract.perf_meas()
        circuit = physical.circuit.circuit
        file = OUTPUT / (label+'.stim')
        pending = file.with_suffix('.stim.pending')
        circuit.to_file(pending)
        pending.replace(file)
        result.update(cycles=cycles, num_qubits=circuit.num_qubits,
                      num_detectors=circuit.num_detectors, num_observables=circuit.num_observables,
                      num_ticks=circuit.num_ticks, circuit_file=str(file),
                      circuit_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        result['stages']['build_seconds'] = time.perf_counter()-start
        print(label, 'built', result['stages'], flush=True)
        stage=time.perf_counter()
        # Determinism validation includes every declared detector and observable.
        dem = circuit.detector_error_model(decompose_errors=False,
                                          approximate_disjoint_errors=False,
                                          allow_gauge_detectors=False)
        dem_file = OUTPUT/(label+'.dem')
        pending_dem = dem_file.with_suffix('.dem.pending')
        dem.to_file(pending_dem)
        pending_dem.replace(dem_file)
        result['dem_file']=str(dem_file)
        result['dem_sha256']=hashlib.sha256(dem_file.read_bytes()).hexdigest()
        result['stages']['dem_seconds']=time.perf_counter()-stage
        stage=time.perf_counter()
        dets, obs = circuit.without_noise().compile_detector_sampler(seed=41819).sample(
            shots=32,separate_observables=True)
        result['noiseless_check']=dict(shots=32, detector_events=int(dets.sum()),
                                      observable_events=int(obs.sum()))
        assert not dets.any() and not obs.any()
        result['stages']['noiseless_seconds']=time.perf_counter()-stage
        stage=time.perf_counter()
        result['fault_audit']=fault_signature_audit(dem)
        result['stages']['fault_signature_seconds']=time.perf_counter()-stage
        result['status']='determinism_and_low_order_signature_checks_complete'
    except Exception as exc:
        result['status']='failed'
        result['error']=repr(exc)
        raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-start
        save_result(label,result)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=8)
    parser.add_argument('--ports',type=int,required=True)
    parser.add_argument('--rounds',type=int,default=2)
    args=parser.parse_args()
    run(args.n,args.ports,args.rounds)
