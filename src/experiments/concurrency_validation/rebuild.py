"""Rebuild ideal full circuits by serializing disjoint cross-block CX layers.

Noise is intentionally omitted: no calibrated duration-dependent idle channel
exists for the new slots. This output is for schedule/ideal equivalence checks
only and is not usable as a reliability benchmark.
"""
import argparse
import hashlib
import time
from collections import Counter
import stim
from .common import ROOT,OUTPUT,digest,save


def is_cross(instruction,coords):
    if instruction.name!='CX':return False
    targets=instruction.targets_copy()
    # Author layout: bus boundary ancillas can have y=-1; cores end at y=-2.
    # Using zero as the boundary would misclassify local bus SE gates.
    flags=[(coords[targets[i].value][1]<-1)!=(coords[targets[i+1].value][1]<-1)
           for i in range(0,len(targets),2)]
    if any(flags) and not all(flags):
        raise ValueError('Mixed local/cross instruction requires explicit classification')
    return bool(flags and all(flags))


def normalized_operations(circuit):
    value=hashlib.sha256()
    for instruction in circuit:
        if isinstance(instruction,stim.CircuitRepeatBlock):raise ValueError('Flatten source before hashing')
        if instruction.name=='TICK':continue
        if instruction.name=='CX':
            ts=instruction.targets_copy()
            for i in range(0,len(ts),2):value.update(f'CX {ts[i].value} {ts[i+1].value}\n'.encode())
        else:value.update((str(instruction)+'\n').encode())
    return value.hexdigest()


def serialize(source,cap):
    # Consecutive CX instructions are not allowed within a common cross layer
    # here: the compiler emits a single combined CX per layer. Enforce it.
    circuit=source.without_noise().flattened()
    coords=circuit.get_final_qubit_coordinates()
    target=stim.Circuit()
    records=[];occupied=set();layer_has_cross=False;tick=0;extra=0
    for instruction in circuit:
        if instruction.name=='TICK':
            occupied.clear();layer_has_cross=False;tick+=1
        if not is_cross(instruction,coords):
            target.append(instruction)
            if instruction.name=='CX':
                used={t.value for t in instruction.targets_copy()}
                assert not occupied.intersection(used)
                occupied.update(used)
            continue
        assert not layer_has_cross and not occupied, 'Cross layer shares local gates or another CX instruction'
        layer_has_cross=True
        ts=instruction.targets_copy()
        assert len({x.value for x in ts})==len(ts)
        partitions=[ts[i:i+2*cap] for i in range(0,len(ts),2*cap)]
        for i,part in enumerate(partitions):
            if i:target.append('TICK',tag='cross_serialization')
            target.append(stim.CircuitInstruction('CX',part,tag=instruction.tag))
        records.append(dict(original_tick=tick,original_gate_count=len(ts)//2,
                            sublayer_gate_counts=[len(x)//2 for x in partitions],
                            added_layers_before=extra))
        extra+=len(partitions)-1
        occupied.update(t.value for t in ts)
    assert normalized_operations(circuit)==normalized_operations(target)
    assert circuit.num_measurements==target.num_measurements
    assert circuit.num_detectors==target.num_detectors
    assert circuit.num_observables==target.num_observables
    # Independently count the output resource usage between ticks.
    used=set();cross_count=0;peak=0
    for inst in target:
        if inst.name=='TICK':
            peak=max(peak,cross_count);cross_count=0;used.clear()
        elif inst.name=='CX':
            ts=inst.targets_copy()
            assert not used.intersection(t.value for t in ts)
            used.update(t.value for t in ts)
            if is_cross(inst,coords):cross_count+=len(ts)//2
    peak=max(peak,cross_count)
    assert peak<=cap
    return target,dict(original_cross_layers=len(records),new_cross_layers=sum(len(x['sublayer_gate_counts']) for x in records),
                       extra_cross_layers=extra,peak_cross_gates=peak,layers=records,
                       ideal_operation_hash=normalized_operations(target),operation_order_unchanged=True,
                       measurement_records_and_detector_definitions_unchanged=True)


def run(n,ports,cap):
    started=time.perf_counter()
    if n==16 and ports==4:
        source=ROOT/'results/circuit_cache/squareberg_n16_pool16_alpha_1_d5_r1.stim'
        rounds=1
    elif n==8 and ports in (4,8):
        source=ROOT/f'results/capacity_validation/n8_q{ports}_d5_r2.stim'
        rounds=2
    else:raise ValueError('No verified full source circuit exists for this case')
    circuit=stim.Circuit.from_file(source)
    target,result=serialize(circuit,cap)
    OUTPUT.mkdir(exist_ok=True,parents=True)
    label=f'n{n}_q{ports}_cap{cap}_r{rounds}'
    path=OUTPUT/(label+'_ideal.stim');pending=path.with_suffix('.stim.pending')
    target.to_file(pending);pending.replace(path)
    target.detector_error_model(allow_gauge_detectors=False)
    ds,obs=target.compile_detector_sampler(seed=41819).sample(32,separate_observables=True)
    assert not ds.any() and not obs.any()
    result.update(n=n,ports=ports,cap=cap,rounds=rounds,source=str(source),source_sha256=digest(source),
                  output=str(path),output_sha256=digest(path),noise='omitted for ideal structural validation only',
                  detectors=target.num_detectors,observables=target.num_observables,
                  deterministic_detectors_and_observables=True,noiseless_shots=32,
                  noiseless_detection_events=int(ds.sum()),noiseless_observable_events=int(obs.sum()),
                  protection='SE round sequence and all measurement boundaries preserved; wall-clock windows gain extra gate layers',
                  elapsed_seconds=time.perf_counter()-started)
    save(label,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=8)
    p.add_argument('--ports',type=int,required=True);p.add_argument('--cap',type=int,required=True)
    a=p.parse_args();run(a.n,a.ports,a.cap)
