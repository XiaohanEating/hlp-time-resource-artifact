"""Exact scheduling under retained SE barriers and 25-gate atomic port bundles.

All requested gates within a batch use disjoint physical data qubits. Splitting
a batch into ceil(gates/C) serial sublayers is feasible and attains its workload
lower bound when C is a multiple of d**2. No operation crosses an SE barrier.
This is not an optimality claim for asynchronous or differently routed HLP.
"""
from collections import defaultdict
from math import ceil
import time
from .common import OUTPUT,save
from ..capacity_validation.schedule import CapacitySquareBerg
from surface_code import Rotated_surface_code,ShuttleBus_code,QubitType
from gate_types import Pauli,CoreType


def cost(n,q,b,cap):
    widths=[min(b,n-i) for i in range(0,n,b)]
    batches=[min(q,n-i) for i in range(0,n,q)]
    rounds=4*len(widths)*(12+5*(len(batches)-1))
    layers=4*sum(ceil(width*count*25/cap) for width in widths for count in batches)
    return rounds,layers


def physical_batches(n,q):
    code=CapacitySquareBerg(n,n,q)
    core=Rotated_surface_code(5)
    maps={}
    for basis in (Pauli.X,Pauli.Z):
        bus=ShuttleBus_code(5,q,basis)
        for port in range(q):
            maps[basis,port]=[(bp,cp) for bp,cp in bus.dock(core,port).items()
                              if bp in bus.position_2_qubit and bus.position_2_qubit[bp].qubit_type==QubitType.Data]
    batches=defaultdict(list)
    for pos,bus in code.distinct_pos_2_qubit.items():
        if pos[1]<=0:continue
        bid=code.bus_distinct_pos_2_bus_index[pos]
        basis=Pauli.X if bus.core_type==CoreType.XBUS else Pauli.Z
        for t,target in bus.SEspacetime_2_qubit.items():
            pairs=[]
            for bp,cp in maps[basis,t.port]:
                bus_vertex=('bus',bid,*bp)
                core_vertex=('core',*target.distinct_position,*cp)
                pairs.append((bus_vertex,core_vertex) if basis==Pauli.X else (core_vertex,bus_vertex))
            assert len(pairs)==25
            batches[t.SE_group_index,t.inblock_index].append(
                dict(bus=bid,port=t.port,target=target.distinct_position,pairs=pairs))
    for bundles in batches.values():
        occupied=set()
        for bundle in bundles:
            for pair in bundle['pairs']:
                assert not occupied.intersection(pair)
                occupied.update(pair)
    return batches


def certificate(n,q,cap,batches):
    assert cap%25==0
    max_ports=cap//25
    groups=defaultdict(list)
    total=0
    entries=[]
    for (group,batch),bundles in sorted(batches.items()):
        layers=[bundles[i:i+max_ports] for i in range(0,len(bundles),max_ports)]
        lower=ceil(25*len(bundles)/cap)
        assert len(layers)==lower
        groups[group].append(len(layers))
        entries.append(dict(group=group,batch=batch,lower_bound=lower,
                            sublayers=[[dict(bus=x['bus'],port=x['port'],target=x['target'])
                                        for x in layer] for layer in layers]))
        total+=len(layers)
    rounds=sum(12+5*(len(v)-1) for v in groups.values())
    assert (rounds,total)==cost(n,q,n,cap)
    return dict(n=n,ports=q,cap=cap,se_rounds=rounds,cross_gate_layers=total,
                lower_bound_attained=True,batches=entries)


def main():
    started=time.perf_counter()
    n=16;budget=18928
    caps=list(range(25,6401,25))
    sampled=(800,1600,2400,3200,6400)
    summaries=[]
    for q in (4,6,8):
        batches=physical_batches(n,q)
        save(f'n16_q{q}_dependency_batches',dict(
            source='Existing abstract checks plus original physical dock mapping; not a generated n16 full circuit',
            n=n,ports=q,d0=5,
            constraint='Group reset; five SE rounds; each cross batch; five SE rounds; group readout; next group',
            batches=[dict(group=g,batch=b,bundles=v) for (g,b),v in sorted(batches.items())]))
        for cap in sampled:
            proof=certificate(n,q,cap,batches)
            save(f'n16_q{q}_cap{cap}_certificate',proof)
            summaries.append({k:proof[k] for k in ('n','ports','cap','se_rounds','cross_gate_layers','lower_bound_attained')})
    all_points=[];optimal=[]
    for cap in caps:
        candidates=[]
        for q in range(4,n+1):
            for b in range(1,n+1):
                used=n*n*49+b*(50*q-1)
                if used>budget:continue
                rounds,layers=cost(n,q,b,cap)
                candidates.append(dict(ports=q,buses=b,qubits=used,se_rounds=rounds,cross_gate_layers=layers))
        nondominated=[a for a in candidates if not any(
            x['se_rounds']<=a['se_rounds'] and x['cross_gate_layers']<=a['cross_gate_layers']
            and (x['se_rounds'],x['cross_gate_layers'])!=(a['se_rounds'],a['cross_gate_layers'])
            for x in candidates)]
        optimal.append(dict(cap=cap,pareto=nondominated))
        for q in (4,6,8):
            rounds,layers=cost(n,q,n,cap)
            all_points.append(dict(cap=cap,ports=q,se_rounds=rounds,cross_gate_layers=layers))
    save('frontier',dict(scope=__doc__,n=n,d0=5,total_qubit_budget=budget,
        port_domain=list(range(4,n+1)),bus_domain=list(range(1,n+1)),
        time_model='T = R*t_SE + L*t_CX; boundary rounds assumed same duration as SE for normalized illustrations',
        caps=caps,summary=summaries,points=all_points,optimal_uniform=optimal,
        qualifications=['Physical connectivity and control overhead uncalibrated',
                        'No SE/cross-gate overlap or cross-group reordering allowed',
                        'Groups can be split into waves with complete reset/SE/readout per wave',
                        'Optimum over uniform configurations in this legal family only',
                        'Five SE rounds retained; physical wait windows increase by inserted gate durations'],
        elapsed_seconds=time.perf_counter()-started))


if __name__=='__main__':main()
