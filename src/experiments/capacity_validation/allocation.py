"""Exact capacity-allocation lower model with arbitrary check-to-bus assignment.

All connections are available. Four check groups execute in sequence; within
each group a bus may serially process several identical checks. Neither
physical routing nor circuit reliability is represented. Dynamic programming
certifies the optimum of this stated model, not unrestricted HLP scheduling.
"""
import math
import time
from collections import Counter
from .common import save_result


def costs_at_deadline(n, deadline, min_ports=1):
    duration = {q: 12 + 5 * (math.ceil(n/q)-1) for q in range(min_ports, n+1)}
    capacity = {q: min(n, deadline//t) for q, t in duration.items() if deadline >= t}
    infinity = 10**12
    dp, allocations = [0] + [infinity]*n, [[] for _ in range(n+1)]
    for jobs in range(1, n+1):
        for q, count in capacity.items():
            prev = max(0, jobs-count)
            candidate = dp[prev] + 50*q-1
            if candidate < dp[jobs]:
                dp[jobs] = candidate
                allocations[jobs] = allocations[prev] + [q]
    uniform = min((math.ceil(n/count)*(50*q-1), q, math.ceil(n/count))
                  for q, count in capacity.items())
    assert sum(capacity[q] for q in allocations[n]) >= n
    assert sum(50*q-1 for q in allocations[n]) == dp[n]
    return dict(group_deadline=deadline, joint_min_bus_qubits=dp[n],
                joint_ports=allocations[n], uniform_min_bus_qubits=uniform[0],
                uniform_ports=uniform[1], uniform_bus_count=uniform[2])


def main():
    started = time.perf_counter()
    results = []
    for n in (8, 16):
        # Square Berg has level-1 distance four. Preserve the author's
        # bus-length protection scale instead of admitting shorter buses.
        min_ports = 4
        durations = {12 + 5*(math.ceil(n/q)-1) for q in range(min_ports, n+1)}
        deadlines = sorted({k*t for t in durations for k in range(1, n+1)})
        frontier = [costs_at_deadline(n, d, min_ports) for d in deadlines]
        anchors = []
        for budget in (n*199, 2*n*199, n*399):
            j = next(row for row in frontier if row['joint_min_bus_qubits'] <= budget)
            u = next(row for row in frontier if row['uniform_min_bus_qubits'] <= budget)
            anchors.append(dict(bus_qubit_budget=budget, total_qubit_budget=n*n*49+budget,
                                joint_cycle_rounds=4*j['group_deadline'],
                                uniform_cycle_rounds=4*u['group_deadline'],
                                joint_allocation=dict(Counter(j['joint_ports'])),
                                uniform_ports=u['uniform_ports'],
                                uniform_bus_count=u['uniform_bus_count']))
        # Check all integer budgets in the original-to-eight-port range using
        # only breakpoints: optima cannot change between minimum-cost events.
        budgets = sorted({n*199, n*399} | {
            row[key] for row in frontier for key in ('joint_min_bus_qubits','uniform_min_bus_qubits')
            if n*199 <= row[key] <= n*399})
        differences = []
        for budget in budgets:
            j = next(row for row in frontier if row['joint_min_bus_qubits'] <= budget)
            u = next(row for row in frontier if row['uniform_min_bus_qubits'] <= budget)
            if j['group_deadline'] < u['group_deadline']:
                differences.append(dict(budget=budget, joint=j['group_deadline'],uniform=u['group_deadline']))
        results.append(dict(n=n, allowed_ports=list(range(min_ports,n+1)), anchors=anchors,
                            original_to_eight_port_budget_breakpoints=budgets,
                            strict_advantages_at_those_breakpoints=differences,
                            full_deadline_cost_table=frontier))
    save_result('allocation',dict(scope=__doc__,results=results,
                                 elapsed_seconds=time.perf_counter()-started))


if __name__ == '__main__':
    main()
