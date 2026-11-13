"""Capacity-only experimental schedule, preserving the author's check order."""
from collections import defaultdict
from .common import save_result
from lv1_codes import SquareBerg
from lv1_utils import SESpaceTime


class CapacitySquareBerg(SquareBerg):
    def __init__(self, length, num_buses, ports):
        if not isinstance(ports, int) or not 1 <= ports <= length:
            raise ValueError('Port count must be an integer between 1 and check weight')
        self.capacity = ports
        super().__init__(length, num_buses)

    def _set_SE_schedule(self):
        super()._set_SE_schedule()
        saved = []
        for pos, bus in self.distinct_pos_2_qubit.items():
            if pos[1] <= 0:
                continue
            ordered = sorted(bus.SEspacetime_2_qubit.items(),
                             key=lambda pair: (pair[0].SE_group_index, pair[0].inblock_index, pair[0].port))
            saved.append((bus, ordered[0][0].SE_group_index, [target for _, target in ordered]))
        for qubit in self.distinct_pos_2_qubit.values():
            qubit.SEspacetime_2_qubit.clear()
        occupied_data = set()
        occupied_ports = set()
        for bus, group, targets in saved:
            bus_id = self.bus_distinct_pos_2_bus_index[bus.distinct_position]
            for index, target in enumerate(targets):
                batch, port = divmod(index, self.capacity)
                t = SESpaceTime(group, batch + 1, port)
                dk = (group, batch, target.distinct_position)
                bk = (group, batch, bus_id, port)
                assert dk not in occupied_data and bk not in occupied_ports
                occupied_data.add(dk)
                occupied_ports.add(bk)
                bus.set_SE_qubit(t, target)
                target.set_SE_qubit(t, bus)


def fingerprint(code):
    return {str(pos): sorted((t.SE_group_index, t.inblock_index, t.port, target.distinct_position)
                            for t, target in qubit.SEspacetime_2_qubit.items())
            for pos, qubit in code.distinct_pos_2_qubit.items()}


def main():
    records = []
    for n in (8, 16):
        original = SquareBerg(n, n)
        assert fingerprint(original) == fingerprint(CapacitySquareBerg(n, n, 4))
        for ports in (4, 6, 8):
            code = CapacitySquareBerg(n, n, ports)
            groups = defaultdict(set)
            edges = []
            for pos, qubit in code.distinct_pos_2_qubit.items():
                if pos[1] <= 0:
                    continue
                old = original.distinct_pos_2_qubit[pos]
                assert {t.distinct_position for t in qubit.SEspacetime_2_qubit.values()} == {
                    t.distinct_position for t in old.SEspacetime_2_qubit.values()}
                for time, target in qubit.SEspacetime_2_qubit.items():
                    groups[time.SE_group_index].add(time.inblock_index)
                    edges.append(dict(check=pos, bus=code.bus_distinct_pos_2_bus_index[pos],
                                      group=time.SE_group_index, batch=time.inblock_index,
                                      port=time.port, target=target.distinct_position))
            records.append(dict(n=n, buses=n, ports=ports, checks_preserved=True,
                                batches_per_group={g: len(v) for g, v in groups.items()},
                                conditional_rounds=sum(12 + 5*(len(v)-1) for v in groups.values()),
                                abstract_resource_conflicts=0, edges=edges))
    save_result('schedules', dict(scope='Abstract check-preserving schedules; no physical routing certification',
                                 records=records))


if __name__ == '__main__':
    main()
