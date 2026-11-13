from __future__ import annotations
from typing import Optional, Iterable
from dataclasses import dataclass, field
from enum import Enum

from gate_types import Gate, ACTIVATE_INSTRUCTIONS, DEACTIVATE_INSTRUCTIONS,\
                       Pauli


@dataclass
class SESpaceTime:
    SE_group_index: int 
    inblock_index: int
    port: int

    def __hash__(self):
        return hash((self.SE_group_index, self.inblock_index, self.port))
    
    def __eq__(self, value: SESpaceTime) -> bool:
        if self.SE_group_index == value.SE_group_index \
            and self.inblock_index == value.inblock_index \
            and self.port == value.port:
            return True
        else:
            return False

class Lv1LogicalRESET(Enum):
    BELL_RESET = 'BELL_RESET'
    X_RESET = 'X_RESET'
    Z_RESET = 'Z_RESET'
    Y_RESET = 'Y_RESET'
    UNDEFINED = 'UNDEFINED'
    
@dataclass
class Lv1Time:
    lv1_cycle_time: int
    clock_time: int

    def __hash__(self):
        return hash((self.lv1_cycle_time, self.clock_time))

    def __eq__(self, value: Lv1Time) -> bool:
        if self.lv1_cycle_time == value.lv1_cycle_time \
           and self.clock_time == value.clock_time:
            return True
        else:
            return False
    
    def __lt__(self, value: Lv1Time) -> bool:
        if self.lv1_cycle_time <= value.lv1_cycle_time \
           and self.clock_time < value.clock_time:
            return True
        else:
            return False


@dataclass(order=True)
class Lv1ID_port:
    patch_id: int
    port: int

    def __hash__(self):
        return hash((self.patch_id,self.port))

    def __eq__(self, other: Lv1ID_port):
        if self.patch_id == other.patch_id \
           and self.port == other.port:
            return True
        else:
            return False

@dataclass
class Lv1_Op:
    gate: Gate
    targets: tuple[Lv1ID_port,...] 

    def __hash__(self):
        targets_full_tuple = tuple([(target.patch_id, target.port) 
                                    for target in self.targets])
        return hash((self.gate,) + targets_full_tuple)
    
    def __eq__(self, other: Lv1_Op):
        if self.gate == other.gate \
            and self.targets == other.targets:
            return True
        else:
            return False
    
    def __lt__(self, other: Lv1_Op):
        if self.gate < other.gate:
            return True
        if other.gate < self.gate:
            return False
        assert self.gate == other.gate
        for index, port in enumerate(self.targets):
            if index >= len(other.targets):
                return False
            if port < other.targets[index]:
                return True
            if other.targets[index] < port:
                return False
        return False
        
    @property
    def lv1_targets(self) -> list[int]:
        if self.gate in ACTIVATE_INSTRUCTIONS + DEACTIVATE_INSTRUCTIONS:
            ids = set([id_port.patch_id for id_port in self.targets])
            return sorted(ids)
        elif self.gate in [Gate.CX, Gate.CZ]:
            assert len(self.targets) == 2
            ids = [self.targets[0].patch_id, self.targets[1].patch_id]
            return ids
        else:
            raise NotImplementedError(f'{self.gate=} not supported at level-1.')



@dataclass
class Lv1Timeline:
    patch_id: int
    time_2_ops: dict[Lv1Time, list[Lv1_Op]] = field(default_factory=dict)
    pass

    @property
    def sorted_time(self) -> list[Lv1Time]:
        return sorted(self.time_2_ops.keys())

    @property
    def most_recent_activation_time(self) -> Optional[Lv1Time]:
        for time in reversed(self.sorted_time):
            ops = self.time_2_ops[time]
            for op in ops:
                if op.gate in ACTIVATE_INSTRUCTIONS:
                    return time
        return None
    
    @property
    def most_recent_activation_time_and_gate(self) -> Optional[tuple[Lv1Time,Gate]]:
        for time in reversed(self.sorted_time):
            ops = self.time_2_ops[time]
            for op in ops:
                if op.gate in ACTIVATE_INSTRUCTIONS:
                    return (time, op.gate)
        return None

    @property
    def most_recent_deactivation_time(self) -> Optional[Lv1Time]:
        for time in reversed(self.sorted_time):
            ops = self.time_2_ops[time]
            for op in ops:
                if op.gate in DEACTIVATE_INSTRUCTIONS:
                    return time
        return None

    @property
    def is_activated(self) -> bool:
        if self.most_recent_activation_time is None:
            return False
        if self.most_recent_deactivation_time is None:
            return True
        if self.most_recent_deactivation_time < self.most_recent_activation_time:
            return True
        return False

    def add_new_event(self, op: Lv1_Op, time: Lv1Time):
        relevance = False
        for target_space in op.targets:
            if target_space.patch_id == self.patch_id:
                relevance = True
        if relevance is False:
            raise ValueError(f'This {self.patch_id=} is not participated in {op=}')
        if time in self.time_2_ops:
            existing_ops = self.time_2_ops[time]
            # test for port collisions
            for existing_op in existing_ops:
                if existing_op.gate in [Gate.MPP_MX, Gate.MPP_MZ, 
                                        Gate.MPP_RX, Gate.MPP_RZ,
                                        Gate.MPP_RESET, Gate.MPP_MEAS]:
                    continue
                for used_port in existing_op.targets:
                    if used_port in op.targets:
                        raise ValueError(
                            f'Trying to add another {op=} at a {used_port=}. '
                            f'Existing op at the used port is {existing_op}.'
                        )
            self.time_2_ops[time].append(op)
        else:
            self.time_2_ops[time] = [op]

    def find_the_next_same_operation(self, time: Lv1Time, current_op: Lv1_Op, 
                                     exclude_targets: list[int] = []) -> tuple[Lv1Time,bool]:
        assert self.patch_id in current_op.lv1_targets
        for t in self.sorted_time:
            if t < time or t == time:
                continue
            for op in self.time_2_ops[t]:
                if op.gate != current_op.gate:
                    continue
                if current_op.gate not in [Gate.CX, Gate.CZ]:
                    return t, True
                if self.patch_id != current_op.lv1_targets[0]:
                    assert self.patch_id == current_op.lv1_targets[1]
                    if op.lv1_targets[0] in exclude_targets:
                        continue
                    if self.patch_id == op.lv1_targets[1]:
                        return t, True
                elif self.patch_id == op.lv1_targets[0]:
                    if op.lv1_targets[1] in exclude_targets:
                        continue
                    else:
                        return t, True
        return self.sorted_time[-1], False

    def find_the_previous_same_operation(self, time: Lv1Time, current_op: Lv1_Op,
                                         exclude_targets: list[int] = []) -> tuple[Lv1Time,bool]:
        assert self.patch_id in current_op.lv1_targets
        for t in reversed(self.sorted_time):
            if t == time or t > time:
                continue
            for op in self.time_2_ops[t]:
                if op.gate != current_op.gate:
                    continue
                if current_op.gate not in [Gate.CX, Gate.CZ]:
                    return t, True
                if self.patch_id != current_op.lv1_targets[0]:
                    assert self.patch_id == current_op.lv1_targets[1]
                    if op.lv1_targets[0] in exclude_targets:
                        continue
                    if self.patch_id == op.lv1_targets[1]:
                        return t, True
                elif self.patch_id == op.lv1_targets[0]:
                    if op.lv1_targets[1] in exclude_targets:
                        continue
                    else:
                        return t, True            
        return self.sorted_time[0], False

    def find_the_previous_activation_time_and_gate(self, time: Lv1Time) -> tuple[Lv1Time,Gate]:
        if not (self.sorted_time[0] < time):
            raise RuntimeError(f'{time=} is already prior to or at the start of this timeline')
        for t in reversed(self.sorted_time):
            if t == time or t > time:
                continue
            for op in self.time_2_ops[t]:
                if op.gate in ACTIVATE_INSTRUCTIONS:
                    return t, op.gate
        raise RuntimeError(f'No activation instruction in the timeline.')

    def find_the_next_deactivation_time_and_gate(self, time: Lv1Time) -> tuple[Lv1Time,Gate]:
        if not (self.sorted_time[-1] > time):
            raise RuntimeError(f'{time=} is already after to or at the end of this timeline')
        for t in self.sorted_time:
            if t < time or t == time:
                continue
            for op in self.time_2_ops[t]:
                if op.gate in DEACTIVATE_INSTRUCTIONS:
                    return t, op.gate
        raise RuntimeError(f'No deactivation instruction in the timeline.')

@dataclass
class Lv1Det:
    '''
    Track the level-1 detecting/observable region. 
    '''
    id: int
    supp: set[tuple[int,Pauli]] = field(default_factory=set)
    meas_ids: set[int] = field(default_factory=set)
    anc_patch_id: Optional[int] = None
    pos: Optional[tuple[int,...]] = None
    lv1_cycle_time: Optional[int] = None
    clock_time: Optional[int] = None
    involved_lv1_meas: list[tuple[Lv1Time,Lv1_Op]] = field(default_factory=list)
    unpaired: bool = False

    def supp_update(self, new_supp: Iterable[tuple[int,Pauli]]):
        for target in new_supp:
            if target[-1] not in [Pauli.X, Pauli.Z]:
                raise TypeError(f'{new_supp=} should only contain X or Z terms')
        self.supp.symmetric_difference_update(new_supp)
    
    def meas_update(self, new_meas: Iterable[int]):
        self.meas_ids.symmetric_difference_update(new_meas)

    def add_lv1_meas(self, lv1_time: Lv1Time, lv1_meas: Lv1_Op):
        if lv1_meas.gate not in (DEACTIVATE_INSTRUCTIONS 
                                 + [Gate.MPP_RX, Gate.MPP_RZ]):
            raise TypeError(f'{lv1_meas.gate=} not an lv-1 measurement')
        self.involved_lv1_meas.append((lv1_time, lv1_meas))

    @property
    def is_active(self) -> bool:
        if len(self.supp) == 0 and self.involved_lv1_meas:
            return False
        elif len(self.supp) == 0:
            raise RuntimeError(f'Lv1Det {self} not properly initialized.')
        else:
            return True
    
    @property
    def latest_meas_time(self) -> Optional[Lv1Time]:
        involved_times = []
        for time, meas_op in self.involved_lv1_meas:
            involved_times.append(time)
        if involved_times:
            return sorted(involved_times)[-1]
        return None



@dataclass
class Lv1Obs:
    '''
    Track the level-1 detecting/observable region. 
    '''
    obs_id: int
    supp: set[tuple[int,Pauli]] = field(default_factory=set)
    meas_ids_buff: set[int] = field(default_factory=set)
    meas_ids_history: set[int] = field(default_factory=set)
    logical_id: Optional[int] = None
    logical_basis: Optional[Pauli] = None
    lv1_cycle_time: Optional[int] = None
    clock_time: Optional[int] = None
    involved_lv1_meas: list[tuple[Lv1Time,Lv1_Op]] = field(default_factory=list)
    lv1_supp: set[tuple[int,Pauli]] = field(default_factory=set)

    def supp_update(self, new_supp: Iterable[tuple[int,Pauli]]):
        for target in new_supp:
            if target[-1] not in [Pauli.X, Pauli.Z]:
                raise TypeError(f'{new_supp=} should only contain X or Z terms')
        self.supp.symmetric_difference_update(new_supp)

    def lv1_supp_update(self, new_supp: Iterable[tuple[int,Pauli]]):
        for target in new_supp:
            if target[-1] not in [Pauli.X, Pauli.Z]:
                raise TypeError(f'{new_supp=} should only contain X or Z terms')
        self.lv1_supp.symmetric_difference_update(new_supp)
    
    def meas_buff_update(self, new_meas: Iterable[int]):
        self.meas_ids_buff.symmetric_difference_update(new_meas)

    def meas_history_update(self):
        self.meas_ids_history.symmetric_difference_update(self.meas_ids_buff)

    def clear_meas_buff(self):
        self.meas_ids_buff.clear()

    def add_lv1_meas(self, lv1_time: Lv1Time, lv1_meas: Lv1_Op):
        if lv1_meas.gate not in (DEACTIVATE_INSTRUCTIONS 
                                 + [Gate.MPP_RX, Gate.MPP_RZ]):
            raise TypeError(f'{lv1_meas.gate=} not an lv-1 measurement')
        self.involved_lv1_meas.append((lv1_time, lv1_meas))

    @property
    def is_active(self) -> bool:
        if len(self.supp) == 0 and self.involved_lv1_meas:
            return False
        elif len(self.supp) == 0:
            raise RuntimeError(f'Lv1Det {self} not properly initialized.')
        else:
            return True
    
    @property
    def earliest_meas_time(self) -> Optional[Lv1Time]:
        involved_times = []
        for time, meas_op in self.involved_lv1_meas:
            involved_times.append(time)
        if involved_times:
            return sorted(involved_times)[0]
        return None
