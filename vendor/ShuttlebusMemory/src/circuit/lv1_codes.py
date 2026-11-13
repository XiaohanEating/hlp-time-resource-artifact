from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from surface_code import QubitType
from lv1_utils import SESpaceTime
from high_level_utils import CodePatch
from gate_types import CoreType, CoreOp, Pauli, Lv1ObsOp


class Lv1Qubit:

    def __init__(self, position: tuple[int,...], core_type: CoreType,
                 distinct_position: tuple[int,...]):
        self.position = position    # placed in circuits
        # self.qubit_type = qubit_type
        self.core_type = core_type
        self.SEspacetime_2_qubit: dict[SESpaceTime,Lv1Qubit] = {}
        self.code_patch: Optional[CodePatch] = None
        self.distinct_position: tuple[int,...] = distinct_position  # internal to the lv-1 code
        self.virtual_patch_id: Optional[int] = None
        pass

    def set_SE_qubit(self, SEspacetime: SESpaceTime, qubit: Lv1Qubit):
        if SEspacetime in self.SEspacetime_2_qubit:
            raise ValueError(f'Logical qubit with {SEspacetime=} is already set')
        self.SEspacetime_2_qubit[SEspacetime] = qubit
    
    @property
    def patch_id(self) -> int:
        if self.code_patch is not None:
            return self.code_patch.patch_id
        if self.virtual_patch_id is None:
            raise RuntimeError(f'Both `code_patch` and `virtual_patch_id` are unset. '
                               f'Failed to extract the patch id.')
        return self.virtual_patch_id
    
    @property
    def min_SE_group_index(self) -> int:
        group_indices = set()
        for SEspacetime in self.SEspacetime_2_qubit:
            group_indices.add(SEspacetime.SE_group_index)
        if len(group_indices) == 0:
            raise ValueError(f'{self.SEspacetime_2_qubit=} is empty')
        return min(group_indices)
    
    def check_SE_at_time(self, SE_group_index: int, 
                         inblock_index: int) -> Optional[SESpaceTime]:
        for SEspacetime in self.SEspacetime_2_qubit:
            if SEspacetime.SE_group_index == SE_group_index \
               and SEspacetime.inblock_index == inblock_index:
                return SEspacetime
        return None
    
    @property
    def simple_info(self) -> str:
        info = f'Lv1Qubit[distinct position: {self.distinct_position} '\
               f'core type: {self.core_type}]'
        return info
    
    def __repr__(self) -> str:
        info = f'Lv1Qubit[distinct position: {self.distinct_position} '\
               f'core type: {self.core_type}'
        if len(self.SEspacetime_2_qubit) != 0:
            info = info + '\n\t' + 'SE instructions:\n'
            for se_space_time in self.SEspacetime_2_qubit:
                info = info + f'\tSE spacetime: {se_space_time} -- '\
                       f'Qubit: {self.SEspacetime_2_qubit[se_space_time].simple_info}\n'
        info += ']\n'
        return info
    
    @property
    def occupied_spacetime(self) -> set[SESpaceTime]:
        return set(self.SEspacetime_2_qubit.keys())
    

@dataclass
class Lv1LogicalQubit:
    id: Optional[int] = None
    x_supp: list[tuple[int,...]] = field(default_factory=list)
    z_supp: list[tuple[int,...]] = field(default_factory=list)
    pos: Optional[tuple[int,...]] = None
    circuit_id: Optional[int] = None
    basis_2_obs_id: dict[Pauli,int] = field(default_factory=dict)
    
@dataclass(frozen=True)
class CoreInstruction:
    core_operation: CoreOp = CoreOp.UNDEFINED
    target_type: CoreType = CoreType.UNDEFINED
    target_pos: Optional[tuple[int,int]] = None
    target_port: int = 0
    ctrl_type: CoreType = CoreType.UNDEFINED
    ctrl_pos: Optional[tuple[int,int]] = None
    ctrl_port: int = 0

@dataclass(frozen=True)
class ObsInstruction:
    obs_operation: Lv1ObsOp
    target_type: CoreType
    target_pos: tuple[int,int]
    basis: Pauli
    internal_logical_qubit_ids: list[int] = field(default_factory=list)
    external_logical_qubit_ids: list[int] = field(default_factory=list)

@dataclass(order=True,frozen=True)
class CoreInstTime:
    lv1_time: int
    SE_group_index: int
    inblock_index: int
    clock_offset: int = 0

class ExtCores:
    '''
    Additional Cores and ShuttleBuses for logical processing.
    '''
    def __init__(self, internal_core_size: int, bus_length: int,
                 external_core_size: int):
        self.internal_core_size = internal_core_size
        self.bus_length = bus_length
        self.external_core_size = external_core_size
        self.external_cores: list[Lv1Qubit] = []
        self.pos_2_core: dict[tuple[int,int],Lv1Qubit] = {}
        self.pos_2_index: dict[tuple[int,int],int] = {}
        self.external_logical_qubits: list[Lv1LogicalQubit] = []
        self.pos_2_external_logical_qubit_index: dict[tuple[int,int],int] = {}
        self.time_2_instructions: dict[CoreInstTime,list[CoreInstruction]] = {}
        self.time_2_obs_instructions: dict[CoreInstTime,list[ObsInstruction]] = {}
        pass

    def add_core(self, pos: tuple[int,int], core_type: CoreType):
        if pos in self.pos_2_core:
            raise ValueError(f'core with {pos=} already added.')
        new_core = Lv1Qubit(pos, core_type, pos)
        self.external_cores.append(new_core)
        self.pos_2_core[pos] = new_core
        self.pos_2_index[pos] = len(self.external_cores) - 1
    
    def auto_assign_logical_qubits(self):
        index = 0
        for external_core in self.external_cores:
            if external_core.core_type != CoreType.EXTCORE:
                continue
            logical_qubit = Lv1LogicalQubit( 
                                x_supp = [external_core.distinct_position],
                                z_supp = [external_core.distinct_position])
            self.external_logical_qubits.append(logical_qubit)
            self.pos_2_external_logical_qubit_index[external_core.position] = index
            index += 1
        
    def add_instruction(self, core_inst: CoreInstruction, time: CoreInstTime):
        if time not in self.time_2_instructions:
            self.time_2_instructions[time] = [core_inst]
        else:
            self.time_2_instructions[time].append(core_inst)
    
    def add_obs_instruction(self, obs_inst: ObsInstruction, time: CoreInstTime):
        if time not in self.time_2_obs_instructions:
            self.time_2_obs_instructions[time] = [obs_inst]
        else:
            self.time_2_obs_instructions[time].append(obs_inst)

    def coord_span_of_core_types(self, coord_index, core_types: list[CoreType]) -> tuple[Optional[int],Optional[int]]:
        coords = set()
        for core in self.external_cores:
            if core.core_type not in core_types:
                continue
            assert (len(core.position) > coord_index)
            coords.add(core.position[coord_index])
        if len(coords) == 0:
            return None, None
        return min(coords), max(coords)

class Lv1CSSCode(ABC):

    def __init__(self):
        self.distinct_pos_2_qubit: dict[tuple[int,...],Lv1Qubit] = {}
        self.time_unit: Optional[float] = None
        self.logical_qubits: list[Lv1LogicalQubit] = []
        self.se_start_padding_time: Optional[float] = None
        self.se_end_padding_time: Optional[float] = None
        self._set_qubits()
        self._set_SE_schedule()
        self._verify_SE()
        self._set_logical_qubits()
        pass

    @abstractmethod
    def _set_logical_qubits(self):
        pass

    @abstractmethod
    def _set_qubits(self):
        pass

    @abstractmethod
    def _set_SE_schedule(self):
        '''
        It is assumed that at the same `SE_group_index`, only lv-1 X ancilla
        or lv-1 Z ancilla are present. 
        '''
        pass

    def _verify_SE(self):
        '''
        1. No empty group
        2. Each SE group have the same ancilla type
        3. Each logical ancilla appear in at most one SE group
        '''
        for SE_group_index in self.SE_group_index_2_anc_qubits:
            anc_qubits = list(self.SE_group_index_2_anc_qubits[SE_group_index])
            if len(anc_qubits) == 0:
                raise ValueError(f'empty lv-1 instruction at {SE_group_index=}')
            # anc_type = anc_qubits[0].core_type
            # for anc in anc_qubits:
            #     if anc.core_type != anc_type:
            #         raise TypeError(f'{anc.core_type=} does not match the '
            #                         f'first ancilla type {anc_type} '
            #                         f'of this {SE_group_index=}')
        for qubit in self.distinct_pos_2_qubit.values():
            if qubit.core_type == CoreType.INTCORE:
                continue
            se_group_index = set()
            for se_spacetime in qubit.SEspacetime_2_qubit:
                se_group_index.add(se_spacetime.SE_group_index)
            if len(se_group_index) > 1:
                raise ValueError(f'logical ancilla at {qubit.distinct_position}'
                                 f'is involved in se groups {se_group_index}')

    @property
    def data_qubits(self) -> list[Lv1Qubit]:
        data_internal_cores = []
        for lv1_qubit in self.distinct_pos_2_qubit.values():
            if lv1_qubit.core_type == CoreType.INTCORE:
                data_internal_cores.append(lv1_qubit)
        return data_internal_cores
    
    @property
    def all_qubits(self) -> list[Lv1Qubit]:
        all_qubits = []
        for lv1_qubit in self.distinct_pos_2_qubit.values():
            assert lv1_qubit.core_type in [CoreType.INTCORE,
                                           CoreType.XBUS,
                                           CoreType.ZBUS]
            all_qubits.append(lv1_qubit)
        return all_qubits
    
    @property
    def anc_qubits(self) -> list[Lv1Qubit]:
        anc_buses = []
        for lv1_qubit in self.distinct_pos_2_qubit.values():
            if lv1_qubit.core_type in [CoreType.XBUS, CoreType.ZBUS]:
                anc_buses.append(lv1_qubit)
        return anc_buses

    @property
    def SE_group_index_2_anc_qubits(self) -> dict[int,set[Lv1Qubit]]:
        se_index_2_unordered_anc_qubits: dict[int,set[Lv1Qubit]] = {}
        for lqubit in self.data_qubits:
            for spacetime in lqubit.SEspacetime_2_qubit:
                if spacetime.SE_group_index in se_index_2_unordered_anc_qubits:
                    se_index_2_unordered_anc_qubits[spacetime.SE_group_index].add(
                        lqubit.SEspacetime_2_qubit[spacetime]
                    )
                else:
                    se_index_2_unordered_anc_qubits[spacetime.SE_group_index] = set([
                        lqubit.SEspacetime_2_qubit[spacetime]
                    ])
        ret_ordered: dict[int,list[Lv1Qubit]] = {}
        for se_index in se_index_2_unordered_anc_qubits:
            ret_ordered[se_index] = sorted(se_index_2_unordered_anc_qubits[se_index],
                                           key = lambda anc : anc.distinct_position)
        return ret_ordered
    
    @property
    def SE_group_index_2_inblock_indices(self) -> dict[int,set[int]]:
        ret_dict: dict[int,set[int]] = {}
        for lqubit in self.data_qubits:
            for spacetime in lqubit.SEspacetime_2_qubit:
                if spacetime.SE_group_index in ret_dict:
                    ret_dict[spacetime.SE_group_index].add(
                        spacetime.inblock_index
                    )
                else:
                    ret_dict[spacetime.SE_group_index] = set([
                        spacetime.inblock_index
                    ])
        return ret_dict
    
    def distinct_position_coord_span(self, coord_index: int) -> tuple[int,int]:
        coords = set()
        for pos in self.distinct_pos_2_qubit:
            assert(len(pos) > coord_index)
            coords.add(pos[coord_index])
        return min(coords), max(coords)
    
    def reset_logical_qubits(self, logical_index_2_x_supp_indices: dict[int,list[int]],
                            logical_index_2_z_supp_indices: dict[int,list[int]]):
        raise NotImplementedError('Reset method is undefined.')
    
    @property
    def stabilizers(self) -> list[set[tuple[tuple[int,int],Pauli]]]:
        stabs: list[set[tuple[tuple[int,int],Pauli]]] = []
        for anc in self.anc_qubits:
            stab: set[tuple[tuple[int,int],Pauli]] = set()
            if anc.core_type == CoreType.ZBUS:
                for qubit in anc.SEspacetime_2_qubit.values():
                    stab.add((qubit.distinct_position,Pauli.Z))
            else:
                for qubit in anc.SEspacetime_2_qubit.values():
                    stab.add((qubit.distinct_position,Pauli.X))
            stabs.append(stab)
        return stabs

    def logical_x(self, logical_qubit_id: int) -> set[tuple[tuple[int,int],Pauli]]:
        logic_x: set[tuple[tuple[int,int],Pauli]] = set()
        logical_qubit = self.logical_qubits[logical_qubit_id]
        for pos in logical_qubit.x_supp:
            logic_x.add((pos, Pauli.X))
        return logic_x

    def logical_z(self, logical_qubit_id: int) -> set[tuple[tuple[int,int],Pauli]]:
        logic_z: set[tuple[tuple[int,int],Pauli]] = set()
        logical_qubit = self.logical_qubits[logical_qubit_id]
        for pos in logical_qubit.z_supp:
            logic_z.add((pos, Pauli.Z))
        return logic_z

    def _self_test(self) -> bool:
        def is_commute(stab_1: set[tuple[tuple[int,int],Pauli]],
                    stab_2: set[tuple[tuple[int,int],Pauli]]) -> bool:
            all_basis: list[Pauli] = [Pauli.X, Pauli.Y, Pauli.Z]
            comm = 0
            for op in stab_1:
                for pauli in all_basis:
                    if op[-1] == pauli:
                        continue
                    if (op[0],pauli) in stab_2:
                        comm ^= 1
            if comm == 0:
                return True
            else:
                return False

        # stabilizer commutativity
        stabilizers = self.stabilizers
        for i in range(len(stabilizers)):
            for j in range(len(stabilizers)):
                if i <= j:
                    continue
                if not is_commute(stabilizers[i],stabilizers[j]):
                    raise ValueError(f'Stabilizer {stabilizers[i]} does not commute with '
                                     f'Stabilizer {stabilizers[j]}')
        # verify logical operators commute with stabilizers
        for i in range(len(self.logical_qubits)):
            for stab in stabilizers:
                if not is_commute(self.logical_x(i), stab):
                    raise ValueError(f'Logical X operator {i}: {self.logical_x(i)} '
                                     f'does not commute with Stabilizer {stab}')
                if not is_commute(self.logical_z(i), stab):
                    raise ValueError(f'Logical Z operator {i}: {self.logical_z(i)} '
                                     f'does not commute with Stabilizer {stab}')
        # verify the independence of all logical operators 
        for i in range(len(self.logical_qubits)):
            for j in range(len(self.logical_qubits)):
                if i < j:
                    continue
                if i == j:
                    if is_commute(self.logical_x(i), self.logical_z(i)):
                        raise ValueError(f'The logical X operator {self.logical_x(i)} does not '
                                         f'anticommute with the logical Z operator {self.logical_z(i)} '
                                         f'on logical qubit {i}')
                    continue
                if not is_commute(self.logical_x(i), self.logical_z(j)):
                    raise ValueError(f'The logical X operator of the logical qubit {i}: {self.logical_x(i)} '
                                     f'does not commute with the logical Z operator of the '
                                     f'logical qubit {j}: {self.logical_z(j)}')
                if not is_commute(self.logical_z(i), self.logical_x(j)):
                    raise ValueError(f'The logical Z operator of the logical qubit {i}: {self.logical_z(i)} '
                                     f'does not commute with the logical X operator of the '
                                     f'logical qubit {j}: {self.logical_x(j)}')
        return True


class IceBerg(Lv1CSSCode):

    def __init__(self, size: int, time_unit = 1/2):
        if size % 2 != 0:
            raise ValueError(f'{size=} of an iceburg code should be an even number')
        self.size = size
        super().__init__()
        self.time_unit = time_unit
        self.se_start_padding_time = time_unit
        self.se_end_padding_time = time_unit
        
    def _set_logical_qubits(self):
        first_pos = self._data_index_2_distinct_position(0)
        final_pos = self._data_index_2_distinct_position(self.size-1)
        for i in range(self.size-2):
            # X_0*X_(i+1)  Z_(i+1)*Z_n  for 0 <= i <= n-3
            logical_qubit = Lv1LogicalQubit(
                id=i,
                x_supp=[self._data_index_2_distinct_position(0),
                        self._data_index_2_distinct_position(i+1)],
                z_supp=[self._data_index_2_distinct_position(i+1),
                        self._data_index_2_distinct_position(self.size-1)],
            )
            self.logical_qubits.append(logical_qubit)
    
    def reset_logical_qubits(self, logical_index_2_x_supp_indices: dict[int,list[int]],
                            logical_index_2_z_supp_indices: dict[int,list[int]]):
        assert logical_index_2_x_supp_indices.keys() == logical_index_2_z_supp_indices.keys()
        for logical_index in logical_index_2_x_supp_indices:
            x_supp_indices = logical_index_2_x_supp_indices[logical_index]
            z_supp_indices = logical_index_2_z_supp_indices[logical_index]
            logical_qubit = self.logical_qubits[logical_index]
            logical_qubit.x_supp = [self._data_index_2_distinct_position(i) for i in x_supp_indices]
            logical_qubit.z_supp = [self._data_index_2_distinct_position(i) for i in z_supp_indices]
        self._self_test()

    def _data_index_2_distinct_position(self, data_index: int) -> tuple[int,int]:
        return (data_index%4, -1-(data_index//4))

    def _set_qubits(self):
        # ancilla patches at bottom layer
        self.distinct_pos_2_qubit[(0,0)] = Lv1Qubit(
            (0,0), CoreType.ZBUS, (0,0)
        )
        self.distinct_pos_2_qubit[(1,0)] = Lv1Qubit(
            (0,0), CoreType.XBUS, (1,0)
        )
        # data patches are arranged into rows of 4
        for i in range(self.size):
            position = self._data_index_2_distinct_position(i)
            self.distinct_pos_2_qubit[position] = Lv1Qubit(
                position, CoreType.INTCORE, position
            )

    def _set_SE_schedule(self):
        for i in range(self.size):
            position = self._data_index_2_distinct_position(i)
            data = self.distinct_pos_2_qubit[position]
            z_anc = self.distinct_pos_2_qubit[(0,0)]
            x_anc = self.distinct_pos_2_qubit[(1,0)]
            data.set_SE_qubit(
                SESpaceTime(0, 1+(i//2), i%2),
                z_anc
            )
            z_anc.set_SE_qubit(
                SESpaceTime(0, 1+(i//2), i%2),
                data
            )
            data.set_SE_qubit(
                SESpaceTime(1, 1+(i//2), i%2),
                x_anc
            )
            x_anc.set_SE_qubit(
                SESpaceTime(1, 1+(i//2), i%2),
                data
            )
        pass



class IceBergCompact(Lv1CSSCode):

    def __init__(self, size: int, time_unit = 1/2):
        if size % 4 != 0:
            raise ValueError(f'{size=} of an iceburg code (run in compact mode) should be devidable by 4')
        self.size = size
        super().__init__()
        self.time_unit = time_unit
        self.se_start_padding_time = time_unit
        self.se_end_padding_time = time_unit
        self._self_test()
        
    def _set_logical_qubits(self):
        first_pos = self._data_index_2_distinct_position(0)
        final_pos = self._data_index_2_distinct_position(self.size-1)
        for i in range(self.size-2):
            # X_0*X_(i+1)  Z_(i+1)*Z_n  for 0 <= i <= n-3
            logical_qubit = Lv1LogicalQubit(
                id=i,
                x_supp=[self._data_index_2_distinct_position(0),
                        self._data_index_2_distinct_position(i+1)],
                z_supp=[self._data_index_2_distinct_position(i+1),
                        self._data_index_2_distinct_position(self.size-1)],
            )
            self.logical_qubits.append(logical_qubit)

    def _data_index_2_distinct_position(self, data_index: int) -> tuple[int,int]:
        return (data_index%4, -1-(data_index//4))

    def _set_qubits(self):
        # ancilla patches at bottom layer
        self.distinct_pos_2_qubit[(0,0)] = Lv1Qubit(
            (0,0), CoreType.ZBUS, (0,0)
        )
        self.distinct_pos_2_qubit[(1,0)] = Lv1Qubit(
            (1,0), CoreType.XBUS, (1,0)
        )
        # data patches are arranged into rows of 4
        for i in range(self.size):
            position = self._data_index_2_distinct_position(i)
            self.distinct_pos_2_qubit[position] = Lv1Qubit(
                position, CoreType.INTCORE, position
            )

    def _set_SE_schedule(self):
        index_offset = 1 + self.size // 4
        z_anc = self.distinct_pos_2_qubit[(0,0)]
        x_anc = self.distinct_pos_2_qubit[(1,0)]
        for i in range(self.size//2):
            front_position = self._data_index_2_distinct_position(i)
            data_front = self.distinct_pos_2_qubit[front_position]
            data_front.set_SE_qubit(
                SESpaceTime(0, 1+(i//2), i%2),
                z_anc
            )
            z_anc.set_SE_qubit(
                SESpaceTime(0, 1+(i//2), i%2),
                data_front
            )
            data_front.set_SE_qubit(
                SESpaceTime(0, index_offset + (i//2), i%2),
                x_anc
            )
            x_anc.set_SE_qubit(
                SESpaceTime(0, index_offset + (i//2), i%2),
                data_front
            )
            back_position = self._data_index_2_distinct_position(self.size//2 + i)
            data_back = self.distinct_pos_2_qubit[back_position]
            data_back.set_SE_qubit(
                SESpaceTime(0, index_offset + (i//2), i%2),
                z_anc
            )
            z_anc.set_SE_qubit(
                SESpaceTime(0, index_offset + (i//2), i%2),
                data_back
            )
            data_back.set_SE_qubit(
                SESpaceTime(0, 1 + (i//2), i%2),
                x_anc
            )
            x_anc.set_SE_qubit(
                SESpaceTime(0, 1 + (i//2), i%2),
                data_back
            )
        pass

        

class SquareBerg(Lv1CSSCode):
    '''
    Quantum two-dimensional parity check codes. 
    '''
    def __init__(self, length: int, 
                 num_buses: int,
                 time_unit = 1):
        if length % 4 != 0:
            raise ValueError(f'{length=} of a squareberg code should be dividable by 4')
        self.length = length
        self.size = self.length**2
        self.num_buses = num_buses
        self.bus_distinct_pos_2_bus_index: dict[tuple[int,int],int] = {}
        super().__init__()
        self.time_unit = time_unit
        self.se_start_padding_time = 1
        self.se_end_padding_time = 1
        

    @property
    def qubit_overhead_rate(self) -> float:
        return (self.size + self.num_buses*4) / (self.size-4*self.length+2)
    
        
    def _set_logical_qubits(self):
        logical_id = 0
        bottom_left_Z_ancor = self.distinct_pos_2_qubit[(0,-1)]
        top_right_X_ancor = self.distinct_pos_2_qubit[(self.length-1, -(self.length))]
        for i in range(self.length):
            for j in range(self.length):
                if i == 0 or j == 0:
                    continue
                intersection = self.distinct_pos_2_qubit[(i,-1-j)]
                if j % 2 == 1 and i % 2 == 1:
                    if j == self.length - 1 or i == self.length - 1:
                        continue
                    logical_qubit = Lv1LogicalQubit(
                        id = logical_id,
                        x_supp=[intersection.distinct_position, 
                                (self.length-1, -1-j),
                                (i, -(self.length)),
                                top_right_X_ancor.distinct_position],
                        z_supp=[bottom_left_Z_ancor.distinct_position,
                                (i, -1), (0, -1-j), intersection.distinct_position]
                    )
                elif j % 2 == 1 and i % 2 == 0:
                    if i == self.length - 2:
                        continue
                    logical_qubit = Lv1LogicalQubit(
                        id = logical_id,
                        x_supp=[intersection.distinct_position, 
                                (self.length-2, -1-j),
                                (i+1, -(self.length)),
                                top_right_X_ancor.distinct_position],
                        z_supp=[bottom_left_Z_ancor.distinct_position,
                                (i, -1), (0, -1-j), intersection.distinct_position]
                    )
                elif j % 2 == 0 and i % 2 == 1:
                    if j == self.length - 2:
                        continue
                    logical_qubit = Lv1LogicalQubit(
                        id = logical_id,
                        x_supp=[intersection.distinct_position, 
                                (self.length-1, -j-2),
                                (i, -(self.length-1)),
                                top_right_X_ancor.distinct_position],
                        z_supp=[bottom_left_Z_ancor.distinct_position,
                                (i, -1), (0, -1-j), intersection.distinct_position]
                    )
                else:
                    logical_qubit = Lv1LogicalQubit(
                        id = logical_id,
                        x_supp=[intersection.distinct_position, 
                                (self.length-2, -j-2),
                                (i+1, -(self.length-1)),
                                top_right_X_ancor.distinct_position],
                        z_supp=[bottom_left_Z_ancor.distinct_position,
                                (i, -1), (0, -1-j), intersection.distinct_position]
                    )
                self.logical_qubits.append(logical_qubit)
                logical_id += 1
    
    def _bus_index_2_circ_position(self, bus_index: int) -> tuple[int,int]:
        x_span = self.length // 4
        return (bus_index % x_span, bus_index // x_span)

    def _set_qubits(self):
        # ancilla patches at bottom layers
        bus_index = 0        
        for y_index, core_type in [(1, CoreType.ZBUS),
                                   (2, CoreType.XBUS),
                                   (3, CoreType.ZBUS),
                                   (4, CoreType.XBUS)]:
            for i in range(self.length):
                self.distinct_pos_2_qubit[(i, y_index)] = Lv1Qubit(
                    self._bus_index_2_circ_position(bus_index),
                    core_type, (i, y_index)
                )
                self.bus_distinct_pos_2_bus_index[(i, y_index)] = bus_index
                bus_index = bus_index + 1
                bus_index = bus_index % self.num_buses
        # data patches are arranged into rows of `self.length`
        for i in range(self.length):
            for j in range(self.length):
                position = (i, -1-j)
                self.distinct_pos_2_qubit[position] = Lv1Qubit(
                    position, CoreType.INTCORE, position
                )

    def _set_SE_schedule(self):
        se_group_index = 0
        bus_index = 0
        used_buses: set[int] = set()
        
        for y_index, core_type in [(1, CoreType.ZBUS),
                                   (2, CoreType.XBUS),
                                   (3, CoreType.ZBUS),
                                   (4, CoreType.XBUS)]:
            for i in range(self.length):
                bus = self.distinct_pos_2_qubit[(i, y_index)]
                assert bus_index == self.bus_distinct_pos_2_bus_index[bus.distinct_position]
                port = 0
                if bus_index in used_buses:
                    se_group_index += 1
                    used_buses.clear()
                ordered_data_targets: list[Lv1Qubit] = []
                if y_index == 1:
                    # Z bi-col checks
                    row_index = i % 2
                    col_index = 2 * (i // 2)
                    for supp_index in range(self.length):
                        data = self.distinct_pos_2_qubit[(col_index + supp_index % 2,
                                                          -1 -row_index - 2 * (supp_index // 2))]
                        ordered_data_targets.append(data)
                elif y_index == 2:
                    if 2*i < self.length:
                        col_index = 2*i 
                    else:
                        col_index = (2*i % self.length) + 1
                    for supp_index in range(self.length):
                        data = self.distinct_pos_2_qubit[(col_index, -1-supp_index)]
                        ordered_data_targets.append(data)
                elif y_index == 3:
                    if 2*i < self.length:
                        row_index = 2*i 
                        col_index = 0
                    else:
                        row_index = (2*i % self.length)
                        col_index = 1
                    for supp_index in range(self.length):
                        data = self.distinct_pos_2_qubit[(2*supp_index % self.length + col_index, 
                                                          -1 - row_index - (2*supp_index // self.length))]
                        ordered_data_targets.append(data)
                else:
                    row_index = i
                    for supp_index in range(self.length):
                        data = self.distinct_pos_2_qubit[(supp_index, -1-row_index)]
                        ordered_data_targets.append(data)

                in_block_index_trial = 1
                port_trial = 0
                for data in ordered_data_targets:
                    trial_se_spacetime = SESpaceTime(se_group_index, in_block_index_trial, port_trial)
                    if trial_se_spacetime in data.occupied_spacetime:
                        se_group_index += 1
                        used_buses.clear()
                    port_trial = port_trial + 1
                    if port_trial == 4:
                        in_block_index_trial += 1
                        port_trial = 0 
                
                in_block_index = 1
                for data in ordered_data_targets:
                    bus.set_SE_qubit(SESpaceTime(se_group_index,
                                                    in_block_index, port), data)
                    data.set_SE_qubit(SESpaceTime(se_group_index,
                                                    in_block_index, port), bus)
                    port = port + 1
                    if port == 4:
                        in_block_index += 1
                        port = 0
                used_buses.add(bus_index)
                bus_index = (bus_index + 1) % self.num_buses


class QHammingCode(Lv1CSSCode):
    pass

class QBCHCode(Lv1CSSCode):
    pass

