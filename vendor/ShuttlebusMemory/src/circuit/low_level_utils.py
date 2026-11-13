from __future__ import annotations

from typing import Literal, Union, Optional
from warnings import warn
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
import copy

import stim

from gate_types import *


class NoiseConfig:

    def __init__(self):
        self.noises: dict[Noise,float] = {}
    
    def add_noise(self, noise: tuple[Noise,float]):
        if noise[0] in self.noises:
            warn(f'{noise[0].name} is reset to {noise[1]}')
        self.noises[noise[0]] = noise[1]


class InstKey:
    '''
    Represent an instruction during an SE round. 

    **Args**:
    - index: index of this instruction layer
    - operation: a particular `Gate` or `Noise` operation 
    - type: operation type, one of `RESET`, `MEAS`, `GATE`, and `NOISE`
    '''
    def __init__(self, index: int, operation: Union[Gate, Noise]):
        self.index = index
        self.operation = operation
        self.type = self._get_type() 

    def _get_type(self) -> Literal['RESET','MEAS','GATE','NOISE']:
        if isinstance(self.operation, Noise):
            return 'NOISE'
        elif self.operation.name in ['RZ','RX']:
            return 'RESET'
        elif self.operation.name in ['MZ','MX']:
            return 'MEAS'
        else:
            return 'GATE'

    def __hash__(self):
        return hash((self.index, self.type, self.operation.name))

    def __eq__(self, value: InstKey) -> bool:
        if self.index == value.index and self.type == value.type \
           and self.operation == value.operation:
            return True
        else:
            return False
        
    def __repr__(self):
        return f'index={self.index}, operation={self.operation.name}, ' \
               f'type={self.type}.'


@dataclass
class InstCell(ABC):

    @property
    @abstractmethod
    def ordered_Insts(self) -> list[Optional[InstKey]]:
        pass

    @property
    @abstractmethod
    def abstractInst_2_InstIndex(self) -> dict[AbstractInst,int]:
        pass
    
    @property
    def time_ordered_abstractInsts(self) -> list[AbstractInst]:
        return sorted(
            self.abstractInst_2_InstIndex.keys(),
            key = lambda abstractInst: self.abstractInst_2_InstIndex[abstractInst]
        )
    
    @property
    def abstractInst_2_InstKey(self) -> dict[AbstractInst,InstKey]:
        ret_dict = {}
        for index, absinst in enumerate(self.time_ordered_abstractInsts):
            ret_dict[absinst] = self.ordered_Insts[index]
        return ret_dict


@dataclass
class GATEInstCell(InstCell):
    GATE_Inst: Optional[InstKey] = None
    GATE_Noise: Optional[InstKey] = None
    IDLE_Noise: Optional[InstKey] = None

    @property
    def index(self) -> int:
        return self.GATE_Inst.index
    
    @property
    def ordered_Insts(self) -> list[Optional[InstKey]]:
        return [self.GATE_Inst, self.GATE_Noise, self.IDLE_Noise]
    
    @property
    def abstractInst_2_InstIndex(self) -> dict[AbstractInst,int]:
        return {AbstractInst.GATE: 0,
                AbstractInst.GATE_NOISE: 1,
                AbstractInst.IDLE_NOISE: 2}


@dataclass
class RESETInstCell(InstCell):
    Z_RESET_Inst: Optional[InstKey] = None
    Z_RESET_Noise: Optional[InstKey] = None
    X_RESET_Inst: Optional[InstKey] = None
    X_RESET_Noise: Optional[InstKey] = None
    IDLE_Noise: Optional[InstKey] = None

    @property
    def ordered_Insts(self) -> list[Optional[InstKey]]:
        return [
            self.Z_RESET_Inst, self.Z_RESET_Noise, 
            self.X_RESET_Inst, self.X_RESET_Noise, self.IDLE_Noise
        ]
    
    @property
    def abstractInst_2_InstIndex(self) -> dict[AbstractInst,int]:
        return {AbstractInst.Z_RESET: 0,
                AbstractInst.Z_RESET_NOISE: 1,
                AbstractInst.X_RESET: 2,
                AbstractInst.X_RESET_NOISE: 3,
                AbstractInst.IDLE_NOISE: 4}
    

@dataclass
class MEASInstCell(InstCell):
    Z_MEAS_Inst: Optional[InstKey] = None
    Z_MEAS_Noise: Optional[InstKey] = None
    X_MEAS_Inst: Optional[InstKey] = None
    X_MEAS_Noise: Optional[InstKey] = None
    MPP: Optional[InstKey] = None
    IDLE_Noise: Optional[InstKey] = None

    @property
    def ordered_Insts(self) -> list[Optional[InstKey]]:
        return [
            self.Z_MEAS_Noise, self.Z_MEAS_Inst,
            self.X_MEAS_Noise, self.X_MEAS_Inst,
            self.MPP, self.IDLE_Noise
        ]
    
    @property
    def abstractInst_2_InstIndex(self) -> dict[AbstractInst,int]:
        return {AbstractInst.Z_MEAS_NOISE: 0,
                AbstractInst.Z_MEAS: 1,
                AbstractInst.X_MEAS_NOISE: 2,
                AbstractInst.X_MEAS: 3,
                AbstractInst.MPP: 4,
                AbstractInst.IDLE_NOISE: 5}


class CircInstCellSinglet:

    def __init__(self):
        self.abstractInst_2_CircInst: dict[AbstractInst,
                                           Optional[CircuitInstruction]] = {}
        self.type: CellType
    
    @staticmethod
    def from_type(type: CellType):
        if type == CellType.MEASInstCell:
            return CircInstCellSinglet.from_InstCell(MEASInstCell())
        elif type == CellType.GATEInstCell:
            return CircInstCellSinglet.from_InstCell(GATEInstCell())
        else:
            assert type == CellType.RESETInstCell
            return CircInstCellSinglet.from_InstCell(RESETInstCell())

    @staticmethod
    def from_InstCell(InstCell: Union[MEASInstCell,
                                      GATEInstCell,
                                      RESETInstCell]) -> CircInstCellSinglet:
        circInstCell = CircInstCellSinglet()
        if isinstance(InstCell,MEASInstCell):
            circInstCell.type = CellType.MEASInstCell
        elif isinstance(InstCell,RESETInstCell):
            circInstCell.type = CellType.RESETInstCell
        else:
            assert isinstance(InstCell,GATEInstCell)
            circInstCell.type = CellType.GATEInstCell
        for abstractInst in InstCell.abstractInst_2_InstIndex:
            circInstCell.abstractInst_2_CircInst[abstractInst] = None
        return circInstCell
    
    def _verify(self) -> bool:
        assert isinstance(self.type,CellType)
        abstractInsts = set(self.abstractInst_2_CircInst.keys())
        if self.type == CellType.MEASInstCell:
            if abstractInsts == set(MEASInstCell().time_ordered_abstractInsts):
                return True
            else: 
                return False
        elif self.type == CellType.GATEInstCell:
            if abstractInsts == set(GATEInstCell().time_ordered_abstractInsts):
                return True
            else:
                return False
        else:
            if abstractInsts == set(RESETInstCell().time_ordered_abstractInsts):
                return True
            else:
                return False
    
    def check_compatibility_with_same_type(self, second: CircInstCellSinglet) -> bool:
        # throw if type mismatches
        if second.type != self.type:
            raise ValueError(f'{self.type=} mismatches with {second.type=}.')
        # verify both singlets
        if not (self._verify() and second._verify()):
            raise ValueError(f'{self._verify()=} and {second._verify()=}\n'
                             'verification failed.')
        for absInst in self.abstractInst_2_CircInst:
            if not instructions_compatibility_check([
                self.abstractInst_2_CircInst[absInst],
                second.abstractInst_2_CircInst[absInst]]):
                return False
        return True
    
    def deepcopy(self) -> CircInstCellSinglet:
        ret_copy = CircInstCellSinglet.from_type(self.type)
        for absInst in self.abstractInst_2_CircInst:
            if self.abstractInst_2_CircInst[absInst] is None:
                ret_copy.abstractInst_2_CircInst[absInst] = None
            else:
                ret_copy.abstractInst_2_CircInst[absInst] \
                    = self.abstractInst_2_CircInst[absInst].deepcopy()
        return ret_copy
    
    def __iadd__(self, second: CircInstCellSinglet) -> CircInstCellSinglet:
        if not self.check_compatibility_with_same_type(second):
            raise ValueError('Addition failure due to incompatibility.')
        for absInst in self.abstractInst_2_CircInst:
            if self.abstractInst_2_CircInst[absInst] is None:
                if second.abstractInst_2_CircInst[absInst] is not None:
                    self.abstractInst_2_CircInst[absInst] \
                    = second.abstractInst_2_CircInst[absInst]
            else:
                if second.abstractInst_2_CircInst[absInst] is not None:
                    self.abstractInst_2_CircInst[absInst] \
                    += second.abstractInst_2_CircInst[absInst]
        return self

    def __add__(self, second: CircInstCellSinglet) -> CircInstCellSinglet:
        if not self.check_compatibility_with_same_type(second):
            raise ValueError('Addition failure due to incompatibility.')
        register = CircInstCellSinglet.from_type(self.type)
        for absInst in self.abstractInst_2_CircInst:
            if self.abstractInst_2_CircInst[absInst] is None:
                if second.abstractInst_2_CircInst[absInst] is not None:
                    register.abstractInst_2_CircInst[absInst] \
                    = second.abstractInst_2_CircInst[absInst].deepcopy()
            else:
                register.abstractInst_2_CircInst[absInst] \
                = self.abstractInst_2_CircInst[absInst].deepcopy()
                if second.abstractInst_2_CircInst[absInst] is not None:
                    register.abstractInst_2_CircInst[absInst] \
                    += second.abstractInst_2_CircInst[absInst]
        return register
    
    def __eq__(self, value: CircInstCellSinglet) -> bool:
        # verify both singlets
        if not (self._verify() and value._verify()):
            raise ValueError(f'{self._verify()=} and {value._verify()=}\n'
                             'verification failed.')
        if self.type != value.type:
            return False
        for absInst in self.abstractInst_2_CircInst:
            if not(self.abstractInst_2_CircInst[absInst] \
                   == value.abstractInst_2_CircInst[absInst]):
                return False
        return True
    
    @property
    def qubit_support(self) -> set[int]:
        '''
        Set of circuit ids for qubits in this cell.
        '''
        qubit_supp = set()
        for circinst in self.abstractInst_2_CircInst.values():
            if circinst is None:
                continue
            else:
                gate_targets = circinst.targets_copy()
                for target in gate_targets:
                    if target.qubit_value is not None:
                        qubit_supp.add(target.qubit_value)
        return qubit_supp
    

class CircCells_Manager:

    def __init__(self):
        self.index_2_CircCell: dict[int,CircInstCellSinglet] = {}
        self.index_2_CircCell_still: dict[int,CircInstCellSinglet] = {}

    def set_CircCell(self, index: int, CircCell: CircInstCellSinglet):
        self.index_2_CircCell_still[index] = CircCell
        self.index_2_CircCell[index] = CircCell
        pass

    def update_CircCell(self, index: int, CircCell: CircInstCellSinglet):
        if index not in self.index_2_CircCell:
            raise KeyError(f'Update failed. CircCell at {index=} is unset.')
        self.index_2_CircCell[index] = CircCell
        pass
    
    def reset(self):
        for index in self.index_2_CircCell:
            self.index_2_CircCell[index] = self.index_2_CircCell_still[index]
    
    def empty_set(self):
        for index in self.index_2_CircCell:
            self.index_2_CircCell[index] = CircInstCellSinglet.from_type(
                self.index_2_CircCell_still[index].type
            )

    def clear(self):
        self.index_2_CircCell.clear()
        self.index_2_CircCell_still.clear()


class CircInstCellStrictBuffer:

    def __init__(self, type: Optional[CellType] = None, strictness = True):
        if type is None:
            self.circInstcellbuff = None
            self.type: CellType
            self.strictness = strictness
        else:
            self.circInstcellbuff = CircInstCellSinglet.from_type(type)
            self.type = type
            self.strictness = strictness
    
    def __iadd__(self, second: CircInstCellSinglet) -> CircInstCellStrictBuffer:
        # First verify the Singlet
        if not second._verify():
            raise ValueError('verification failure.\n'
                    'Input CircInstCellSinglet does not match with its type.')
        # check if the buffer is empty
        if self.circInstcellbuff is None:
            self.circInstcellbuff = second.deepcopy()
            self.type = second.type
            return self
        # Type checking
        if second.type != self.type:
            raise ValueError(f'{self.type=} while {second.type=}.\n'
                             'Addition failure due to type mismatch.')
        
        if self.strictness:
            # check no qubit overlap
            if self.circInstcellbuff.qubit_support.intersection(second.qubit_support):
                raise ValueError('to-be-added cell intersects with the current buffer')
            # start merging
            if self.circInstcellbuff.check_compatibility_with_same_type(second):
                self.circInstcellbuff += second
                return self
            else:
                raise RuntimeError('failed to merge. incompatible with the added cell')
        else:
            self.circInstcellbuff += second
            return self

    def extend(self, seconds: Iterable[CircInstCellSinglet]) -> CircInstCellStrictBuffer:
        cells = list(seconds)
        if not cells:
            return self
        if not self.strictness:
            for second in cells:
                self += second
            return self
        if self.circInstcellbuff is not None:
            cells.insert(0, self.circInstcellbuff)
        cell_type = cells[0].type
        qubit_support = set()
        for cell in cells:
            if not cell._verify():
                raise ValueError('verification failure.\n'
                                 'Input CircInstCellSinglet does not match with its type.')
            if cell.type != cell_type:
                raise ValueError(f'{cell_type=} while {cell.type=}.\n'
                                 'Addition failure due to type mismatch.')
            if qubit_support.intersection(cell.qubit_support):
                raise ValueError('to-be-added cell intersects with the current buffer')
            qubit_support.update(cell.qubit_support)
        merged = CircInstCellSinglet.from_type(cell_type)
        for abs_inst in merged.abstractInst_2_CircInst:
            instructions = [cell.abstractInst_2_CircInst[abs_inst]
                            for cell in cells]
            if not instructions_compatibility_check(instructions):
                raise RuntimeError('failed to merge. incompatible with the added cell')
            merged.abstractInst_2_CircInst[abs_inst] = merge_instructions(instructions)
        self.circInstcellbuff = merged
        self.type = cell_type
        return self

# TODO: Deprecate this.        
class CircInstCellBuffer:

    def __init__(self):
        self.CircInst_layers: list[CircInstCellSinglet] = []
        self.type: CellType
    
    def __iadd__(self, second: CircInstCellSinglet):
        # First verify the Singlet
        if not second._verify():
            raise ValueError('verification failure.\n'
                    'Input CircInstCellSinglet does not match with its type.')
        if not self.CircInst_layers:
            self.CircInst_layers.append(second)
            self.type = second.type
            return self
        # Type checking
        if second.type != self.type:
            raise ValueError(f'{self.type=} while {second.type=}.\n'
                             'Addition failure due to type mismatch.')
        # merge iff compatible
        merge_success = False
        for i in range(len(self.CircInst_layers)):
            if self.CircInst_layers[i].check_compatibility_with_same_type(second):
                self.CircInst_layers[i] = self.CircInst_layers[i] + second
                merge_success = True
        if not merge_success:
            self.CircInst_layers.append(second)
        return self


class StabFlow:
    '''
    Track the flow of stabilizers. This information is used to automatically infer
    detectors. 

    **Args**:
    - `stab_id`: circuit id of the qubit corresponding to the stabilizer
    - `pauli_default`: Pauli basis of the stabilizer
    - `flow_from`: If not `None`, this represents the state is initially stabilized
    by a stabilizer with id `flow_from[0]` and Pauli basis `flow_from[1]`.
    - `flow_to': If not empty, this represents the state is currently stabilied 
    by the product of stabilizers in `flow_to`.
    
    '''
    def __init__(self, stab_id: int, pauli_default: Optional[Pauli] = None):
        self.stab_id: int = stab_id
        self.pauli_default = pauli_default
        self.flow_from: Optional[tuple[int,Pauli]] = None
        self.flow_to: set[tuple[int,Pauli]] = set()
        self.destruct_to: Optional[tuple[set[int],Pauli]] = None 
        self.trans_init = False
        pass

    def __repr__(self) -> str:
        info = f'StabFlow-{self.stab_id}:[' \
               f'Default Pauli basis: {self.pauli_default}, ' \
               f'Flow from: {self.flow_from}, ' \
               f'Flow to: {self.flow_to}, ' \
               f'Tranvsersal Init: {self.trans_init}]\n'
        return info

    def stab_reset(self):
        assert isinstance(self.pauli_default,Pauli)
        self.flow_from = (self.stab_id, self.pauli_default)
        self.flow_to = set([(self.stab_id, self.pauli_default)])
        self.destruct_to = None
        self.trans_init = False

    def stab_trans_init(self):
        assert isinstance(self.pauli_default,Pauli)
        self.flow_from = (self.stab_id, self.pauli_default)
        self.flow_to = set([(self.stab_id, self.pauli_default)])
        self.destruct_to = None
        self.trans_init = True
    
    def random_reset(self):
        assert isinstance(self.pauli_default,Pauli)
        self.flow_from = None
        self.flow_to = set([(self.stab_id, self.pauli_default)])
        self.destruct_to = None
        self.trans_init = False

    def empty_reset(self):
        self.flow_from = None
        self.flow_to = set()
        self.destruct_to = None
        self.trans_init = False

    def destructive_meas(self, project_targets: set[int], basis: Pauli):
        if len(project_targets) == 0 or len(self.flow_to) == 0:
            raise ValueError('empty detector')
        self.destruct_to = (project_targets, basis)

class LogicalFlow:

    def __init__(self, logical_id: int):
        self.logical_id = logical_id
        self.logical_x_supp_ids: set[int] = set()
        self.logical_z_supp_ids: set[int] = set()
        self.flow_from: Optional[tuple[int,Pauli]] = None
        self.flow_to: set[tuple[int,Pauli]] = set()
        self.destruct_to: set[tuple[int,Pauli]] = set()
    
    def logical_pauli_reset(self, pauli: Pauli):
        if pauli not in [Pauli.X, Pauli.Z]:
            raise NotImplementedError(f'{pauli=} should be Pauli X or Z')
        self.flow_from = (self.logical_id, pauli)
        self.flow_to = (self.logical_id, pauli)
    
    def empty_reset(self):
        self.flow_from = None
        self.flow_to = set()
    
    def destructive_meas(self, targets_w_basis: set[tuple[int,Pauli]]):
        if len(targets_w_basis) == 0 or len(self.flow_to) == 0:
            raise ValueError('empty logical observable')
        self.destruct_to = targets_w_basis


class CircuitInstruction(stim.CircuitInstruction):
    '''
    wrapper around stim.CircuitInstruction to allow the += operation.
    '''
    def __init__(self, name, targets = None, gate_args = None, *, tag = "",
                 anc_id_2_mpp_group_index: dict[int,int] = {}):
        super().__init__(name, targets, gate_args, tag=tag)
        if self.name == 'MPP':
            group_length = len(self.target_groups())
            for group_index in anc_id_2_mpp_group_index.values():
                if group_index > group_length-1:
                    raise ValueError(f'{group_index=} should be no larger than the '
                                     f'maximum group index {group_length-1}.')
            self.anc_id_2_mpp_group_index = anc_id_2_mpp_group_index

    def __iadd__(self, second: Union[CircuitInstruction,list[int]]) -> CircuitInstruction:
        if isinstance(second,CircuitInstruction):
            merged = merge_instructions([self, second])
            return merged 
        elif isinstance(second,list):
            if self.name == 'MPP':
                raise NotImplementedError(f'Cannot add a list of targets to'
                                          f'an {self.name} Instruction')
            # check no intersection with original targets
            if set(self.qubit_target_values()).intersection(set(second)):
                raise ValueError(f'{second=} intersects with previous targets'
                                f'{self.qubit_target_values()}.')
            qubit_targets = self.qubit_target_values() + second
            return CircuitInstruction(self.name, qubit_targets,
                                    self.gate_args_copy(), tag=self.tag)
        else:
            raise TypeError(f'{second} should be either a CircuitInstruction ot a'
                            'list of integers.')
    
    # DEPRECATED...
    def add_targets(self, targets: list[int]) -> CircuitInstruction:
        # check no intersection with original targets
        if set(self.qubit_target_values()).intersection(set(targets)):
            raise ValueError(f'{targets=} intersects with previous targets'
                             f'{self.qubit_target_values()}.')
        qubit_targets = self.qubit_target_values() + targets
        return CircuitInstruction(self.name, qubit_targets,
                                  self.gate_args_copy(), tag=self.tag)

    def qubit_target_values(self) -> list[int]:
        if self.name == 'MPP':
            raise NotImplementedError(f'{self.name} not supported.')
        return get_qubit_target_values_from_instruction(self)
    
    def deepcopy(self) -> CircuitInstruction:
        if self.name == 'MPP':
            anc_id_2_mpp_group_index = copy.deepcopy(self.anc_id_2_mpp_group_index)
        else:
            anc_id_2_mpp_group_index = {}
        register = CircuitInstruction(self.name, self.targets_copy(),
                            self.gate_args_copy(), tag=self.tag,
                            anc_id_2_mpp_group_index=anc_id_2_mpp_group_index)
        return register
    
    def __eq__(self, arg0: CircuitInstruction):
        if self.name == 'MPP':
            if self.anc_id_2_mpp_group_index == arg0.anc_id_2_mpp_group_index:
                return super().__eq__(arg0)
            else:
                return False
        else:
            return super().__eq__(arg0)


@dataclass
class QubitMeas_recorder:
    time_basis_2_meas_id: dict[tuple[int,Pauli],int] = field(default_factory=dict)


def is_instruction_empty(circ_instruction: CircuitInstruction) -> bool:
    if circ_instruction.targets_copy():
        return False
    else:
        return True

def get_qubit_target_values(gate_targets: list[stim.GateTarget]) -> list[int]:
    ret_list = []
    for gate_target in gate_targets:
        if not gate_target.is_qubit_target:
            raise ValueError(f'{gate_target=} is not a qubit target.') # raise for Pauli targets.
        ret_list.append(gate_target.value)
    return ret_list

def get_qubit_target_values_from_instruction(
                    circ_instruction: CircuitInstruction) -> list[int]:
    if circ_instruction.name == 'MPP':
        raise NotImplementedError('Undefined behavior to extract qubit ids from '
                                  'an MPP instruction.')
    return get_qubit_target_values(circ_instruction.targets_copy())
    
def instructions_compatibility_check(circ_instructions: 
                                     list[Optional[CircuitInstruction]]) -> bool:
    '''
    Reasons for incompatibility::

        1. instruction name mismatch.
        2. instruction args mismatch.
        3. instruction tag mismatch. (since stim 1.15)
        4. non-MPP instructions have repeated arguments. (To make sure gates/resets/measurements
        are contained in one layer.)

    '''
    if len(circ_instructions) == 0:
        raise ValueError('input circuit instructions should not be empty.')
    AllNone = True
    for circ_instruction in circ_instructions:
        if circ_instruction is None:
            continue
        else:
            instruction_name = circ_instruction.name
            instruction_tag = circ_instruction.tag
            instruction_args = circ_instruction.gate_args_copy()
            AllNone = False
    if AllNone:
        return True
    targets: set[int] = set()
    mpp_anc_ids: set[int] = set()
    for circ_instruction in circ_instructions:
        if circ_instruction is None:
            continue
        if circ_instruction.name != instruction_name:
            return False
        if circ_instruction.gate_args_copy() != instruction_args:
            return False
        if circ_instruction.tag != instruction_tag:
            return False
        if circ_instruction.name == 'MPP':
            new_anc_ids = set(circ_instruction.anc_id_2_mpp_group_index.keys())
            if mpp_anc_ids.intersection(new_anc_ids):
                return False
            mpp_anc_ids.update(new_anc_ids)
            continue
        circ_instruction_targets \
            = get_qubit_target_values_from_instruction(circ_instruction)
        if len(circ_instruction_targets) != len(set(circ_instruction_targets)):
            raise ValueError(f'{circ_instruction_targets=} contain '
                             'repeated target values.')
        if targets.intersection(circ_instruction_targets):
            return False
        targets.update(circ_instruction_targets)
    return True

def merge_instructions(circ_instructions: list[Optional[CircuitInstruction]]) \
                                                -> Optional[CircuitInstruction]:
    if len(circ_instructions) == 0:
        raise ValueError('input circuit instructions should not be empty.')
    AllNone = True
    for circ_instruction in circ_instructions:
        if circ_instruction is None:
            continue
        else:
            instruction_name = circ_instruction.name
            instruction_tag = circ_instruction.tag
            instruction_args = circ_instruction.gate_args_copy()
            instruction_targets: list[stim.GateTarget] = []
            AllNone = False
            break
    if AllNone:
        return None
    group_index_offset = 0  # MPP
    anc_id_2_group_index: dict[int,int] = {}  # MPP
    existing_target_vals = set()
    for circ_instruction in circ_instructions:
        if circ_instruction is None:
            continue
        if circ_instruction.name != instruction_name:
            raise ValueError('merge failure due to instruction name mismatch.\n '
                    f'one {circ_instruction.name=} while {instruction_name=}')
        if circ_instruction.gate_args_copy() != instruction_args:
            raise ValueError('merge failure due to instruction args mismatch.\n '
                            f'one {circ_instruction.gate_args_copy()=} while '
                            f'{instruction_args=}')
        if circ_instruction.tag != instruction_tag:
            raise ValueError('merge failure due to instruction tag mismatch.\n '
                    f'one {circ_instruction.tag=} while {instruction_tag=}')
        if circ_instruction.name != 'MPP':
            new_target_vals = set([target.value for target in
                                                circ_instruction.targets_copy()])
            repeated_target_vals = existing_target_vals.intersection(new_target_vals)
            if repeated_target_vals:
                raise ValueError('merge failure due to repeated instructions.\n '
                                f'{repeated_target_vals=}')
            existing_target_vals.update(new_target_vals)
        else:
            for anc_id in circ_instruction.anc_id_2_mpp_group_index:
                if anc_id in anc_id_2_group_index:
                    raise ValueError(f'more than one {anc_id=} is measured at '
                                     'the same step.')
                anc_id_2_group_index[anc_id] \
                        = (circ_instruction.anc_id_2_mpp_group_index[anc_id] 
                           + group_index_offset)
            group_index_offset += len(circ_instruction.target_groups())
            pass
        instruction_targets += circ_instruction.targets_copy()
    return CircuitInstruction(instruction_name, 
                instruction_targets, instruction_args, tag=instruction_tag,
                anc_id_2_mpp_group_index=anc_id_2_group_index)
        
def mpp_type(targets: list[stim.GateTarget]) -> Optional[Pauli]:
    if not targets:
        raise ValueError(f'{targets=} should be non-empty')
    try:
        ret_type = pauli_target_type(targets[0])
    except:
        raise TypeError(f'{targets=} should be a valid MPP instance')
    for target in targets:
        if pauli_target_type(target) != ret_type:
            return None
    return ret_type

def pauli_target_type(target: stim.GateTarget) -> Pauli:
    if target.is_x_target:
        return Pauli.X
    elif target.is_y_target:
        return Pauli.Y
    elif target.is_z_target:
        return Pauli.Z
    raise TypeError(f'{target=} should be a Pauli target')