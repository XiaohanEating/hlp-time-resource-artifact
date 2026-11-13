from __future__ import annotations

from enum import Enum, unique
from dataclasses import dataclass, asdict
from typing import Literal

@dataclass(order=True)
class _Operation:
    op_name: str
    op_instruction: str

@unique
class Noise(Enum):
    '''
    Use _Operation to avoid value clashing. Otherwise noises with 
    the same instruction will not be differentiated.
    '''
    IDLE = _Operation('IDLE','DEPOLARIZE1')
    SQGATE = _Operation('SQGATE','DEPOLARIZE1')
    TQGATE = _Operation('TQGATE','DEPOLARIZE2')
    Z_RESET = _Operation('Z_RESET','X_ERROR')
    X_RESET = _Operation('X_RESET','Z_ERROR')
    Z_MEAS = _Operation('Z_MEAS','X_ERROR')
    X_MEAS = _Operation('X_MEAS','Z_ERROR')
    X_ERROR = _Operation('X_ERROR','X_ERROR')
    Z_ERROR = _Operation('Z_ERROR','Z_ERROR')

    @property
    def instruction(self) -> str:
        return self.value.op_instruction

NOISE_NAMES = list([noise.name for noise in Noise])
NOISE_INSTRUCTIONS = list([noise.instruction for noise in Noise])

@unique
class Gate(Enum):
    # not all gates on stim are supported here. 
    X = _Operation('X','X')
    Y = _Operation('Y','Y')
    Z = _Operation('Z','Z')
    CX = _Operation('CX','CX')
    H = _Operation('H','H')
    S = _Operation('S','S')
    CZ = _Operation('CZ','CZ')
    RZ = _Operation('RZ','RZ')
    RX = _Operation('RX','RX')
    MZ = _Operation('MZ','MZ')
    MX = _Operation('MX','MX')
    MPP = _Operation('MPP','MPP') 
    MPP_RZ = _Operation('MPP_RZ','MPP') # TODO: add supports
    MPP_RX = _Operation('MPP_RX','MPP')
    MPP_MX = _Operation('MPP_MX','MPP')
    MPP_MZ = _Operation('MPP_MZ','MPP')
    MPP_RESET = _Operation('MPP_RESET','MPP')
    MPP_MEAS = _Operation('MPP_MEAS','MPP')

    @property
    def instruction(self) -> str:
        return self.value.op_instruction
    
    def __lt__(self, other: Gate) -> bool:
        return self.value < other.value 
    
    def __gt__(self, other: Gate) -> bool:
        return self.value > other.value
    
    def __eq__(self, other: Gate):
        return self.value == other.value
    
    def __hash__(self):
        return hash((self.name,self.value.op_instruction,self.value.op_name))



GATE_INSTRUCTIONS = list([gate.instruction for gate in Gate])
ACTIVATE_INSTRUCTIONS = [Gate.RX, Gate.RZ, Gate.MPP_RX, Gate.MPP_RZ, Gate.MPP_RESET]
DEACTIVATE_INSTRUCTIONS = [Gate.MX, Gate.MZ, Gate.MPP_MX, Gate.MPP_MZ, Gate.MPP_MEAS]


@unique
class LogicalOp(Enum):
    '''
    * LRZ, LRZ_DET_PERF: Data qubits RZ.
    * LRX, LRX_DET_PERF: Data qubits RX.
    * LMZ: Data qubits MZ; No logical reset next round (for easier DEM construction).
    * LMX: Data qubits MX; No logical reset next round (for easier DEM construction).
    * *_PERF: MPP all stabilizers. 
    * LRZ_UDET_PERF, LRX_UDET_PERF, 
      LMZ_PERF, LMX_PERF: MPP the corresponding Logical operator.  
    '''
    LRZ = _Operation('LRZ','RZ')
    LRX = _Operation('LRX','RX')
    LMZ = _Operation('LMZ','MZ')
    LMX = _Operation('LMX','MX')
    LRZ_DET_PERF = _Operation('LRZPERF','MPP')
    LRX_DET_PERF = _Operation('LRXPERF','MPP')
    LR_PERF = _Operation('LR_PERF','MPP')
    LSE = _Operation('LSE','SE')
    # LRZ_UDET_PERF = _Operation('LRZ_UDET_PERF', 'MPP')  # Unimplemented
    # LRX_UDET_PERF = _Operation('LRX_UDET_PERF', 'MPP')  # Unimplemented
    LMZ_PERF = _Operation('LMZPERF','MPP')
    LMX_PERF = _Operation('LMXPERF','MPP')
    LM_PERF = _Operation('LM_PERF','MPP')
    LNONE = _Operation('LNONE','LNONE') 

@unique
class CellType(Enum):
    '''
    '''
    RESETInstCell = 'RESETInstCell'
    GATEInstCell = 'GATEInstCell'
    MEASInstCell = 'MEASInstCell'

@unique
class AbstractInst(Enum):
    '''
    '''
    Z_RESET = 'Z_RESET_Inst'
    Z_RESET_NOISE = 'Z_RESET_Noise'
    X_RESET = 'X_RESET_Inst'
    X_RESET_NOISE = 'X_RESET_Noise'
    GATE = 'GATE_Inst'
    GATE_NOISE = 'GATE_Noise'
    Z_MEAS = 'Z_MEAS_Inst'
    Z_MEAS_NOISE = 'Z_MEAS_Noise'
    X_MEAS = 'X_MEAS_Inst'
    X_MEAS_NOISE = 'X_MEAS_Noise'
    IDLE_NOISE = 'IDLE_Noise'
    MPP = 'MPP'
    

@unique
class Pauli(Enum):
    X = _Operation('X','X')
    Y = _Operation('Y','Y')
    Z = _Operation('Z','Z')

    @property
    def basis(self) -> str:
        return self.value.op_instruction
    
@dataclass
class Logical_RM_Instruction:
    type: Literal['LR','LM','LRPP','LMPP']
    basis: Pauli
    patch_id: int

    def __setattr__(self, name, value):
        if name == 'basis' and value == Pauli.Y:
            raise NotImplementedError(f'Logical reset/measurement in '
                                      f'{value} basis is not implemented.')
        
class CoreType(Enum):
    INTCORE = 'INTERNAL_CORE'
    EXTCORE = 'ExTERNAL_CORE'
    XBUS = 'X_SHUTTLEBUS'
    ZBUS = 'Z_SHUTTLEBUS'
    UNDEFINED = 'UNDEFINED'
    OBSDUMMY = 'OBSDUMMY'
    
class CoreOp(Enum):
    RX = 'RX'
    RZ = 'RZ'
    MX = 'MX'
    MZ = 'MZ'
    MPP = 'MPP'
    MPP_X = 'MPP_X'
    MPP_Z = 'MPP_Z'
    MPP_BELL_RESET = 'MPP_BELL_RESET'
    MPP_BELL_MEAS = 'MPP_BELL_MEAS'
    CX = 'CX'
    # CX gate between shuttle buses and external cores
    # `RD` stands for reversed docking.
    CX_RD = 'CX_RD' 
    IDLE = 'IDLE'
    UNDEFINED = 'UNDEFINED'

class Lv1LogicalOp(Enum):
    RX = 'RX'
    RZ = 'RZ'
    BELL_RESET = 'BELL_RESET'

class Lv1ObsOp(Enum):
    ADD = 'ADD'
    MERGE = 'MERGE'

@dataclass
class TagBuffer:
    patch_id: int
    lv1_posx: int
    lv1_posy: int
    pauli: Literal['X','Z']
    boundary: bool
    boundary_index: Literal['0','1','UNDEF']
    lv1_time: int = 0
    # clock_time: int = 0
    aftercnot: bool = False
    cnot: Literal['C','T','UNDEF'] = 'UNDEF'
    patch_type: Literal['DATA','X_ANC','Z_ANC', 'UNDEF'] = 'UNDEF' 

    @property
    def tag(self) -> str:
        name_2_val = asdict(self)
        ret_tag = ''
        for name in name_2_val:
            if name == 'patch_type' and name_2_val[name] == 'UNDEF':
                    continue
            ret_tag += f'{name}={name_2_val[name]},'
        if name_2_val['boundary'] is False \
            and name_2_val['boundary_index'] in ['0','1']:
            return ValueError(f'Non-boundary term does not have a '
                              f'well-defined {self.boundary_index=}')
        if name_2_val['boundary'] is True \
            and name_2_val['boundary_index'] == 'UNDEF':
            return ValueError(f'boundary term must be assigned with a '
                              f'specific {self.boundary_index=}')
        if name_2_val['aftercnot'] is True and name_2_val['cnot'] == 'UNDEF':
            raise ValueError('Must be either the target or contrl of CNOT. '
                             f'Got {self.cnot=} instead.')
        if name_2_val['aftercnot'] is False and name_2_val['cnot'] != 'UNDEF':
            raise ValueError(f'{self.cnot=} should be undefined if the qubit '
                             'is not involved in a CNOT.')
        return ret_tag[:-1]
    
    def cnot_clear(self):
        self.aftercnot = False
        self.cnot = 'UNDEF'
