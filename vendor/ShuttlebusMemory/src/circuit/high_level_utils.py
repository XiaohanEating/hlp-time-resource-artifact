from __future__ import annotations
from typing import Iterable

from surface_code import *
from circuit_interface import *
from low_level_utils import *


class CodePatch:
    '''
    Code patch encoding a single logical qubit.
    '''
    def __init__(self, code: CSS_code_2D):
        self.code = code
        self.qubit_ids: set[int] = set()
        self.qubit_id_2_pos: dict[int,tuple[int,int]] = {}
        self.id_assignment_flag: bool = False
        self.SE_instructions: dict[InstKey, CircuitInstruction] = {}
        self.SEInstKeys: list[InstKey] = []   # time-ordered
        self.SEResetCell: RESETInstCell = RESETInstCell()
        self.SEMeasCell: MEASInstCell = MEASInstCell()
        self.SEGateCells: list[GATEInstCell] = []   # time-ordered
        self.SECircCells_manager = CircCells_Manager()
        self.GateNoiseSEInst_2_GateSEInst: dict[InstKey,InstKey] = {}
        self.SE_CX_ops: set[InstKey] = set() 
        self.SE_compilation_flag: bool = False
        self.special_cells_for_logic_ops: dict[tuple[LogicalOp,CellType],
                                               CircInstCellSinglet] = {}
        self.logical_op_following_se: Optional[LogicalOp] = None
        self.noise_config: NoiseConfig = NoiseConfig()
        self.stab_id_2_stab_flow: dict[int,StabFlow] = {}
        self.logical_id: Optional[int] = None
        self.logical_flow: Optional[LogicalFlow] = None
        self.active: bool = False
        self.patch_id: Optional[int] = None
        pass

    def _assign_qubit_circuit_ids(self, pos_2_id: dict[tuple[int,int],int]):
        if self.id_assignment_flag == True:
            raise ValueError(f'qubit circuit id already assigned')
        if len(pos_2_id) != len(self.code.qubits):
            raise ValueError(f'qubit circuit id set failure.\n'
                             f'{len(pos_2_id)=} is not the same as '
                             f'the number of qubits {len(self.code.qubits)}.')
        self.qubit_ids.clear()
        for pos in pos_2_id:
            self.code.position_2_qubit[pos].set_circuit_id(pos_2_id[pos])
            qubit_type = self.code.position_2_qubit[pos].qubit_type
            if qubit_type in [QubitType.X_ancilla, QubitType.Z_ancilla]:
                pauli_type = Pauli.X if qubit_type==QubitType.X_ancilla else Pauli.Z
                self.stab_id_2_stab_flow[pos_2_id[pos]] = StabFlow(pos_2_id[pos],
                                                                   pauli_type)
            if pos_2_id[pos] in self.qubit_ids:
                raise ValueError(f'qubit id {pos_2_id[pos]} already added.\n'
                                 f'the same id is assigned to multiple qubits.')
            self.qubit_ids.add(pos_2_id[pos])
            self.qubit_id_2_pos[pos_2_id[pos]] = pos
        self.id_assignment_flag = True

    def _compile_SE_instructions(self):
        if not self.id_assignment_flag:
            raise RuntimeError('Qubit ID unset.')
        self.SE_instructions.clear()
        SE_indices: set[int] = set()
        self.SE_CX_ops.clear()
        self.SEInstKeys.clear()
        self.GateNoiseSEInst_2_GateSEInst.clear()
        self.SEGateCells.clear()
        self.SECircCells_manager.clear()
        self.SEResetCell = RESETInstCell()
        self.SEMeasCell = MEASInstCell()
        # get all SE indices
        for qubit in self.code.qubits:
            SE_indices.update(qubit.SEIndex_2_neighbor_qubit.keys())
        # generate empty instructions: reset -> CX gates -> measurement
        z_reset_op = InstKey(0, Gate.RZ)
        self.SEResetCell.Z_RESET_Inst = z_reset_op
        self.SEInstKeys.append(z_reset_op)
        self.SE_instructions[z_reset_op] = CircuitInstruction(Gate.RZ.instruction)
        if Noise.Z_RESET in self.noise_config.noises:
            z_reset_noise = InstKey(0, Noise.Z_RESET)
            self.SEResetCell.Z_RESET_Noise = z_reset_noise
            self.SEInstKeys.append(z_reset_noise)
            self.GateNoiseSEInst_2_GateSEInst[z_reset_noise] = z_reset_op
            self.SE_instructions[z_reset_noise] = CircuitInstruction(
                        Noise.Z_RESET.instruction, 
                        gate_args=[self.noise_config.noises[Noise.Z_RESET]])
        x_reset_op = InstKey(0, Gate.RX)
        self.SEResetCell.X_RESET_Inst = x_reset_op
        self.SEInstKeys.append(x_reset_op)
        self.SE_instructions[x_reset_op] = CircuitInstruction(Gate.RX.instruction)
        if Noise.X_RESET in self.noise_config.noises:
            x_reset_noise = InstKey(0, Noise.X_RESET)
            self.SEResetCell.X_RESET_Noise = x_reset_noise
            self.SEInstKeys.append(x_reset_noise)
            self.GateNoiseSEInst_2_GateSEInst[x_reset_noise] = x_reset_op
            self.SE_instructions[x_reset_noise] = CircuitInstruction(
                    Noise.X_RESET.instruction,
                    gate_args=[self.noise_config.noises[Noise.X_RESET]])
        if Noise.IDLE in self.noise_config.noises:
            reset_idle_noise = InstKey(0, Noise.IDLE)
            self.SEResetCell.IDLE_Noise = reset_idle_noise
            self.SEInstKeys.append(reset_idle_noise)
            self.SE_instructions[reset_idle_noise] = CircuitInstruction(
                Noise.IDLE.instruction,
                gate_args=[self.noise_config.noises[Noise.IDLE]]
            )
        for index in SE_indices:
            if index <= 0:
                raise ValueError(f'SE {index=} should be greater than 0.')
            CX_op = InstKey(index, Gate.CX)
            CX_Cell = GATEInstCell(CX_op)
            self.SEInstKeys.append(CX_op)
            self.SEGateCells.append(CX_Cell)
            self.SE_CX_ops.add(CX_op)
            self.SE_instructions[CX_op] = CircuitInstruction(Gate.CX.instruction)
            if Noise.TQGATE in self.noise_config.noises:
                CX_noise = InstKey(index, Noise.TQGATE)
                CX_Cell.GATE_Noise = CX_noise
                self.SEInstKeys.append(CX_noise)
                self.GateNoiseSEInst_2_GateSEInst[CX_noise] = CX_op
                self.SE_instructions[CX_noise] = CircuitInstruction(
                    Noise.TQGATE.instruction,
                    gate_args=[self.noise_config.noises[Noise.TQGATE]])
            if Noise.IDLE in self.noise_config.noises:
                CX_idle_noise = InstKey(index, Noise.IDLE)
                CX_Cell.IDLE_Noise = CX_idle_noise
                self.SEInstKeys.append(CX_idle_noise)
                self.SE_instructions[CX_idle_noise] = CircuitInstruction(
                    Noise.IDLE.instruction,
                    gate_args=[self.noise_config.noises[Noise.IDLE]]
                )
        if Noise.Z_MEAS in self.noise_config.noises:
            z_meas_noise = InstKey(-1, Noise.Z_MEAS)
            self.SEMeasCell.Z_MEAS_Noise = z_meas_noise
            self.SEInstKeys.append(z_meas_noise)
            self.GateNoiseSEInst_2_GateSEInst[z_meas_noise] = InstKey(
                -1, Gate.MZ
            )
            self.SE_instructions[z_meas_noise] = CircuitInstruction(
                    Noise.Z_MEAS.instruction,
                    gate_args=[self.noise_config.noises[Noise.Z_MEAS]])
        z_meas_op = InstKey(-1, Gate.MZ)
        self.SEMeasCell.Z_MEAS_Inst = z_meas_op
        self.SEInstKeys.append(z_meas_op)
        self.SE_instructions[z_meas_op] = CircuitInstruction(Gate.MZ.instruction)
        if Noise.X_MEAS in self.noise_config.noises:
            x_meas_noise = InstKey(-1, Noise.X_MEAS)
            self.SEMeasCell.X_MEAS_Noise = x_meas_noise
            self.SEInstKeys.append(x_meas_noise)
            self.GateNoiseSEInst_2_GateSEInst[x_meas_noise] = InstKey(
                -1, Gate.MX
            )
            self.SE_instructions[x_meas_noise] = CircuitInstruction(
                    Noise.X_MEAS.instruction,
                    gate_args=[self.noise_config.noises[Noise.X_MEAS]])
        x_meas_op = InstKey(-1, Gate.MX)
        self.SEMeasCell.X_MEAS_Inst = x_meas_op
        self.SEInstKeys.append(x_meas_op)
        self.SE_instructions[x_meas_op] = CircuitInstruction(Gate.MX.instruction)
        if Noise.IDLE in self.noise_config.noises:
            meas_idle_noise = InstKey(-1, Noise.IDLE)
            self.SEMeasCell.IDLE_Noise = meas_idle_noise
            self.SEInstKeys.append(meas_idle_noise)
            self.SE_instructions[meas_idle_noise] = CircuitInstruction(
                Noise.IDLE.instruction,
                gate_args=[self.noise_config.noises[Noise.IDLE]]
            )
        # compile the gates (no noise for now) using ancilla qubits
        for qubit in self.code.qubits:
            if qubit.qubit_type == QubitType.X_ancilla:
                self.SE_instructions[x_reset_op] += CircuitInstruction(
                    Gate.RX.instruction, [qubit.circuit_id]
                )
                self.SE_instructions[x_meas_op] += CircuitInstruction(
                    Gate.MX.instruction, [qubit.circuit_id]
                )
                for CX_op in self.SE_CX_ops:
                    if CX_op.index not in qubit.SEIndex_2_neighbor_qubit:
                        continue
                    else:
                        target = qubit.SEIndex_2_neighbor_qubit[CX_op.index]
                    self.SE_instructions[CX_op] += CircuitInstruction(
                        Gate.CX.instruction, [qubit.circuit_id, target.circuit_id]
                    )
            elif qubit.qubit_type == QubitType.Z_ancilla:
                self.SE_instructions[z_reset_op] += CircuitInstruction(
                    Gate.RZ.instruction, [qubit.circuit_id]
                )
                self.SE_instructions[z_meas_op] += CircuitInstruction(
                    Gate.MZ.instruction, [qubit.circuit_id]
                )
                for CX_op in self.SE_CX_ops:
                    if CX_op.index not in qubit.SEIndex_2_neighbor_qubit:
                        continue
                    else:
                        control = qubit.SEIndex_2_neighbor_qubit[CX_op.index]
                    self.SE_instructions[CX_op] += CircuitInstruction(
                        Gate.CX.instruction, [control.circuit_id, qubit.circuit_id]
                    )
        # compile the noise channels 
        active_qubits: set[int] = set([qubit.circuit_id for 
                                       qubit in self.code.qubits])
        used_qubits: set[int] = set()
        for SEInst in self.SEInstKeys:
            if isinstance(SEInst.operation, Gate):
                used_qubits |= set(self.SE_instructions[SEInst].qubit_target_values())
            elif isinstance(SEInst.operation, Noise):
                if SEInst in self.GateNoiseSEInst_2_GateSEInst:
                    GateInst = self.GateNoiseSEInst_2_GateSEInst[SEInst]
                    self.SE_instructions[SEInst] \
                            += self.SE_instructions[GateInst].qubit_target_values()
                elif SEInst.operation == Noise.IDLE:
                    self.SE_instructions[SEInst] \
                            += list(active_qubits-used_qubits) 
                    used_qubits.clear()
                else:
                    raise ValueError(f'{SEInst=} ill-defined in the circuit.')
        # Generate the CellInst Singlets
        self.SECircCells_manager.set_CircCell(
            0, self._generate_circInstCell_from_InstkeyCell(self.SEResetCell)
        )
        self.SECircCells_manager.set_CircCell(
            -1, self._generate_circInstCell_from_InstkeyCell(self.SEMeasCell)
        )
        for gatecell in self.SEGateCells:
            self.SECircCells_manager.set_CircCell(
                gatecell.index, 
                self._generate_circInstCell_from_InstkeyCell(gatecell)
            )
        # SE compilation complete
        self.SE_compilation_flag = True
    
    def _compile_special_Cells(self):
        self.special_cells_for_logic_ops.clear()
        active_qubits = set([qubit.circuit_id for qubit in self.code.qubits])
        # RESETCELL for logical RZ RX RZ_DET_PERF RX_DET_PERF
        lrz_cell = CircInstCellSinglet.from_type(CellType.RESETInstCell)
        lrx_cell = CircInstCellSinglet.from_type(CellType.RESETInstCell)
        lrz_detperf_cell = CircInstCellSinglet.from_type(CellType.RESETInstCell)
        lrx_detperf_cell = CircInstCellSinglet.from_type(CellType.RESETInstCell)
        data_anc_rz_circInst = CircuitInstruction(Gate.RZ.instruction)
        data_rz_circInst = CircuitInstruction(Gate.RZ.instruction)
        data_anc_rz_circInst += self.SE_instructions[InstKey(0,Gate.RZ)]
        data_anc_rx_circInst = CircuitInstruction(Gate.RX.instruction)
        data_rx_circInst = CircuitInstruction(Gate.RX.instruction)
        data_anc_rx_circInst += self.SE_instructions[InstKey(0,Gate.RX)]
        for qubit in self.code.qubits:
            if qubit.qubit_type == QubitType.Data:
                data_rx_circInst += [qubit.circuit_id]
                data_rz_circInst += [qubit.circuit_id]
        data_anc_rz_circInst += data_rz_circInst
        data_anc_rx_circInst += data_rx_circInst
        lrz_cell.abstractInst_2_CircInst[AbstractInst.Z_RESET] = data_anc_rz_circInst
        lrz_cell.abstractInst_2_CircInst[AbstractInst.X_RESET] = \
                                                self.SE_instructions[InstKey(0, Gate.RX)]
        lrx_cell.abstractInst_2_CircInst[AbstractInst.X_RESET] = data_anc_rx_circInst
        lrx_cell.abstractInst_2_CircInst[AbstractInst.Z_RESET] = \
                                                self.SE_instructions[InstKey(0, Gate.RZ)]
        lrz_detperf_cell.abstractInst_2_CircInst[AbstractInst.Z_RESET] = \
                                                            data_rz_circInst.deepcopy()
        lrx_detperf_cell.abstractInst_2_CircInst[AbstractInst.X_RESET] = \
                                                            data_rx_circInst.deepcopy()
        if Noise.Z_RESET in self.noise_config.noises:
            lrz_noise_circInst = CircuitInstruction(Noise.Z_RESET.instruction,
                                                   data_anc_rz_circInst.targets_copy(),
                                                   [self.noise_config.noises[Noise.Z_RESET]])
            lrz_cell.abstractInst_2_CircInst[AbstractInst.Z_RESET_NOISE] = lrz_noise_circInst
            lrx_cell.abstractInst_2_CircInst[AbstractInst.Z_RESET_NOISE] = \
                                                self.SE_instructions[InstKey(0, Noise.Z_RESET)]
        if Noise.X_RESET in self.noise_config.noises:
            lrx_noise_circInst = CircuitInstruction(Noise.X_RESET.instruction,
                                                   data_anc_rx_circInst.targets_copy(),
                                                   [self.noise_config.noises[Noise.X_RESET]])
            lrz_cell.abstractInst_2_CircInst[AbstractInst.X_RESET_NOISE] = \
                                                self.SE_instructions[InstKey(0, Noise.X_RESET)]
            lrx_cell.abstractInst_2_CircInst[AbstractInst.X_RESET_NOISE] = lrx_noise_circInst
        self.special_cells_for_logic_ops[(LogicalOp.LRZ,CellType.RESETInstCell)] = lrz_cell
        self.special_cells_for_logic_ops[(LogicalOp.LRX,CellType.RESETInstCell)] = lrx_cell
        self.special_cells_for_logic_ops[(LogicalOp.LRZ_DET_PERF,CellType.RESETInstCell)] = lrz_detperf_cell
        self.special_cells_for_logic_ops[(LogicalOp.LRX_DET_PERF,CellType.RESETInstCell)] = lrx_detperf_cell
        # MEASCELL for MX MZ 
        lmz_cell = CircInstCellSinglet.from_type(CellType.MEASInstCell)
        lmx_cell = CircInstCellSinglet.from_type(CellType.MEASInstCell)
        anc_data_mz_circInst = CircuitInstruction(Gate.MZ.instruction)
        anc_data_mx_circInst = CircuitInstruction(Gate.MX.instruction)
        anc_data_mz_circInst += self.SE_instructions[InstKey(-1,Gate.MZ)]
        anc_data_mx_circInst += self.SE_instructions[InstKey(-1,Gate.MX)]
        for qubit in self.code.qubits:
            if qubit.qubit_type == QubitType.Data:
                anc_data_mx_circInst += [qubit.circuit_id]
                anc_data_mz_circInst += [qubit.circuit_id]
        lmz_cell.abstractInst_2_CircInst[AbstractInst.Z_MEAS] = anc_data_mz_circInst
        lmz_cell.abstractInst_2_CircInst[AbstractInst.X_MEAS] = \
                                            self.SE_instructions[InstKey(-1,Gate.MX)]
        lmx_cell.abstractInst_2_CircInst[AbstractInst.X_MEAS] = anc_data_mx_circInst
        lmx_cell.abstractInst_2_CircInst[AbstractInst.Z_MEAS] = \
                                            self.SE_instructions[InstKey(-1,Gate.MZ)]
        if Noise.Z_MEAS in self.noise_config.noises:
            anc_data_mz_noise_circInst = CircuitInstruction(
                                            Noise.Z_MEAS.instruction,
                                            anc_data_mz_circInst.targets_copy(),
                                            [self.noise_config.noises[Noise.Z_MEAS]],
                                            tag=anc_data_mz_circInst.tag)
            lmz_cell.abstractInst_2_CircInst[AbstractInst.Z_MEAS_NOISE] = \
                                                    anc_data_mz_noise_circInst
            lmx_cell.abstractInst_2_CircInst[AbstractInst.Z_MEAS_NOISE] = \
                                    self.SE_instructions[InstKey(-1,Noise.Z_MEAS)]
        if Noise.X_MEAS in self.noise_config.noises:
            anc_data_mx_noise_circInst = CircuitInstruction(
                                            Noise.X_MEAS.instruction,
                                            anc_data_mx_circInst.targets_copy(),
                                            [self.noise_config.noises[Noise.X_MEAS]],
                                            tag=anc_data_mx_circInst.tag)
            lmz_cell.abstractInst_2_CircInst[AbstractInst.X_MEAS_NOISE] = \
                                    self.SE_instructions[InstKey(-1,Noise.X_MEAS)]
            lmx_cell.abstractInst_2_CircInst[AbstractInst.X_MEAS_NOISE] = \
                                                    anc_data_mx_noise_circInst
        self.special_cells_for_logic_ops[(LogicalOp.LMZ,CellType.MEASInstCell)] = lmz_cell
        self.special_cells_for_logic_ops[(LogicalOp.LMX,CellType.MEASInstCell)] = lmx_cell   
        # MPP instruction for all stabilizers
        mpp_all_stabs = CircuitInstruction('MPP')
        for qubit in self.code.qubits:
            mpp_targets: list[stim.DemTarget] = []
            mpp_group_index = 0
            if qubit.qubit_type == QubitType.X_ancilla:
                for neighbor in qubit.SEIndex_2_neighbor_qubit.values():
                    mpp_targets.append(stim.target_x(neighbor.circuit_id))
                    mpp_targets.append(stim.target_combiner())
            elif qubit.qubit_type == QubitType.Z_ancilla:
                for neighbor in qubit.SEIndex_2_neighbor_qubit.values():
                    mpp_targets.append(stim.target_z(neighbor.circuit_id))
                    mpp_targets.append(stim.target_combiner())
            else:
                continue
            mpp_targets.pop()
            new_mpp_circInst = CircuitInstruction('MPP', mpp_targets,
                                        anc_id_2_mpp_group_index={qubit.circuit_id:
                                                                  mpp_group_index})
            mpp_group_index += 1
            mpp_all_stabs += new_mpp_circInst
        mpp_cell = CircInstCellSinglet.from_type(CellType.MEASInstCell)
        mpp_cell.abstractInst_2_CircInst[AbstractInst.MPP] = mpp_all_stabs
        mpp_logic_z = CircuitInstruction('MPP')
        mpp_logic_z += mpp_all_stabs
        mpp_logic_x = CircuitInstruction('MPP')
        mpp_logic_x += mpp_all_stabs
        if self.logical_id is not None:
            if self.logical_id >= 0:
                raise ValueError(f'logical id {self.logical_id} should be '
                                 'a negative integer')
            else:
                logic_id_2_mpp_index = {self.logical_id:0}
        else:
            logic_id_2_mpp_index = {}
        logic_z_targets = []
        logic_x_targets = []
        for qubit in self.code.logical_Z_operator:
            logic_z_targets.append(stim.target_z(qubit.circuit_id))
            logic_z_targets.append(stim.target_combiner())
        logic_z_targets.pop()
        mpp_logic_z += CircuitInstruction('MPP',logic_z_targets,
                                anc_id_2_mpp_group_index=logic_id_2_mpp_index) 
        for qubit in self.code.logical_X_operator:
            logic_x_targets.append(stim.target_x(qubit.circuit_id))
            logic_x_targets.append(stim.target_combiner())
        logic_x_targets.pop()
        mpp_logic_x += CircuitInstruction('MPP',logic_x_targets,
                                anc_id_2_mpp_group_index=logic_id_2_mpp_index)
        mpp_logic_z_cell = CircInstCellSinglet.from_type(CellType.MEASInstCell)
        mpp_logic_z_cell.abstractInst_2_CircInst[AbstractInst.MPP] = mpp_logic_z
        mpp_logic_x_cell = CircInstCellSinglet.from_type(CellType.MEASInstCell)
        mpp_logic_x_cell.abstractInst_2_CircInst[AbstractInst.MPP] = mpp_logic_x
        self.special_cells_for_logic_ops[(LogicalOp.LRX_DET_PERF,
                                          CellType.MEASInstCell)] \
                                                    = mpp_cell
        self.special_cells_for_logic_ops[(LogicalOp.LRZ_DET_PERF,
                                          CellType.MEASInstCell)] \
                                                    = mpp_cell
        self.special_cells_for_logic_ops[(LogicalOp.LMZ_PERF,
                                          CellType.MEASInstCell)] \
                                                    = mpp_logic_z_cell
        self.special_cells_for_logic_ops[(LogicalOp.LMX_PERF,
                                          CellType.MEASInstCell)] \
                                                    = mpp_logic_x_cell
        self.special_cells_for_logic_ops[(LogicalOp.LR_PERF,
                                          CellType.MEASInstCell)] \
                                                    = mpp_cell
        self.special_cells_for_logic_ops[(LogicalOp.LM_PERF,
                                          CellType.MEASInstCell)] \
                                                    = mpp_cell
        pass

    @staticmethod
    def _generate_patch_id(code: CSS_code_2D) -> None:
        pass

    def _generate_circInstCell_from_InstkeyCell(self,
                        InstkeyCell: Union[RESETInstCell, 
                        MEASInstCell, GATEInstCell]) -> CircInstCellSinglet:
        circInstCell = CircInstCellSinglet().from_InstCell(InstkeyCell)
        absInst_2_instKey = InstkeyCell.abstractInst_2_InstKey
        for absInst in absInst_2_instKey:
            if absInst_2_instKey[absInst] is None:
                continue
            circInstCell.abstractInst_2_CircInst[absInst] \
                = self.SE_instructions[absInst_2_instKey[absInst]]
        return circInstCell

    @staticmethod
    def intersection(patch_1: CodePatch, patch_2: CodePatch) -> bool:
        x_points_1 = set(range(patch_1.patch_boundaries['x_min'],
                               patch_1.patch_boundaries['x_max']+1))
        x_points_2 = set(range(patch_2.patch_boundaries['x_min'],
                               patch_2.patch_boundaries['x_max']+1))
        y_points_1 = set(range(patch_1.patch_boundaries['y_min'],
                               patch_1.patch_boundaries['y_max']+1))
        y_points_2 = set(range(patch_2.patch_boundaries['y_min'],
                               patch_2.patch_boundaries['y_max']+1))
        if x_points_1.intersection(x_points_2) and \
           y_points_1.intersection(y_points_2):
            return True
        else:
            return False
    
    def get_SEGateCell_from_index(self, index: int) -> Optional[GATEInstCell]:
        for GateCell in self.SEGateCells:
            if GateCell.index == index:
                return GateCell
        return None

    @property
    def patch_boundaries(self) -> dict[Literal['x_min','x_max',
                                               'y_min','y_max'],int]:
        return {'x_min':self.code.qubit_lattice.x_min, 
                'x_max':self.code.qubit_lattice.x_max, 
                'y_min':self.code.qubit_lattice.y_min, 
                'y_max':self.code.qubit_lattice.y_max}
    
    @property
    def X_stab_ids(self) -> list[int]:
        ret_list = []
        for qubit in self.code.qubits:
            if qubit.qubit_type == QubitType.X_ancilla:
                ret_list.append(qubit.circuit_id)
        return ret_list
    
    @property
    def Z_stab_ids(self) -> list[int]:
        ret_list = []
        for qubit in self.code.qubits:
            if qubit.qubit_type == QubitType.Z_ancilla:
                ret_list.append(qubit.circuit_id)
        return ret_list
    
    @property
    def data_qubit_ids(self) -> list[int]:
        ret_list = []
        for qubit in self.code.qubits:
            if qubit.qubit_type == QubitType.Data:
                ret_list.append(qubit.circuit_id)
        return sorted(ret_list)
    
    @property
    def SEGateIndices(self) -> list[int]:
        return sorted([GateCell.index for GateCell in self.SEGateCells])

    def compile_SE_round_circuit(self) -> stim.Circuit:
        circuit = Circuit_helper()
        if self.id_assignment_flag:
            circuit.introduce_new_qubits([
                self.qubit_id_2_pos[id] for id in sorted(self.qubit_ids)
            ])
        else:
            pos_2_id = circuit.introduce_new_qubits(
                [qubit.position for qubit in self.code.qubits]
            )
            self._assign_qubit_circuit_ids(pos_2_id)
        self._compile_SE_instructions()
        if self.SE_compilation_flag:
            print('Recompiling SE instructions.')
        for SEInst in self.SEInstKeys:
            circuit.append_Instruction(self.SE_instructions[SEInst])
            circuit.tick
        return circuit.circuit

    def draw_SE(self, svg_filename: str):
        circ = self.compile_SE_round_circuit()
        with open(svg_filename,'w') as file:
            out = circ.diagram('timeslice-svg')
            file.write(out.__str__())

    def update_SE_cells(self):
        '''
        Update the SE cells for the following SE round according to the
        (patch-level) logical operation `LogicalOp`. 
        '''
        # reset 
        self.SECircCells_manager.reset()
        if self.logical_op_following_se == LogicalOp.LSE:
            return
        elif self.logical_op_following_se is None \
             or self.logical_op_following_se == LogicalOp.LNONE:
            self.SECircCells_manager.empty_set()
            return
        # update reset circcell for patch reset
        if self.logical_op_following_se in [LogicalOp.LMX_PERF, LogicalOp.LR_PERF,
                                            LogicalOp.LMZ_PERF, LogicalOp.LM_PERF]:
            self.SECircCells_manager.index_2_CircCell[0] \
                = CircInstCellSinglet.from_type(CellType.RESETInstCell)
        elif self.logical_op_following_se in [LogicalOp.LRX, LogicalOp.LRZ,
                                            LogicalOp.LRZ_DET_PERF,
                                            LogicalOp.LRX_DET_PERF]:
            self.SECircCells_manager.index_2_CircCell[0] \
                = self.special_cells_for_logic_ops[(self.logical_op_following_se,
                                                    CellType.RESETInstCell)]
        # update meas circcell for patch measurements
        if self.logical_op_following_se in [LogicalOp.LMZ, LogicalOp.LMX,
                                            LogicalOp.LMZ_PERF, 
                                            LogicalOp.LMX_PERF,
                                            LogicalOp.LM_PERF,
                                            LogicalOp.LR_PERF,
                                            LogicalOp.LRZ_DET_PERF,
                                            LogicalOp.LRX_DET_PERF]:
            self.SECircCells_manager.index_2_CircCell[-1] \
                = self.special_cells_for_logic_ops[(self.logical_op_following_se,
                                                    CellType.MEASInstCell)]
        # update CX gate circcells in SE
        # For MPP reset or measurements, CX gate cells are set to be empty.
        if self.logical_op_following_se in [LogicalOp.LMZ_PERF, LogicalOp.LMX_PERF,
                                            LogicalOp.LR_PERF, LogicalOp.LM_PERF,
                                            LogicalOp.LRZ_DET_PERF,
                                            LogicalOp.LRX_DET_PERF]:
            for index in self.SECircCells_manager.index_2_CircCell:
                if index != 0 and index != -1:
                    self.SECircCells_manager.index_2_CircCell[index] \
                        = CircInstCellSinglet.from_type(CellType.GATEInstCell)
                    
    def update_active_status_after_reset(self):
        if self.logical_op_following_se in [LogicalOp.LRZ, LogicalOp.LRX,
                                            LogicalOp.LRZ_DET_PERF,
                                            LogicalOp.LRX_DET_PERF,
                                            LogicalOp.LR_PERF]:
            if self.active == False:
                self.active = True
            else:
                raise RuntimeError('Trying to logically reset an active patch')
    
    def update_active_status_after_meas(self):
        if self.logical_op_following_se in [LogicalOp.LMZ, LogicalOp.LMX,
                                    LogicalOp.LMZ_PERF,
                                    LogicalOp.LMX_PERF,
                                    LogicalOp.LM_PERF]:
            if self.active == True:
                self.active = False
            else:
                raise RuntimeError('Trying to logically measure an inactive patch')

    def clear_current_logic_op(self):
        self.logical_op_following_se = None


class MultiPatchCircuit:
    '''
    Implementation of Synchronized circuits on multiple code patches. 
    Synchronization means reset and meas steps of all patches occur 
    at the same time. 
    '''
    def __init__(self):
        self.code_patches: list[CodePatch] = []
        self.circuit = Circuit_helper()
        self.time = 0
        self.time_unit = 2
        self.time_half_unit = 1
        self.qubit_id_time_2_meas_id: dict[tuple[int,int],int] = {}
        self.qubit_id_2_patch_ids: dict[int,list[int]] = {}
        self.dummy_qubits: set[int] = set()
        self.qubit_id_2_pos: dict[int,tuple[int,...]] = {}
        self.SEGateIndices: list[int] = []
        self.SEIndex_2_circInstCells: dict[int,CircInstCellStrictBuffer] = {}
        self.noise_config: NoiseConfig = NoiseConfig()
        self.SE_compilation_flag = False
        self.qubit_id_2_meas_records: dict[int,QubitMeas_recorder] = {}
        self.used_ports: set[tuple[int,int]] = set()    #(patch_id, port)
        self.inter_cnot_register: dict[tuple[int,int],
                                       tuple[int,int]] = {}  
        # bus patch_id to core patch_id 
        self.inter_reversed_cnot_register: dict[int,int] = {}
        pass

    @property
    def current_tick(self) -> int:
        return self.circuit.circuit.num_ticks

    def add_code_patch(self, new_code_patch: CodePatch):
        '''
        New code patch is allowed to intersect with old code patches. 
        '''
        # for code_patch in self.code_patches:
        #     if CodePatch.intersection(code_patch,new_code_patch):
        #         raise ValueError(f'new code patch (with '
        #                          f'{new_code_patch.patch_boundaries=}) intersects '
        #                          f'nontrivially with a previously added patch '
        #                          f'with {code_patch.patch_boundaries=}')
        if new_code_patch.patch_id is not None:
            raise ValueError(f'{new_code_patch=} is already assigned a patch id '
                             f'{new_code_patch.patch_id=}')
        new_code_patch.patch_id = len(self.code_patches)
        self.code_patches.append(new_code_patch)
        pos_2_id = self.circuit.introduce_new_qubits(sorted(new_code_patch.code.
                                                 position_2_qubit.keys()))
        for pos in pos_2_id:
            if pos_2_id[pos] in self.qubit_id_2_pos:
                if pos != self.qubit_id_2_pos[pos_2_id[pos]]:
                    raise RuntimeError(f'Trying to assign qubit {pos_2_id[pos]} '
                                       f'with a new {pos=} different from the '
                                       f'previously assigned one '
                                       f'{self.qubit_id_2_pos[pos_2_id[pos]]}')
            self.qubit_id_2_pos[pos_2_id[pos]] = pos
        new_code_patch._assign_qubit_circuit_ids(pos_2_id)
        new_code_patch.logical_id = -len(self.code_patches)
        new_code_patch.logical_flow = LogicalFlow(new_code_patch.logical_id)
        for qubit_id in pos_2_id.values():
            if qubit_id in self.qubit_id_2_patch_ids:
                self.qubit_id_2_patch_ids[qubit_id].append(len(self.code_patches)-1)
            else:
                self.qubit_id_2_patch_ids[qubit_id] = [len(self.code_patches)-1]
        self.qubit_id_2_meas_records[new_code_patch.logical_id] = len(self.code_patches)-1
        for qubit in new_code_patch.code.qubits:
            self.qubit_id_2_meas_records[qubit.circuit_id] = QubitMeas_recorder()
        self.qubit_id_2_meas_records[new_code_patch.logical_id] = QubitMeas_recorder()
        new_code_patch.noise_config = self.noise_config
        new_code_patch._compile_SE_instructions()
        new_code_patch._compile_special_Cells()
        self.SEGateIndices = sorted(set(self.SEGateIndices)\
                                    .union(new_code_patch.SEGateIndices))
    
    def _SE_compile(self):
        # Reset and Meas:
        resetCircBuffer = CircInstCellStrictBuffer()
        measCircBuffer = CircInstCellStrictBuffer()
        resetCircBuffer.extend([code_patch.SECircCells_manager.index_2_CircCell[0]
                                for code_patch in self.code_patches])
        measCircBuffer.extend([code_patch.SECircCells_manager.index_2_CircCell[-1]
                               for code_patch in self.code_patches])
        self.SEIndex_2_circInstCells[0] = resetCircBuffer
        self.SEIndex_2_circInstCells[-1] = measCircBuffer
        for gateIndex in self.SEGateIndices:
            gateCircBuffer = CircInstCellStrictBuffer()
            gate_cells = []
            for code_patch in self.code_patches:
                if gateIndex in code_patch.SECircCells_manager.index_2_CircCell:
                    gate_cells.append(code_patch.SECircCells_manager
                                      .index_2_CircCell[gateIndex])
                else:
                    if Noise.IDLE in code_patch.noise_config.noises:
                        idling_cell = CircInstCellSinglet().from_InstCell(GATEInstCell())
                        idling_cell.abstractInst_2_CircInst[AbstractInst.IDLE_NOISE]\
                            = CircuitInstruction(
                                Noise.IDLE.instruction,
                                targets=code_patch.qubit_ids,
                                gate_args=code_patch.noise_config.noises[Noise.IDLE]
                            )
                        gate_cells.append(idling_cell)
            gateCircBuffer.extend(gate_cells)
            self.SEIndex_2_circInstCells[gateIndex] = gateCircBuffer
        self.SE_compilation_flag = True
    
    def _commit_cellInstBuffer(self, cellInstBuffer: CircInstCellStrictBuffer):
        if cellInstBuffer.type == CellType.RESETInstCell:
            time_ordered_absinsts = RESETInstCell().time_ordered_abstractInsts
        elif cellInstBuffer.type == CellType.GATEInstCell:
            time_ordered_absinsts = GATEInstCell().time_ordered_abstractInsts
        else:
            assert cellInstBuffer.type == CellType.MEASInstCell
            time_ordered_absinsts = MEASInstCell().time_ordered_abstractInsts
        for absInst in time_ordered_absinsts:
            circInst = cellInstBuffer.circInstcellbuff.abstractInst_2_CircInst[absInst]
            if circInst is None or is_instruction_empty(circInst):
                continue
            if absInst in [AbstractInst.Z_MEAS, AbstractInst.X_MEAS,
                           AbstractInst.MPP]:
                current_meas_id = self.circuit.num_measurements
                self.circuit.append_Instruction(circInst)
                new_meas_ids = list(range(current_meas_id,self.circuit.num_measurements))
                if circInst.name in ['M','MZ','MX']:
                    for target, meas_id in zip(circInst.targets_copy(),new_meas_ids):
                        if circInst.name in ['M','MZ']:
                            self.qubit_id_2_meas_records[target.qubit_value]\
                                .time_basis_2_meas_id[(self.time,Pauli.Z)] = meas_id
                        else:
                            self.qubit_id_2_meas_records[target.qubit_value]\
                                .time_basis_2_meas_id[(self.time,Pauli.X)] = meas_id
                elif circInst.name in ['MPP']:
                    if len(new_meas_ids) != len(circInst.anc_id_2_mpp_group_index):
                        raise ValueError('either not every mpp meas instance is accounted for '
                                         'or mpp measurements are redundantly accounted for')
                    for qubit_id in circInst.anc_id_2_mpp_group_index:
                        mpp_term_index = circInst.anc_id_2_mpp_group_index[qubit_id]
                        mpp_term_type = mpp_type(circInst.target_groups()[mpp_term_index])
                        if mpp_term_type is None or mpp_term_type == Pauli.Y:
                            raise NotImplementedError(f'MPP term'
                                        f'{circInst.target_groups()[mpp_term_index]}'
                                        f'should be pure Pauli X or Z MPP')
                        self.qubit_id_2_meas_records[qubit_id]\
                            .time_basis_2_meas_id[(self.time, mpp_term_type)] \
                            = new_meas_ids[mpp_term_index]
                        # elif mpp_term_type == Pauli.X:
                        #     self.qubit_id_2_meas_records[qubit_id]\
                        #         .time_2_x_meas_id[self.time] = new_meas_ids[mpp_term_index]
                        # else:
                        #     assert mpp_term_type == Pauli.Z
                        #     self.qubit_id_2_meas_records[qubit_id]\
                        #         .time_2_z_meas_id[self.time] = new_meas_ids[mpp_term_index]
            else:
                self.circuit.append_Instruction(circInst)
            self.circuit.tick

    def set_detectors(self):
        '''
        Detectors immediately after measurement at TIME = self.time. 
        Each detector is composed of a measurement at (TIME-1) and 
        a combination of measurements at TIME.  
        Detector corresponding to an ancilla qubit (associated with a
        stabilizer) is constructed by accessing the StabFlow of the qubit.
        If StabFlow.flow_from is None: No detector;
        Else: Detector = MEAS(StabFlow.flow_from) + MEAS(StabFlow.flow_to)
        '''
        for code_patch in self.code_patches:
            # if len(code_patch.code.logical_X_operator) == 6 \
            #     and len(code_patch.code.logical_Z_operator) == 6:
            #     continue
            if code_patch.active == False:
                continue
            for stab_id in code_patch.stab_id_2_stab_flow:
                stab_flow = code_patch.stab_id_2_stab_flow[stab_id]
                if stab_flow.flow_from is None:
                    continue
                flow_from_basis = stab_flow.flow_from[-1]
                meas_ids = []
                if not stab_flow.trans_init:
                    if (self.time-self.time_unit, flow_from_basis) in \
                        self.qubit_id_2_meas_records[stab_id].time_basis_2_meas_id:
                        meas_ids.append(self.qubit_id_2_meas_records[stab_id]
                                        .time_basis_2_meas_id[
                                            (self.time-self.time_unit, flow_from_basis)
                                        ])
                for anc_id, pauli_basis in stab_flow.flow_to:
                    # if (self.time, pauli_basis) not in self.qubit_id_2_meas_records[anc_id]\
                    #                                        .time_basis_2_meas_id:
                    #     print(stab_flow)
                    #     print(self.qubit_id_2_meas_records[anc_id].time_basis_2_meas_id)
                    meas_ids.append(self.qubit_id_2_meas_records[anc_id]
                                    .time_basis_2_meas_id[
                                        (self.time, pauli_basis)
                                    ])
                stab_qubit = code_patch.code.position_2_qubit[
                                code_patch.qubit_id_2_pos[stab_id]]
                if stab_qubit.tag_buffer is not None:
                    self.circuit.detector(
                        meas_ids,
                        code_patch.qubit_id_2_pos[stab_id] + (self.time,),
                        tag_buffer=stab_qubit.tag_buffer
                    )
                    stab_qubit.tag_buffer.cnot_clear()
                else:
                    self.circuit.detector(
                    meas_ids,
                    code_patch.qubit_id_2_pos[stab_id] + (self.time,)
                )
            if code_patch.logical_op_following_se in [LogicalOp.LMX,
                                                      LogicalOp.LMZ]:
                for stab_id in code_patch.stab_id_2_stab_flow:
                    stab_flow = code_patch.stab_id_2_stab_flow[stab_id]
                    if stab_flow.destruct_to is None:
                        continue
                    assert stab_flow.flow_to == set([(stab_id,stab_flow.pauli_default)])
                    meas_ids = []
                    meas_ids.append(self.qubit_id_2_meas_records[stab_id]
                                    .time_basis_2_meas_id[
                                        (self.time, stab_flow.pauli_default)
                                    ])
                    for qubit_id in stab_flow.destruct_to[0]:
                        meas_ids.append(self.qubit_id_2_meas_records[qubit_id]
                                        .time_basis_2_meas_id[
                                            (self.time, stab_flow.destruct_to[-1])
                                        ])
                    stab_qubit = code_patch.code.position_2_qubit[
                                code_patch.qubit_id_2_pos[stab_id]]
                    if stab_qubit.tag_buffer is not None:
                        self.circuit.detector(
                            meas_ids,
                            code_patch.qubit_id_2_pos[stab_id] 
                            + (self.time+self.time_half_unit,),
                            tag_buffer=stab_qubit.tag_buffer
                        )
                        stab_qubit.tag_buffer.cnot_clear()
                    else:
                        self.circuit.detector(
                            meas_ids,
                            code_patch.qubit_id_2_pos[stab_id] 
                            + (self.time+self.time_half_unit,)
                        )

    def do_SE_round(self):
        '''
        Perform a synchronized round of SE for each code patch.
        '''
        if not self.SE_compilation_flag:
            self._SE_compile()
        for se_index in sorted(self.SEIndex_2_circInstCells.keys()):
            if se_index == -1:
                continue
            self._commit_cellInstBuffer(self.SEIndex_2_circInstCells[se_index])
        self._commit_cellInstBuffer(self.SEIndex_2_circInstCells[-1])
        # TODO: setting up the detectors
        pass

    def commit_SE_round(self):
        for se_index in sorted(self.SEIndex_2_circInstCells.keys()):
            if se_index == -1:
                continue
            self._commit_cellInstBuffer(self.SEIndex_2_circInstCells[se_index])
        self._commit_cellInstBuffer(self.SEIndex_2_circInstCells[-1])
        # TODO: setting up the detectors
        pass

    @property
    def stim_circuit(self) -> stim.Circuit:
        return self.circuit.circuit
    
    def clear_circuit(self):
        self.time = 0
        self.circuit.circuit.clear()
    
    def set_logic_op(self, patch_id: int, logic_op: LogicalOp):
        '''
        Set the logical operation for the next SE round. Raise Exception 
        if the logical operation is already set. 
        '''
        if self.code_patches[patch_id].logical_op_following_se is not None:
            raise ValueError(f'logical op on code patch {patch_id} already set to '
                          f'{self.code_patches[patch_id].logical_op_following_se}')
        self.code_patches[patch_id].logical_op_following_se = logic_op
    
    def set_logic_op_no_overwrite(self, patch_id: int, logic_op: LogicalOp):
        '''
        Attempt to set the logical operation for the next SE round. If the patch
        is already assigned a logical operation, then nothing happens. 
        
        :param patch_id: 
        :type patch_id: int
        :param logic_op: 
        :type logic_op: LogicalOp
        '''
        if self.code_patches[patch_id].logical_op_following_se is not None:
            return
        else:
            self.code_patches[patch_id].logical_op_following_se = logic_op
    
    def set_inter_cnot(self, patch_1_id: int, patch_2_id: int,
                       ctrl_target: Literal['CT','TC'] = 'CT',
                       port: Optional[int] = None):
        patch_1 = self.code_patches[patch_1_id]
        patch_2 = self.code_patches[patch_2_id]
        if port is not None:
            if not (isinstance(patch_1.code,ShuttleBus_code)) \
                or not (isinstance(patch_2.code,Rotated_surface_code)):
                raise TypeError(f'{patch_1=} should be a shuttle bus '
                                f'{patch_2=} should be a surface code')
        else:
            if not (isinstance(patch_1.code, Rotated_surface_code)) \
                or not (isinstance(patch_2.code, Rotated_surface_code)):
                raise TypeError(f'{patch_1=} should be a surface code '
                                f'{patch_2=} should be a surface code')
            port = 0
        if (patch_1_id, port) in self.used_ports:
            raise RuntimeError(f'{(patch_1_id, port)=} already occupied')
        self.used_ports.add((patch_1_id, port))
        if (patch_2_id, 0) in self.used_ports:
            raise RuntimeError(f'{(patch_2_id, 0)=} already occupied')
        self.used_ports.add((patch_2_id, 0))
        if ctrl_target == 'CT':
            self.inter_cnot_register[(patch_1_id, port)] = (patch_2_id, 0)
        else:
            assert ctrl_target == 'TC'
            self.inter_cnot_register[(patch_2_id, 0)] = (patch_1_id, port)         
    
    def set_inter_reversed_cnot(self, bus_id: int, core_id: int,
                                cnot_direction: Literal['CT','TC'] = 'TC'):
        bus_patch = self.code_patches[bus_id]
        if not isinstance(bus_patch.code, ShuttleBus_code):
            raise TypeError(f'{bus_patch.code=} should be a shuttle bus')
        if bus_patch.code.longBasis == Pauli.Z and cnot_direction != 'TC':
            raise TypeError(f'{bus_patch=} with a {Pauli.Z} basis should be the target.')
        if bus_patch.code.longBasis == Pauli.X and cnot_direction != 'CT':
            raise TypeError(f'{bus_patch=} with a {Pauli.X} basis should be the control.')
        core_patch = self.code_patches[core_id]
        if not isinstance(core_patch.code, Rotated_surface_code):
            raise TypeError(f'{core_patch.code=} should be a core')
        for i in range(bus_patch.code.length):
            if (bus_id, i) in self.used_ports:
                raise RuntimeError(f'Port {(bus_id,i)} occupied. CNOT assignment failed.')
            self.used_ports.add((bus_id,i))
        if (core_id, 0) in self.used_ports:
            raise RuntimeError(f'Port {(core_id,0)} occupied. CNOT assignment failed.')
        self.used_ports.add((core_id,0))
        self.inter_reversed_cnot_register[bus_id] = core_id

        pass

    def commit_inter_cnots(self):
        if len(self.inter_cnot_register) == 0 and len(self.inter_reversed_cnot_register) == 0:
            return
        cnot_cell = CircInstCellStrictBuffer(type=CellType.GATEInstCell)
        cnot_cells = []
        used_qubits = set()
        for ctrl_port in self.inter_cnot_register:
            ctrl_patch = self.code_patches[ctrl_port[0]]
            c_port = ctrl_port[-1]
            target_patch = self.code_patches[self.inter_cnot_register[ctrl_port][0]]
            t_port = self.inter_cnot_register[ctrl_port][-1]
            if isinstance(ctrl_patch.code, Rotated_surface_code) \
                and isinstance(target_patch.code, Rotated_surface_code):
                c_pos_2_t_pos = ctrl_patch.code.dock(target_patch.code)
                t_pos_2_c_pos = target_patch.code.dock(ctrl_patch.code)
            elif isinstance(ctrl_patch.code, ShuttleBus_code) \
                and isinstance(target_patch.code, Rotated_surface_code):
                c_pos_2_t_pos = ctrl_patch.code.dock(target_patch.code, c_port)
                t_pos_2_c_pos = {c_pos_2_t_pos[c_pos]: c_pos for c_pos in c_pos_2_t_pos}
            else:
                assert isinstance(ctrl_patch.code, Rotated_surface_code) \
                       and isinstance(target_patch.code, ShuttleBus_code)
                t_pos_2_c_pos = target_patch.code.dock(ctrl_patch.code, t_port)
                c_pos_2_t_pos = {t_pos_2_c_pos[t_pos]: t_pos for t_pos in t_pos_2_c_pos}
            cnot_small_cell = CircInstCellSinglet.from_type(CellType.GATEInstCell)
            cnot_targets = []
            for qubit in ctrl_patch.code.qubits:
                if qubit.qubit_type == QubitType.Data:
                    if qubit.position not in c_pos_2_t_pos:
                        continue
                    target_qubit = target_patch.code.position_2_qubit[
                        c_pos_2_t_pos[qubit.position]
                    ]
                    cnot_targets.append(qubit.circuit_id)
                    cnot_targets.append(target_qubit.circuit_id)
            cnot_small_cell.abstractInst_2_CircInst[AbstractInst.GATE] \
                = CircuitInstruction('CX', targets=cnot_targets)
            used_qubits.update(cnot_targets)
            if Noise.TQGATE in self.noise_config.noises:
                cnot_small_cell.abstractInst_2_CircInst[AbstractInst.GATE_NOISE] \
                    = CircuitInstruction(
                        Noise.TQGATE.instruction,
                        targets=cnot_targets,
                        gate_args=[self.noise_config.noises[Noise.TQGATE]]
                    )
            cnot_cells.append(cnot_small_cell)
            for qubit in ctrl_patch.code.qubits:
                if qubit.qubit_type == QubitType.X_ancilla:
                    if qubit.position not in c_pos_2_t_pos:
                        continue
                    t_qubit = target_patch.code.position_2_qubit[
                        c_pos_2_t_pos[qubit.position]
                    ]
                    ctrl_patch.stab_id_2_stab_flow[qubit.circuit_id].flow_to\
                        .symmetric_difference_update(
                            [(t_qubit.circuit_id, Pauli.X)]
                        )
            for qubit in target_patch.code.qubits:
                if qubit.qubit_type == QubitType.Z_ancilla:
                    if qubit.position not in t_pos_2_c_pos:
                        continue
                    c_qubit = ctrl_patch.code.position_2_qubit[
                        t_pos_2_c_pos[qubit.position]
                    ]
                    target_patch.stab_id_2_stab_flow[qubit.circuit_id].flow_to\
                        .symmetric_difference_update(
                            [(c_qubit.circuit_id, Pauli.Z)]
                        )
        for bus_id in self.inter_reversed_cnot_register:
            bus_patch = self.code_patches[bus_id]
            core_patch = self.code_patches[self.inter_reversed_cnot_register[bus_id]]
            assert isinstance(bus_patch.code, ShuttleBus_code)
            assert isinstance(core_patch.code, Rotated_surface_code)
            if bus_patch.code.longBasis == Pauli.Z:
                ctrl_patch = core_patch
                target_patch = bus_patch
                c_pos_2_t_pos = core_patch.code.reverse_dock(bus_patch.code)
                t_pos_2_c_pos = {c_pos_2_t_pos[c_pos]: c_pos for c_pos in c_pos_2_t_pos}
            else:
                assert bus_patch.code.longBasis == Pauli.X
                ctrl_patch = bus_patch
                target_patch = core_patch
                t_pos_2_c_pos = core_patch.code.reverse_dock(bus_patch.code)
                c_pos_2_t_pos = {t_pos_2_c_pos[t_pos]: t_pos for t_pos in t_pos_2_c_pos}
                raise NotImplementedError(f'X bus not implemented yet.') 
            cnot_small_cell = CircInstCellSinglet.from_type(CellType.GATEInstCell)
            cnot_targets = []
            for qubit in ctrl_patch.code.qubits:
                if qubit.qubit_type == QubitType.Data:
                    if qubit.position not in c_pos_2_t_pos:
                        continue
                    target_qubit = target_patch.code.position_2_qubit[
                        c_pos_2_t_pos[qubit.position]
                    ]
                    cnot_targets.append(qubit.circuit_id)
                    cnot_targets.append(target_qubit.circuit_id)
            cnot_small_cell.abstractInst_2_CircInst[AbstractInst.GATE] \
                = CircuitInstruction('CX', targets=cnot_targets)
            used_qubits.update(cnot_targets)
            if Noise.TQGATE in self.noise_config.noises:
                cnot_small_cell.abstractInst_2_CircInst[AbstractInst.GATE_NOISE] \
                    = CircuitInstruction(
                        Noise.TQGATE.instruction,
                        targets=cnot_targets,
                        gate_args=[self.noise_config.noises[Noise.TQGATE]]
                    )
            cnot_cells.append(cnot_small_cell)
            for qubit in ctrl_patch.code.qubits:
                if qubit.qubit_type == QubitType.X_ancilla:
                    if qubit.position not in c_pos_2_t_pos:
                        continue
                    t_qubit = target_patch.code.position_2_qubit[
                        c_pos_2_t_pos[qubit.position]
                    ]
                    ctrl_patch.stab_id_2_stab_flow[qubit.circuit_id].flow_to\
                        .symmetric_difference_update(
                            [(t_qubit.circuit_id, Pauli.X)]
                        )
            for qubit in target_patch.code.qubits:
                if qubit.qubit_type == QubitType.Z_ancilla:
                    if qubit.position not in t_pos_2_c_pos:
                        continue
                    c_qubit = ctrl_patch.code.position_2_qubit[
                        t_pos_2_c_pos[qubit.position]
                    ]
                    target_patch.stab_id_2_stab_flow[qubit.circuit_id].flow_to\
                        .symmetric_difference_update(
                            [(c_qubit.circuit_id, Pauli.Z)]
                        )
        cnot_cell.extend(cnot_cells)
        if Noise.IDLE in self.noise_config.noises:
            idling_qubits = set()
            for code_patch in self.code_patches:
                if code_patch.active == True:
                    idling_qubits.update(set(code_patch.data_qubit_ids).difference(used_qubits))
            cnot_cell.circInstcellbuff.abstractInst_2_CircInst[AbstractInst.IDLE_NOISE] \
                = CircuitInstruction(
                    Noise.IDLE.instruction,
                    targets=sorted(idling_qubits),
                    gate_args=[self.noise_config.noises[Noise.IDLE]]
                )
        self._commit_cellInstBuffer(cnot_cell)
        self.inter_cnot_register.clear()
        self.inter_reversed_cnot_register.clear()
        self.used_ports.clear()
        pass   
            
    def do_next_SE(self):
        reset_buffer = CircInstCellStrictBuffer(type=CellType.RESETInstCell)
        reset_cells = []
        for code_patch in self.code_patches:
            code_patch.update_SE_cells()
            reset_cells.append(code_patch.SECircCells_manager.index_2_CircCell[0])
            code_patch.update_active_status_after_reset()
        reset_buffer.extend(reset_cells)
        self.update_flow_pre_SE()
        self.SEIndex_2_circInstCells[0] = reset_buffer
        for index in sorted(self.SEGateIndices):
            gate_buffer = CircInstCellStrictBuffer(type=CellType.GATEInstCell)
            gate_cells = []
            for code_patch in self.code_patches:
                if code_patch.active == False \
                   or code_patch.logical_op_following_se is None \
                   or code_patch.logical_op_following_se == LogicalOp.LNONE:
                    continue
                if index in code_patch.SECircCells_manager.index_2_CircCell:
                    gate_cells.append(code_patch.SECircCells_manager.index_2_CircCell[index])
                else:
                    if Noise.IDLE in code_patch.noise_config:
                        idling_cell = CircInstCellSinglet().from_InstCell(GATEInstCell())
                        idling_cell.abstractInst_2_CircInst[AbstractInst.IDLE_NOISE]\
                            = CircuitInstruction(
                                Noise.IDLE.instruction,
                                targets=code_patch.qubit_ids,
                                gate_args=code_patch.noise_config.noises[Noise.IDLE]
                            )
                        gate_cells.append(idling_cell)
            gate_buffer.extend(gate_cells)
            self.SEIndex_2_circInstCells[index] = gate_buffer
        meas_buffer = CircInstCellStrictBuffer(type=CellType.MEASInstCell)
        meas_buffer.extend([code_patch.SECircCells_manager.index_2_CircCell[-1]
                            for code_patch in self.code_patches])
        self.SEIndex_2_circInstCells[-1] = meas_buffer
        self.commit_SE_round()
        self.set_detectors()
        for code_patch in self.code_patches:
            code_patch.update_active_status_after_meas()
            code_patch.clear_current_logic_op()
        self.update_flow_post_SE()
        pass

    @property
    def clock_move_forward(self):
        self.time += self.time_unit

    def draw(self, svg_filename: str, *, type = 'timeslice-svg',
             det_or_obs: Literal['DET','OBS'] = 'DET',
             tick_start: Optional[int] = None,
             tick_end: Optional[int] = None):
        '''
        **Args**

            type: recommend 'timeslice-svg' (default) or 'detslice-with-ops-svg'
        '''
        dets = [f'D{i}' for i in range(self.stim_circuit.num_detectors)]
        obs = [f'L{i}' for i in range(self.stim_circuit.num_observables)]
        tick_max = self.stim_circuit.num_ticks
        with open(svg_filename,'w') as file:
            if det_or_obs == 'DET':
                if tick_start is None and tick_end is None:
                    out = self.stim_circuit.diagram(type, filter_coords=dets)
                elif tick_end is None:
                    out = self.stim_circuit.diagram(type, filter_coords=dets, tick=range(tick_start,tick_max+1))
                elif tick_start is None:
                    out = self.stim_circuit.diagram(type, filter_coords=dets, tick=range(1,tick_end+1))
                else:
                    out = self.stim_circuit.diagram(type, filter_coords=dets, tick=range(tick_start,tick_end+1))
            else:
                assert det_or_obs == 'OBS'
                if tick_start is None and tick_end is None:
                    out = self.stim_circuit.diagram(type, filter_coords=obs)
                elif tick_end is None:
                    out = self.stim_circuit.diagram(type, filter_coords=obs, tick=range(tick_start,tick_max+1))
                elif tick_start is None:
                    out = self.stim_circuit.diagram(type, filter_coords=obs, tick=range(1,tick_end+1))
                else:
                    out = self.stim_circuit.diagram(type, filter_coords=obs, tick=range(tick_start,tick_end+1))
            file.write(out.__str__())
    
    def update_flow_pre_SE(self):
        for patch_id, _ in enumerate(self.code_patches):
            self._update_flow_pre_SE_for_given_patch(patch_id)
    
    def update_flow_post_SE(self):
        for patch_id, _ in enumerate(self.code_patches):
            self._update_flow_post_SE_for_given_patch(patch_id)

    def _update_flow_pre_SE_for_given_patch(self, patch_id: int):
        code_patch = self.code_patches[patch_id]
        if code_patch.logical_op_following_se in [LogicalOp.LRZ]:
            for qubit in code_patch.code.qubits:
                if qubit.qubit_type == QubitType.Z_ancilla:
                    code_patch.stab_id_2_stab_flow[qubit.circuit_id].stab_trans_init()
                elif qubit.qubit_type == QubitType.X_ancilla:
                    code_patch.stab_id_2_stab_flow[qubit.circuit_id].random_reset()
            code_patch.logical_flow.logical_pauli_reset(Pauli.Z)
        elif code_patch.logical_op_following_se == LogicalOp.LRX:
            for qubit in code_patch.code.qubits:
                if qubit.qubit_type == QubitType.Z_ancilla:
                    code_patch.stab_id_2_stab_flow[qubit.circuit_id].random_reset()
                elif qubit.qubit_type == QubitType.X_ancilla:
                    code_patch.stab_id_2_stab_flow[qubit.circuit_id].stab_trans_init()
            code_patch.logical_flow.logical_pauli_reset(Pauli.X)
        elif code_patch.logical_op_following_se in [LogicalOp.LRZ_DET_PERF,
                                              LogicalOp.LRX_DET_PERF]:
            for qubit in code_patch.code.qubits:
                if qubit.qubit_type in [QubitType.X_ancilla,
                                        QubitType.Z_ancilla]:
                    code_patch.stab_id_2_stab_flow[qubit.circuit_id].empty_reset()
            if code_patch.logical_op_following_se == LogicalOp.LRZ_DET_PERF:
                code_patch.logical_flow.logical_pauli_reset(Pauli.Z)
            else:
                code_patch.logical_flow.logical_pauli_reset(Pauli.X)
        elif code_patch.logical_op_following_se in [LogicalOp.LR_PERF]:
            for qubit in code_patch.code.qubits:
                if qubit.qubit_type in [QubitType.X_ancilla,
                                        QubitType.Z_ancilla]:
                    code_patch.stab_id_2_stab_flow[qubit.circuit_id].empty_reset()
        elif code_patch.logical_op_following_se == LogicalOp.LMX:
            for qubit in code_patch.code.qubits:
                if qubit.qubit_type == QubitType.X_ancilla:
                    assert code_patch.stab_id_2_stab_flow[qubit.circuit_id].flow_to \
                           == set([(qubit.circuit_id,Pauli.X)])
                    neighbors = [neighbor.circuit_id for neighbor 
                                 in qubit.SEIndex_2_neighbor_qubit.values()]
                    code_patch.stab_id_2_stab_flow[qubit.circuit_id].destructive_meas(
                        set(neighbors), Pauli.X
                    )
        elif code_patch.logical_op_following_se == LogicalOp.LMZ:
            for qubit in code_patch.code.qubits:
                if qubit.qubit_type == QubitType.Z_ancilla:
                    assert code_patch.stab_id_2_stab_flow[qubit.circuit_id].flow_to \
                           == set([(qubit.circuit_id,Pauli.Z)])
                    neighbors = [neighbor.circuit_id for neighbor 
                                 in qubit.SEIndex_2_neighbor_qubit.values()]
                    code_patch.stab_id_2_stab_flow[qubit.circuit_id].destructive_meas(
                        set(neighbors), Pauli.Z
                    )            
    
    def _update_flow_post_SE_for_given_patch(self, patch_id: int):
        code_patch = self.code_patches[patch_id]
        # empty reset if no longer active
        if code_patch.active == False:
            for qubit_id in code_patch.stab_id_2_stab_flow:
                code_patch.stab_id_2_stab_flow[qubit_id].empty_reset()
            code_patch.logical_flow.empty_reset()
            return
        # otherwise stab reset
        for qubit_id in code_patch.stab_id_2_stab_flow:
            code_patch.stab_id_2_stab_flow[qubit_id].stab_reset()
        pass

    def _get_qubit_supp_on_patch(self, qubit_supp: Iterable[tuple[int,Pauli]],
                                 patch_id: int) -> list[tuple[int,Pauli]]:
        restricted_supp = []
        for qubit_id, pauli in qubit_supp:
            if qubit_id in self.dummy_qubits:
                continue
            if patch_id in self.qubit_id_2_patch_ids[qubit_id]:
                restricted_supp.append((qubit_id, pauli))
        return restricted_supp


        