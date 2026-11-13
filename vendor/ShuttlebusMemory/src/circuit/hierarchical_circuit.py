from math import ceil
from copy import deepcopy

from high_level_utils import *
from lv1_codes import *
from lv1_utils import *
from lv1_circuits import Lv1Circuit


class HierarchicalCircuit(MultiPatchCircuit):

    def __init__(self, lv1_code: Lv1CSSCode,
                 inner_distance: int, 
                 bus_length: int,
                 bus_width: Optional[int] = None,
                 noise_config: Optional[NoiseConfig] = None,
                 external_cores: Optional[ExtCores] = None):
        super().__init__()
        if noise_config is not None:
            self.noise_config = noise_config
        self.inner_distance = inner_distance
        self.bus_length = bus_length
        if bus_width is None:
            self.bus_width = self.inner_distance
        else:
            raise NotImplementedError('Thickened buses not yet supported')
        self.data_code_template = Rotated_surface_code(self.inner_distance)
        self.anc_code_template = ShuttleBus_code(self.inner_distance,
                                                 self.bus_length, Pauli.Z)
        self.external_cores = external_cores
        self.lv1_code = lv1_code
        self.patch_2_lv1_qubit: dict[int,Lv1Qubit] = {}
        self.lv1_det_trackers: list[Lv1Det] = []
        self.lv1_circuit = Lv1Circuit()
        self.core_time_2_tick: dict[CoreInstTime,int] = {}
        self._add_all_code_patches()
        self._add_logical_dummy_qubits()
        self._initialize_lv1_circuit()

    def _initialize_lv1_circuit(self):
        self.lv1_circuit.initalize_lv1_circuit()
        
    def _add_all_code_patches(self):
        for lv1_qubit in self.lv1_code.distinct_pos_2_qubit.values():
            if lv1_qubit.core_type == CoreType.INTCORE:
                lv1_qubit.code_patch = CodePatch(Rotated_surface_code(self.inner_distance))
                lv1_qubit.code_patch.code.shift(
                    lv1_qubit.position[0]*(self.data_code_template.x_span+1),
                    lv1_qubit.position[1]*(self.data_code_template.y_span+1)
                )
            elif lv1_qubit.core_type == CoreType.ZBUS:
                lv1_qubit.code_patch = CodePatch(ShuttleBus_code(
                    self.inner_distance,
                    self.bus_length,
                    longBasis = Pauli.Z
                ))
                lv1_qubit.code_patch.code.shift(
                    lv1_qubit.position[0]*(self.anc_code_template.x_span+1),
                    lv1_qubit.position[1]*(self.anc_code_template.y_span+1)
                )
            else:
                assert lv1_qubit.core_type == CoreType.XBUS
                lv1_qubit.code_patch = CodePatch(ShuttleBus_code(
                    self.inner_distance,
                    self.bus_length,
                    longBasis = Pauli.X
                ))
                lv1_qubit.code_patch.code.shift(
                    lv1_qubit.position[0]*(self.anc_code_template.x_span+1),
                    lv1_qubit.position[1]*(self.anc_code_template.y_span+1)
                )
            self.add_code_patch(lv1_qubit.code_patch)
            self.patch_2_lv1_qubit[lv1_qubit.code_patch.patch_id] = lv1_qubit
            self.lv1_circuit.add_new_core(lv1_qubit)
            # self.lv1_circuit.add_new_patch(lv1_qubit.code_patch.patch_id,
            #                                    lv1_qubit.position)
        _, y_max = self.circuit.coordinate_span(1)
        # make sure that external cores are added at the bottom. 
        if self.external_cores is None:
            return
        external_bus_y_min, _ = self.external_cores.coord_span_of_core_types(
                                            1, [CoreType.XBUS, CoreType.ZBUS]) 
        for lv1_qubit in self.external_cores.pos_2_core.values():
            if lv1_qubit.core_type not in [CoreType.XBUS, CoreType.ZBUS]:
                continue
            pauli_basis = Pauli.X if lv1_qubit.core_type == CoreType.XBUS else Pauli.Z
            lv1_qubit.code_patch = CodePatch(ShuttleBus_code(self.external_cores.internal_core_size,
                                                length=self.external_cores.bus_length,
                                                longBasis=pauli_basis))
            lv1_qubit.code_patch.code.shift(
                lv1_qubit.position[0] * (lv1_qubit.code_patch.code.x_span+1),
                (lv1_qubit.position[1]-external_bus_y_min) * (lv1_qubit.code_patch.code.y_span+1) + y_max + 2
            )
            self.add_code_patch(lv1_qubit.code_patch)
            self.patch_2_lv1_qubit[lv1_qubit.code_patch.patch_id] = lv1_qubit
            self.lv1_circuit.add_new_core(lv1_qubit, external=True)
        _, y_max = self.circuit.coordinate_span(1)
        external_core_y_min, _ = self.external_cores.coord_span_of_core_types(
                                                            1, [CoreType.EXTCORE]) 
        if external_core_y_min is None:
            external_core_y_min = 0
        for lv1_qubit in self.external_cores.pos_2_core.values():
            if lv1_qubit.core_type != CoreType.EXTCORE:
                continue
            lv1_qubit.code_patch = CodePatch(Rotated_surface_code(self.external_cores.external_core_size))
            lv1_qubit.code_patch.code.shift(
                lv1_qubit.position[0] * (lv1_qubit.code_patch.code.x_span+1),
                (lv1_qubit.position[1]-external_core_y_min) * (lv1_qubit.code_patch.code.y_span+1) + y_max + 2
            )
            self.add_code_patch(lv1_qubit.code_patch)
            self.patch_2_lv1_qubit[lv1_qubit.code_patch.patch_id] = lv1_qubit
            self.lv1_circuit.add_new_core(lv1_qubit, external=True)

    def _add_logical_dummy_qubits(self):
        '''
        Add additional clean qubits to the stim circuit. Each clean qubit monitors both 
        logical X and Z operators of a certain code patch. 
        '''
        y_min, _ = self.circuit.coordinate_span(1)
        for logical_qubit in self.lv1_code.logical_qubits:
            logical_qubit.pos = (logical_qubit.id, y_min-2)
            pos_2_id = self.circuit.introduce_new_qubits([logical_qubit.pos])
            logical_qubit.circuit_id = pos_2_id[logical_qubit.pos]
            self.lv1_circuit.add_new_lv1_dummy_qubit(logical_qubit)
            self.dummy_qubits.add(logical_qubit.circuit_id)
        if self.external_cores is None:
            return
        _, y_max = self.circuit.coordinate_span(1)
        for index, ext_logical_qubit in enumerate(self.external_cores.external_logical_qubits):
            ext_logical_qubit.pos = (index, y_max+2)
            pos_2_id = self.circuit.introduce_new_qubits([ext_logical_qubit.pos])
            ext_logical_qubit.circuit_id = pos_2_id[ext_logical_qubit.pos]
            new_logical_id = max(self.lv1_circuit.logical_qubit_id_2_dummy_qubit.keys()) + 1
            ext_logical_qubit.id = new_logical_id
            self.lv1_circuit.add_new_lv1_dummy_qubit(ext_logical_qubit)
            self.dummy_qubits.add(ext_logical_qubit.circuit_id)

    def add_obs_from_existing_ones(self, obs_ids: list[int]) -> Lv1Obs:
        '''
        Generate a new observing region (`LV1Obs`) from existing ones.
        
        :param obs_ids: ids of existing `LV1Obs` objects
        :type obs_ids: list[int]
        '''
        new_obs = Lv1Obs(self.lv1_circuit.lv1Obs_id_new)
        self.lv1_circuit.generate_new_lv1_Obs_id()
        for obs_id in obs_ids:
            assert obs_id in self.lv1_circuit.lv1Obs_id_2_lv1Obs 
            old_obs = self.lv1_circuit.lv1Obs_id_2_lv1Obs[obs_id]
            new_obs.supp.symmetric_difference_update(old_obs.supp)
            new_obs.lv1_supp.symmetric_difference_update(old_obs.lv1_supp)
            new_obs.meas_buff_update(old_obs.meas_ids_history)
            for lv1_time, lv1_meas in old_obs.involved_lv1_meas:
                new_obs.add_lv1_meas(lv1_time, lv1_meas)
        for patch_id, _ in new_obs.lv1_supp:
            if self.lv1_circuit.patch_2_coretype[patch_id] != CoreType.OBSDUMMY:
                self.lv1_circuit.link_lv1_obs_to_patch(new_obs, patch_id)
        self.lv1_circuit.introduce_new_lv1_obs(new_obs)
        return new_obs
    
    def transversal_reset(self, basis: Pauli):
        if basis not in [Pauli.X, Pauli.Z]:
            raise NotImplementedError(f'{basis=} needs to be either X or Z')
        if basis == Pauli.X:
            lop = LogicalOp.LRX
        else:
            lop = LogicalOp.LRZ 
        for lv1_qubit in self.lv1_code.data_qubits:
            self.set_logic_op(lv1_qubit.patch_id, lop)
        self.do_next_SE()
        self.clock_move_forward

    def perfect_reset(self):
        lop = LogicalOp.LR_PERF
        for qubit in self.lv1_code.data_qubits:
            self.set_logic_op(qubit.patch_id, lop)
            self._reset_lv0_qubit_tags_for_lv1_qubit(qubit)
        self.do_next_SE()
        self._lv1_perf_reset()
        self._lv1_perf_logical_reset()
        self.clock_move_forward

    def perfect_manual_reset(self, lv1_log_qubit_cid_2_op: dict[int,Lv1LogicalOp]):
        lop = LogicalOp.LR_PERF
        for qubit in self.lv1_code.data_qubits:
            self.set_logic_op(qubit.patch_id, lop)
            self._reset_lv0_qubit_tags_for_lv1_qubit(qubit)
        self.do_next_SE()
        self._lv1_perf_reset()
        self._lv1_logical_manual_reset(lv1_log_qubit_cid_2_op)
        self.clock_move_forward
    
    def _reset_lv0_qubit_tags_for_lv1_qubit(self, lv1_qubit: Lv1Qubit):
        patch_type = 'UNDEF'
        if lv1_qubit.core_type == CoreType.XBUS:
            patch_type = 'X_ANC'
        elif lv1_qubit.core_type == CoreType.ZBUS:
            patch_type = 'Z_ANC'
        elif lv1_qubit.core_type in [CoreType.INTCORE, CoreType.EXTCORE]:
            patch_type = 'DATA'
        for qubit in lv1_qubit.code_patch.code.qubits:
            if qubit.qubit_type in [QubitType.X_ancilla, QubitType.Z_ancilla]:
                boundary, boundary_index = lv1_qubit.code_patch\
                                            .code.is_boundary_stabilizer(qubit)
                pauli = 'X' if qubit.qubit_type == QubitType.X_ancilla\
                            else 'Z'
                qubit.tag_buffer = TagBuffer(
                    lv1_qubit.patch_id,
                    lv1_qubit.distinct_position[0],
                    lv1_qubit.distinct_position[1],
                    pauli,
                    boundary,
                    boundary_index,
                    self.current_lv1_time.lv1_cycle_time,
                    patch_type = patch_type
                )
    
    def _update_lv0_qubit_tags_for_lv1_qubit(self, lv1_qubit: Lv1Qubit):
        '''
        Update the level-1 time for level-0 qubits contained in a level-1 qubit. 
        Should be called at the beginning of a level-1 round. 
        
        :param lv1_qubit: 
        :type lv1_qubit: Lv1Qubit
        '''
        for qubit in lv1_qubit.code_patch.code.qubits:
            if qubit.qubit_type in [QubitType.X_ancilla, QubitType.Z_ancilla]:
                qubit.tag_buffer.lv1_time = self.current_lv1_time.lv1_cycle_time
    
    def transversal_meas(self, basis: Pauli):
        if basis not in [Pauli.X, Pauli.Z]:
            raise NotImplementedError(f'{basis=} needs to be either X or Z')
        if basis == Pauli.X:
            lop = LogicalOp.LMX
        else:
            lop = LogicalOp.LMZ
        for lv1_qubit in self.lv1_code.data_qubits:
            self.set_logic_op(lv1_qubit.patch_id, lop)
        self.do_next_SE()
        self.clock_move_forward

    def _lv1_perf_reset(self):
        '''
        MPP all lv-1 stabilizers. 
        1. For each ancilla logical qubit, generate its corresponding MPP term
        along with a new `lv1Det`, which tracks a level-1 detecting region.
        2. Measure each MPP term and add the measurement id to its corresponding
        `lv1Det`.
        '''
        anc_id_2_mpp_supp = self._lv1_perf_anc_id_2_mpp_supp()
        sorted_anc_ids = sorted(anc_id_2_mpp_supp.keys())
        mpp_instruction = CircuitInstruction('MPP')
        anc_id_2_lv1_mpp_op: dict[int,Lv1_Op] = {}
        for anc_id in sorted_anc_ids:
            anc = self.patch_2_lv1_qubit[anc_id]
            sorted_data_ids = sorted(
                qubit.patch_id for qubit in anc.SEspacetime_2_qubit.values()
            )
            if anc.core_type == CoreType.XBUS:
                lv1_op = Lv1_Op(
                    Gate.MPP_RX,
                    targets=tuple([Lv1ID_port(id,0) for id in sorted_data_ids])
                )
            else:
                assert anc.core_type == CoreType.ZBUS
                lv1_op = Lv1_Op(
                    Gate.MPP_RZ,
                    targets=tuple([Lv1ID_port(id,0) for id in sorted_data_ids])
                )
            anc_id_2_lv1_mpp_op[anc_id] = lv1_op
            for data_id in sorted_data_ids:
                self.lv1_circuit.patch_2_timeline[data_id].add_new_event(
                    lv1_op, self.current_lv1_time
                )
            supp = anc_id_2_mpp_supp[anc_id]
            mpp_targets = []
            assert len(supp) > 0
            for target in supp:
                if target[-1] == Pauli.X:
                    mpp_targets.append(stim.target_x(target[0]))
                else:
                    assert target[-1] == Pauli.Z
                    mpp_targets.append(stim.target_z(target[0]))
                mpp_targets.append(stim.target_combiner())
            mpp_targets.pop(-1) # get rid of the last combiner
            mpp_instruction += CircuitInstruction(
                                'MPP',
                                targets=mpp_targets,
                                anc_id_2_mpp_group_index={anc_id:0}
                                )
        current_num_meas = self.circuit.num_measurements
        self.circuit.append_Instruction(mpp_instruction)
        new_meas_ids = sorted(range(current_num_meas,
                                    self.circuit.num_measurements))
        assert len(new_meas_ids) == len(sorted_anc_ids)
        for anc_id, meas_id in zip(sorted_anc_ids, new_meas_ids):
            anc = self.patch_2_lv1_qubit[anc_id]
            supp = anc_id_2_mpp_supp[anc_id]
            new_lv1_det = Lv1Det(self.lv1_circuit.lv1Det_id_new,
                                 supp=set(supp),
                                 anc_patch_id=anc.patch_id,
                                 meas_ids=set([meas_id]),
                                 pos=anc.distinct_position)
            new_lv1_det.add_lv1_meas(
                self.current_lv1_time,
                anc_id_2_lv1_mpp_op[anc_id]
            )
            self.lv1_circuit.generate_new_lv1_Det_id
            self.lv1_circuit.introduce_new_lv1_det_from_patch(anc.patch_id,
                                                                  new_lv1_det)
            for data_qubit in anc.SEspacetime_2_qubit.values():
                self.lv1_circuit.add_new_lv1_det_to_patch(new_lv1_det,
                                                              data_qubit.patch_id)
    
    def _lv1_logical_manual_reset(self, lv1_log_qubit_cid_2_op: dict[int,Lv1LogicalOp]):
        '''
        1. Commit MPP instruction.
        2. Add logical MPP event to timeline *
        3. Add lv1 obs for each logical MPP term
        '''
        mpp, qubit_id_basis_2_mpp_group_index,\
             mpp_supps = self._lv1_compile_manually_set_logical_mpp(lv1_log_qubit_cid_2_op)
        current_meas_id = self.circuit.num_measurements
        self.circuit.append_Instruction(mpp)
        new_meas_ids = sorted(range(current_meas_id,
                                    self.circuit.num_measurements))
        for logical_qubit in self.lv1_code.logical_qubits:
            assert logical_qubit.circuit_id in lv1_log_qubit_cid_2_op
            lv1_dummy_qubit = self.lv1_circuit.logical_qubit_id_2_dummy_qubit[logical_qubit.id]
            if lv1_log_qubit_cid_2_op[logical_qubit.circuit_id] == Lv1LogicalOp.RX:
                mpp_x_index = qubit_id_basis_2_mpp_group_index[(
                    logical_qubit.circuit_id, Pauli.X
                )]
                lv1_obs_x_op = Lv1_Op(Gate.MPP_RX,
                                      targets=tuple(sorted([Lv1ID_port(id,0) for id in lv1_dummy_qubit.x_supp]))
                                     )
                lv1_obs_x = Lv1Obs(
                    self.lv1_circuit.lv1Obs_id_new,
                    supp=set(mpp_supps[mpp_x_index]),
                    meas_ids_buff=set([new_meas_ids[mpp_x_index]]),
                    logical_id=logical_qubit.id,
                    logical_basis=Pauli.X,
                    involved_lv1_meas=[(self.current_lv1_time,lv1_obs_x_op)],
                    lv1_supp=set([(patch_id, Pauli.X) for patch_id in lv1_dummy_qubit.x_supp])
                )
                self.lv1_circuit.introduce_new_lv1_obs(lv1_obs_x)
                logical_qubit.basis_2_obs_id[Pauli.X] = lv1_obs_x.obs_id
                for qubit_pos in logical_qubit.x_supp:
                    qubit = self.lv1_code.distinct_pos_2_qubit[qubit_pos]
                    self.lv1_circuit.link_lv1_obs_to_patch(lv1_obs_x, qubit.patch_id)
                    self.lv1_circuit.patch_2_timeline[qubit.patch_id].add_new_event(
                                            lv1_obs_x_op, self.current_lv1_time)
                self._add_observable_include_from_lv1_obs(lv1_obs_x)
                self.lv1_circuit.generate_new_lv1_Obs_id()
            elif lv1_log_qubit_cid_2_op[logical_qubit.circuit_id] == Lv1LogicalOp.RZ:
                mpp_z_index = qubit_id_basis_2_mpp_group_index[(
                    logical_qubit.circuit_id, Pauli.Z
                )]
                lv1_obs_z_op = Lv1_Op(Gate.MPP_RZ,
                                      targets=tuple(sorted([Lv1ID_port(id,0) for id in lv1_dummy_qubit.z_supp]))
                                     )
                lv1_obs_z = Lv1Obs(
                    self.lv1_circuit.lv1Obs_id_new,
                    supp=set(mpp_supps[mpp_z_index]),
                    meas_ids_buff=set([new_meas_ids[mpp_z_index]]),
                    logical_id=logical_qubit.id,
                    logical_basis=Pauli.Z,
                    involved_lv1_meas=[(self.current_lv1_time,lv1_obs_z_op)],
                    lv1_supp=set([(patch_id, Pauli.Z) for patch_id in lv1_dummy_qubit.z_supp])
                )
                self.lv1_circuit.introduce_new_lv1_obs(lv1_obs_z)
                logical_qubit.basis_2_obs_id[Pauli.Z] = lv1_obs_z.obs_id
                for qubit_pos in logical_qubit.z_supp:
                    qubit = self.lv1_code.distinct_pos_2_qubit[qubit_pos]
                    self.lv1_circuit.link_lv1_obs_to_patch(lv1_obs_z, qubit.patch_id)
                    self.lv1_circuit.patch_2_timeline[qubit.patch_id].add_new_event(
                                            lv1_obs_z_op, self.current_lv1_time)
                self._add_observable_include_from_lv1_obs(lv1_obs_z)
                self.lv1_circuit.generate_new_lv1_Obs_id()
            else:
                assert lv1_log_qubit_cid_2_op[logical_qubit.circuit_id] == Lv1LogicalOp.BELL_RESET
                mpp_x_index = qubit_id_basis_2_mpp_group_index[(
                    logical_qubit.circuit_id, Pauli.X
                )]
                mpp_z_index = qubit_id_basis_2_mpp_group_index[(
                    logical_qubit.circuit_id, Pauli.Z
                )]
                lv1_obs_x_op = Lv1_Op(
                    Gate.MPP_RX,
                    targets=tuple(sorted([Lv1ID_port(lv1_dummy_qubit.circuit_id,0)] 
                                + [Lv1ID_port(id,0) for id in lv1_dummy_qubit.x_supp]))
                )
                lv1_obs_x = Lv1Obs(
                    self.lv1_circuit.lv1Obs_id_new,
                    supp=set(mpp_supps[mpp_x_index]),
                    meas_ids_buff=set([new_meas_ids[mpp_x_index]]),
                    logical_id=logical_qubit.id,
                    logical_basis=Pauli.X,
                    involved_lv1_meas=[(self.current_lv1_time,lv1_obs_x_op)],
                    lv1_supp=set([(patch_id, Pauli.X) for patch_id in lv1_dummy_qubit.x_supp]
                                + [(lv1_dummy_qubit.circuit_id, Pauli.X)])
                )
                self.lv1_circuit.introduce_new_lv1_obs(lv1_obs_x)
                logical_qubit.basis_2_obs_id[Pauli.X] = lv1_obs_x.obs_id
                for qubit_pos in logical_qubit.x_supp:
                    qubit = self.lv1_code.distinct_pos_2_qubit[qubit_pos]
                    self.lv1_circuit.link_lv1_obs_to_patch(lv1_obs_x, qubit.patch_id)
                self._add_observable_include_from_lv1_obs(lv1_obs_x)
                self.lv1_circuit.generate_new_lv1_Obs_id()
                self.lv1_circuit.patch_2_timeline[lv1_dummy_qubit.circuit_id].add_new_event(
                    lv1_obs_x_op, self.current_lv1_time
                )
                lv1_obs_z_op = Lv1_Op(
                    Gate.MPP_RZ,
                    targets=tuple(sorted([Lv1ID_port(lv1_dummy_qubit.circuit_id,0)] 
                                + [Lv1ID_port(id,0) for id in lv1_dummy_qubit.z_supp]))
                )
                self.lv1_circuit.patch_2_timeline[lv1_dummy_qubit.circuit_id].add_new_event(
                    lv1_obs_z_op, self.current_lv1_time
                )
                lv1_obs_z = Lv1Obs(
                    self.lv1_circuit.lv1Obs_id_new,
                    supp=set(mpp_supps[mpp_z_index]),
                    meas_ids_buff=set([new_meas_ids[mpp_z_index]]),
                    logical_id=logical_qubit.id,
                    logical_basis=Pauli.Z,
                    involved_lv1_meas=[(self.current_lv1_time,lv1_obs_z_op)],
                    lv1_supp=set([(patch_id, Pauli.Z) for patch_id in lv1_dummy_qubit.z_supp]
                                + [(lv1_dummy_qubit.circuit_id, Pauli.Z)])
                )
                self.lv1_circuit.introduce_new_lv1_obs(lv1_obs_z)
                logical_qubit.basis_2_obs_id[Pauli.Z] = lv1_obs_z.obs_id
                for qubit_pos in logical_qubit.z_supp:
                    qubit = self.lv1_code.distinct_pos_2_qubit[qubit_pos]
                    self.lv1_circuit.link_lv1_obs_to_patch(lv1_obs_z, qubit.patch_id)
                self._add_observable_include_from_lv1_obs(lv1_obs_z)
                self.lv1_circuit.generate_new_lv1_Obs_id()
        pass

    def _lv1_perf_logical_reset(self):
        '''
        1. Commit MPP instruction.
        2. Add logical MPP event to timeline *
        3. Add lv1 obs for each logical MPP term
        '''
        mpp, qubit_id_basis_2_mpp_group_index,\
             mpp_supps = self._lv1_compile_logical_mpp()
        current_meas_id = self.circuit.num_measurements
        self.circuit.append_Instruction(mpp)
        new_meas_ids = sorted(range(current_meas_id,
                                    self.circuit.num_measurements))
        for logical_qubit in self.lv1_code.logical_qubits:
            mpp_x_index = qubit_id_basis_2_mpp_group_index[(
                logical_qubit.circuit_id, Pauli.X
            )]
            mpp_z_index = qubit_id_basis_2_mpp_group_index[(
                logical_qubit.circuit_id, Pauli.Z
            )]
            lv1_dummy_qubit = self.lv1_circuit.logical_qubit_id_2_dummy_qubit[logical_qubit.id]
            lv1_obs_x_op = Lv1_Op(
                Gate.MPP_RX,
                targets=tuple(sorted([Lv1ID_port(lv1_dummy_qubit.circuit_id,0)] 
                              + [Lv1ID_port(id,0) for id in lv1_dummy_qubit.x_supp]))
            )
            lv1_obs_x = Lv1Obs(
                self.lv1_circuit.lv1Obs_id_new,
                supp=set(mpp_supps[mpp_x_index]),
                meas_ids_buff=set([new_meas_ids[mpp_x_index]]),
                logical_id=logical_qubit.id,
                logical_basis=Pauli.X,
                involved_lv1_meas=[(self.current_lv1_time,lv1_obs_x_op)],
                lv1_supp=set([(patch_id, Pauli.X) for patch_id in lv1_dummy_qubit.x_supp]
                             + [(lv1_dummy_qubit.circuit_id, Pauli.X)])
            )
            self.lv1_circuit.introduce_new_lv1_obs(lv1_obs_x)
            logical_qubit.basis_2_obs_id[Pauli.X] = lv1_obs_x.obs_id
            for qubit_pos in logical_qubit.x_supp:
                qubit = self.lv1_code.distinct_pos_2_qubit[qubit_pos]
                self.lv1_circuit.link_lv1_obs_to_patch(lv1_obs_x, qubit.patch_id)
            self._add_observable_include_from_lv1_obs(lv1_obs_x)
            self.lv1_circuit.generate_new_lv1_Obs_id()
            self.lv1_circuit.patch_2_timeline[lv1_dummy_qubit.circuit_id].add_new_event(
                lv1_obs_x_op, self.current_lv1_time
            )
            lv1_obs_z_op = Lv1_Op(
                Gate.MPP_RZ,
                targets=tuple(sorted([Lv1ID_port(lv1_dummy_qubit.circuit_id,0)] 
                              + [Lv1ID_port(id,0) for id in lv1_dummy_qubit.z_supp]))
            )
            self.lv1_circuit.patch_2_timeline[lv1_dummy_qubit.circuit_id].add_new_event(
                lv1_obs_z_op, self.current_lv1_time
            )
            lv1_obs_z = Lv1Obs(
                self.lv1_circuit.lv1Obs_id_new,
                supp=set(mpp_supps[mpp_z_index]),
                meas_ids_buff=set([new_meas_ids[mpp_z_index]]),
                logical_id=logical_qubit.id,
                logical_basis=Pauli.Z,
                involved_lv1_meas=[(self.current_lv1_time,lv1_obs_z_op)],
                lv1_supp=set([(patch_id, Pauli.Z) for patch_id in lv1_dummy_qubit.z_supp]
                             + [(lv1_dummy_qubit.circuit_id, Pauli.Z)])
            )
            self.lv1_circuit.introduce_new_lv1_obs(lv1_obs_z)
            logical_qubit.basis_2_obs_id[Pauli.Z] = lv1_obs_z.obs_id
            for qubit_pos in logical_qubit.z_supp:
                qubit = self.lv1_code.distinct_pos_2_qubit[qubit_pos]
                self.lv1_circuit.link_lv1_obs_to_patch(lv1_obs_z, qubit.patch_id)
            self._add_observable_include_from_lv1_obs(lv1_obs_z)
            self.lv1_circuit.generate_new_lv1_Obs_id()
        pass
    
    def _lv1_perf_meas(self):
        anc_id_2_mpp_supp = self._lv1_perf_anc_id_2_mpp_supp()
        sorted_anc_ids = sorted(anc_id_2_mpp_supp.keys())
        mpp_instruction = CircuitInstruction('MPP')
        anc_id_2_lv1_mpp_op: dict[int,Lv1_Op] = {}
        for anc_id in sorted_anc_ids:
            anc = self.patch_2_lv1_qubit[anc_id]
            sorted_data_ids = sorted(
                qubit.patch_id for qubit in anc.SEspacetime_2_qubit.values()
            )
            if anc.core_type == CoreType.XBUS:
                lv1_op = Lv1_Op(
                    Gate.MPP_MX,
                    targets=tuple([Lv1ID_port(id,0) for id in sorted_data_ids])
                )
            else:
                assert anc.core_type == CoreType.ZBUS
                lv1_op = Lv1_Op(
                    Gate.MPP_MZ,
                    targets=tuple([Lv1ID_port(id,0) for id in sorted_data_ids])
                )
            anc_id_2_lv1_mpp_op[anc_id] = lv1_op
            for data_id in sorted_data_ids:
                self.lv1_circuit.patch_2_timeline[data_id].add_new_event(
                    lv1_op, self.current_lv1_time
                )
            supp = anc_id_2_mpp_supp[anc_id]
            mpp_targets = []
            assert len(supp) > 0
            for target in supp:
                if target[-1] == Pauli.X:
                    mpp_targets.append(stim.target_x(target[0]))
                else:
                    assert target[-1] == Pauli.Z
                    mpp_targets.append(stim.target_z(target[0]))
                mpp_targets.append(stim.target_combiner())
            mpp_targets.pop(-1) # get rid of the last combiner
            mpp_instruction += CircuitInstruction(
                                'MPP',
                                targets=mpp_targets,
                                anc_id_2_mpp_group_index={anc_id:0}
                                )
        current_num_meas = self.circuit.num_measurements
        self.circuit.append_Instruction(mpp_instruction)
        new_meas_ids = sorted(range(current_num_meas,
                                    self.circuit.num_measurements))
        assert len(new_meas_ids) == len(sorted_anc_ids)
        # verify each anc is only associated with a single active lv-1 det
        for anc_id, meas_id in zip(sorted_anc_ids, new_meas_ids):
            if len(self.lv1_circuit
                   .patch_2_generated_active_lv1Det_ids[anc_id]) != 1:
                raise RuntimeError(f'{anc_id=} has more than 1 '
                                   f'active lv-1 dets left')
            lv1_det_id = list(self.lv1_circuit
                              .patch_2_generated_active_lv1Det_ids[anc_id])[0]
            lv1_det = self.lv1_circuit.lv1Det_id_2_lv1Det[lv1_det_id]
            if lv1_det.supp != set(anc_id_2_mpp_supp[anc_id]):
                raise RuntimeError(f'{lv1_det.supp=} cannot be perfectly measured '
                            f'by the following mpp {anc_id_2_mpp_supp[anc_id]}')
            lv1_det.meas_update([meas_id])
            lv1_det.supp_update(lv1_det.supp)
            lv1_det.add_lv1_meas(
                self.current_lv1_time,
                anc_id_2_lv1_mpp_op[anc_id]
            )
            self.lv1_circuit.deactivate_lv1_det(lv1_det.id)
            self.circuit.detector(lv1_det.meas_ids,
                            (999,) + lv1_det.pos
                            + (self.current_lv1_time.lv1_cycle_time,))
        if len(self.lv1_circuit.active_lv1_dets) != 0:
            for lv1_det in self.lv1_circuit.active_lv1_dets:
                if lv1_det.unpaired is not True:
                    raise RuntimeError(f'There are still active lv-1 dets '
                                    f'{self.lv1_circuit.active_lv1_dets} '
                                    f'after perfectly measuring everthing.')
    
    def _extcore_bell_mpp_reset(self, patch_id: int):
        lv1_qubit = self.patch_2_lv1_qubit[patch_id]
        assert lv1_qubit.core_type == CoreType.EXTCORE
        obs_qubit = self.external_cores.external_logical_qubits[
                            self.external_cores.pos_2_external_logical_qubit_index[
                                lv1_qubit.position
                            ]]
        mppx_supps = [(obs_qubit.circuit_id, Pauli.X)]
        mppx_targets = [stim.target_x(obs_qubit.circuit_id), stim.target_combiner()]
        for qubit in lv1_qubit.code_patch.code.logical_X_operator:
            mppx_supps.append((qubit.circuit_id, Pauli.X)) 
            mppx_targets.append(stim.target_x(qubit.circuit_id))
            mppx_targets.append(stim.target_combiner())
        mppx_targets.pop()
        current_meas_id = self.circuit.num_measurements
        self.circuit.append_Instruction(CircuitInstruction('MPP', mppx_targets))
        obs_qubit_lv1_circ = self.lv1_circuit.obs_id_2_obs_qubit[obs_qubit.id]
        lv1_mppx_op = Lv1_Op(
                Gate.MPP_RX,
                targets=tuple(sorted([Lv1ID_port(obs_qubit_lv1_circ.circuit_id,0),
                               Lv1ID_port(lv1_qubit.patch_id, 0)]))
            )
        lv1_obs_x = Lv1Obs(
            self.lv1_circuit.lv1Obs_id_new,
            supp=set(mppx_supps),
            meas_ids_buff=set([current_meas_id]),
            logical_id=obs_qubit.id,
            logical_basis=Pauli.X,
            involved_lv1_meas=[(self.current_lv1_time,lv1_mppx_op)],
            lv1_supp=set([(obs_qubit_lv1_circ.circuit_id, Pauli.X),
                          (lv1_qubit.patch_id, Pauli.X)])
        )
        self.lv1_circuit.introduce_new_lv1_obs(lv1_obs_x)
        obs_qubit.basis_2_obs_id[Pauli.X] = lv1_obs_x.obs_id
        self.lv1_circuit.link_lv1_obs_to_patch(lv1_obs_x, lv1_qubit.patch_id)
        self._add_observable_include_from_lv1_obs(lv1_obs_x)
        self.lv1_circuit.generate_new_lv1_Obs_id()
        self.lv1_circuit.patch_2_timeline[obs_qubit_lv1_circ.circuit_id].add_new_event(
            lv1_mppx_op, self.current_lv1_time
        )
        self.lv1_circuit.patch_2_timeline[patch_id].add_new_event(
            lv1_mppx_op, self.current_lv1_time
        )
        mppz_supps = [(obs_qubit.circuit_id, Pauli.Z)]
        mppz_targets = [stim.target_z(obs_qubit.circuit_id), stim.target_combiner()]
        for qubit in lv1_qubit.code_patch.code.logical_Z_operator:
            mppz_supps.append((qubit.circuit_id, Pauli.Z)) 
            mppz_targets.append(stim.target_z(qubit.circuit_id))
            mppz_targets.append(stim.target_combiner())
        mppz_targets.pop()
        current_meas_id = self.circuit.num_measurements
        self.circuit.append_Instruction(CircuitInstruction('MPP', mppz_targets))
        lv1_mppz_op = Lv1_Op(
                Gate.MPP_RZ,
                targets=tuple(sorted([Lv1ID_port(obs_qubit_lv1_circ.circuit_id,0),
                               Lv1ID_port(lv1_qubit.patch_id, 0)]))
            )
        lv1_obs_z = Lv1Obs(
            self.lv1_circuit.lv1Obs_id_new,
            supp=set(mppz_supps),
            meas_ids_buff=set([current_meas_id]),
            logical_id=obs_qubit.id,
            logical_basis=Pauli.Z,
            involved_lv1_meas=[(self.current_lv1_time,lv1_mppx_op)],
            lv1_supp=set([(obs_qubit_lv1_circ.circuit_id, Pauli.Z),
                          (lv1_qubit.patch_id, Pauli.Z)])
        )
        self.lv1_circuit.introduce_new_lv1_obs(lv1_obs_z)
        obs_qubit.basis_2_obs_id[Pauli.Z] = lv1_obs_z.obs_id
        self.lv1_circuit.link_lv1_obs_to_patch(lv1_obs_z, lv1_qubit.patch_id)
        self._add_observable_include_from_lv1_obs(lv1_obs_z)
        self.lv1_circuit.generate_new_lv1_Obs_id()
        self.lv1_circuit.patch_2_timeline[obs_qubit_lv1_circ.circuit_id].add_new_event(
            lv1_mppz_op, self.current_lv1_time
        )
        self.lv1_circuit.patch_2_timeline[patch_id].add_new_event(
            lv1_mppz_op, self.current_lv1_time
        )
        pass

    def _register_extcore_bell_reset(self, patch_id: int):
        pass

    def _lv1_compile_manually_set_logical_mpp(self, 
                                              lv1_log_qubit_cid_2_op: dict[int, Lv1LogicalOp]
                                              ) -> tuple[CircuitInstruction,
                                                dict[tuple[int,Pauli],int],
                                                list[list[tuple[int,Pauli]]]]:
        '''
        Compile the MPP instruction for measuring each level-1 logical operator. 
        For each logical qubit, its logical X operator is measured first and 
        its logical Z opeartor is measured next. 
        
        :param self: Description
        :param lv1_log_qubit_cid_2_log_Op: Description
        :type lv1_log_qubit_cid_2_log_Op: dict[int, Lv1LogicalOp]
        :return: Description
        :rtype: tuple[CircuitInstruction, dict[tuple[int, Pauli], int], list[list[tuple[int, Pauli]]]]
        '''
        mpp = CircuitInstruction('MPP')
        qubit_id_basis_2_mpp_group_index: dict[tuple[int,Pauli],int] = {}
        mpp_group_index = 0
        mpp_supps: list[list[tuple[int,Pauli]]] = []
        for lv1_logical_qubit in self.lv1_code.logical_qubits:
            assert lv1_logical_qubit.circuit_id in lv1_log_qubit_cid_2_op
            x_targets: list[stim.GateTarget] = []
            x_supp: list[tuple[int,Pauli]] = []
            z_targets: list[stim.GateTarget] = []
            z_supp: list[tuple[int,Pauli]] = []
            # x_targets = [stim.target_x(lv1_logical_qubit.circuit_id),
            #              stim.target_combiner()]
            # x_supp: list[tuple[int,Pauli]] = [(lv1_logical_qubit.circuit_id, Pauli.X)]
            # z_targets = [stim.target_z(lv1_logical_qubit.circuit_id),
            #               stim.target_combiner()]
            # z_supp: list[tuple[int,Pauli]] = [(lv1_logical_qubit.circuit_id, Pauli.Z)]
            for data_pos in lv1_logical_qubit.x_supp:
                lv1_data = self.lv1_code.distinct_pos_2_qubit[data_pos]
                for qubit in lv1_data.code_patch.code.logical_X_operator:
                    x_supp.append((qubit.circuit_id,Pauli.X))
                    x_targets.append(stim.target_x(qubit.circuit_id))
                    x_targets.append(stim.target_combiner())
            for data_pos in lv1_logical_qubit.z_supp:
                lv1_data = self.lv1_code.distinct_pos_2_qubit[data_pos]
                for qubit in lv1_data.code_patch.code.logical_Z_operator:
                    z_supp.append((qubit.circuit_id,Pauli.Z))
                    z_targets.append(stim.target_z(qubit.circuit_id))
                    z_targets.append(stim.target_combiner())
            x_targets.pop()
            z_targets.pop()
            if lv1_log_qubit_cid_2_op[lv1_logical_qubit.circuit_id] == Lv1LogicalOp.RX:
                mpp += CircuitInstruction('MPP',x_targets)
                mpp_supps.append(x_supp)
                qubit_id_basis_2_mpp_group_index[(lv1_logical_qubit.circuit_id,Pauli.X)] = mpp_group_index
                mpp_group_index += 1
            elif lv1_log_qubit_cid_2_op[lv1_logical_qubit.circuit_id] == Lv1LogicalOp.RZ:
                mpp += CircuitInstruction('MPP',z_targets)
                mpp_supps.append(z_supp)
                qubit_id_basis_2_mpp_group_index[(lv1_logical_qubit.circuit_id,Pauli.Z)] = mpp_group_index
                mpp_group_index += 1
            else:
                assert lv1_log_qubit_cid_2_op[lv1_logical_qubit.circuit_id] == Lv1LogicalOp.BELL_RESET
                x_targets.append(stim.target_combiner())
                x_targets.append(stim.target_x(lv1_logical_qubit.circuit_id))
                x_supp.append((lv1_logical_qubit.circuit_id,Pauli.X))
                mpp += CircuitInstruction('MPP',x_targets)
                mpp_supps.append(x_supp)
                qubit_id_basis_2_mpp_group_index[
                    (lv1_logical_qubit.circuit_id,Pauli.X)] = mpp_group_index
                mpp_group_index += 1
                z_targets.append(stim.target_combiner())
                z_targets.append(stim.target_z(lv1_logical_qubit.circuit_id))
                z_supp.append((lv1_logical_qubit.circuit_id,Pauli.Z))
                mpp += CircuitInstruction('MPP',z_targets)
                mpp_supps.append(z_supp)
                qubit_id_basis_2_mpp_group_index[
                    (lv1_logical_qubit.circuit_id,Pauli.Z)] = mpp_group_index
                mpp_group_index += 1
        return (mpp, qubit_id_basis_2_mpp_group_index, mpp_supps)

    def _lv1_compile_logical_mpp(self) -> tuple[CircuitInstruction,
                                                dict[tuple[int,Pauli],int],
                                                list[list[tuple[int,Pauli]]]]:
        '''
        Compile the MPP instruction for measuring each level-1 logical operator. 
        For each logical qubit, its logical X operator is measured first and 
        its logical Z opeartor is measured next. 
        
        :Return: 
        * CircuitInstruction 
        * dict {(id,basis): mpp_group_index}
        * MPP supps
        '''
        mpp = CircuitInstruction('MPP')
        qubit_id_basis_2_mpp_group_index: dict[tuple[int,Pauli],int] = {}
        mpp_group_index = 0
        mpp_supps: list[list[tuple[int,Pauli]]] = []
        for lv1_logical_qubit in self.lv1_code.logical_qubits:
            x_targets = [stim.target_x(lv1_logical_qubit.circuit_id),
                         stim.target_combiner()]
            x_supp: list[tuple[int,Pauli]] = [(lv1_logical_qubit.circuit_id, Pauli.X)]
            z_targets = [stim.target_z(lv1_logical_qubit.circuit_id),
                          stim.target_combiner()]
            z_supp: list[tuple[int,Pauli]] = [(lv1_logical_qubit.circuit_id, Pauli.Z)]
            for data_pos in lv1_logical_qubit.x_supp:
                lv1_data = self.lv1_code.distinct_pos_2_qubit[data_pos]
                for qubit in lv1_data.code_patch.code.logical_X_operator:
                    x_supp.append((qubit.circuit_id,Pauli.X))
                    x_targets.append(stim.target_x(qubit.circuit_id))
                    x_targets.append(stim.target_combiner())
            for data_pos in lv1_logical_qubit.z_supp:
                lv1_data = self.lv1_code.distinct_pos_2_qubit[data_pos]
                for qubit in lv1_data.code_patch.code.logical_Z_operator:
                    z_supp.append((qubit.circuit_id,Pauli.Z))
                    z_targets.append(stim.target_z(qubit.circuit_id))
                    z_targets.append(stim.target_combiner())
            x_targets.pop()
            z_targets.pop()
            mpp += CircuitInstruction(
                'MPP',
                x_targets,
            )
            mpp_supps.append(x_supp)
            qubit_id_basis_2_mpp_group_index[
                (lv1_logical_qubit.circuit_id,Pauli.X)] = mpp_group_index
            mpp_group_index += 1
            mpp += CircuitInstruction(
                'MPP',
                z_targets,
            )
            mpp_supps.append(z_supp)
            qubit_id_basis_2_mpp_group_index[
                (lv1_logical_qubit.circuit_id,Pauli.Z)] = mpp_group_index
            mpp_group_index += 1
        return (mpp, qubit_id_basis_2_mpp_group_index, mpp_supps)
    
    def _lv1_perf_logical_auto_meas(self):
        for lv1_obs in self.lv1_circuit.lv1Obs_id_2_lv1Obs.values():
            if not lv1_obs.is_active:
                continue
            mpp_targets: list[stim.GateTarget] = []
            op_basis = Pauli.X
            for cid, basis in sorted(lv1_obs.supp):
                if basis == Pauli.X:
                    mpp_targets.append(stim.target_x(cid))
                else:
                    assert basis == Pauli.Z
                    op_basis = Pauli.Z
                    mpp_targets.append(stim.target_z(cid))
                mpp_targets.append(stim.target_combiner())
            mpp_targets.pop()
            mpp = stim.CircuitInstruction('MPP', mpp_targets)
            current_meas_id = self.circuit.num_measurements
            self.circuit.append_Instruction(mpp)
            assert len(lv1_obs.meas_ids_buff) == 0
            lv1_obs.meas_buff_update([current_meas_id])
            self._add_observable_include_from_lv1_obs(lv1_obs)
            lv1_op = Lv1_Op(
                Gate.MPP_MX if op_basis == Pauli.X else Gate.MPP_MZ,
                targets=tuple(sorted([Lv1ID_port(id,0) for id, _ in lv1_obs.lv1_supp]))
            )
            for lv1_cid, basis in lv1_obs.lv1_supp:
                assert basis == op_basis 
                self.lv1_circuit.patch_2_timeline[lv1_cid].add_new_event(lv1_op, self.current_lv1_time)
            lv1_obs.add_lv1_meas(self.current_lv1_time, lv1_op)

    def _lv1_perf_logical_meas(self):
        mpp, qubit_id_basis_2_mpp_group_index,\
             mpp_supps = self._lv1_compile_logical_mpp()
        current_meas_id = self.circuit.num_measurements
        self.circuit.append_Instruction(mpp)
        new_meas_ids = sorted(range(current_meas_id,
                                    self.circuit.num_measurements))
        for lv1_obs in self.lv1_circuit.lv1Obs_id_2_lv1Obs.values():
            assert lv1_obs.logical_id is not None
            if lv1_obs.logical_id >= len(self.lv1_code.logical_qubits):
                continue
            lv1_logical_qubit = self.lv1_code.logical_qubits[lv1_obs.logical_id]
            assert len(lv1_obs.meas_ids_buff) == 0
            mpp_group_index = qubit_id_basis_2_mpp_group_index[(
                    lv1_logical_qubit.circuit_id, lv1_obs.logical_basis
                )]
            lv1_obs.meas_buff_update([
                new_meas_ids[mpp_group_index]
            ])
            self._add_observable_include_from_lv1_obs(lv1_obs)
            
            lv1_dummy_qubit = self.lv1_circuit.logical_qubit_id_2_dummy_qubit[lv1_obs.logical_id]
            lv1_obs_x_op = Lv1_Op(
                Gate.MPP_MX,
                targets=tuple(sorted([Lv1ID_port(lv1_dummy_qubit.circuit_id,0)] 
                              + [Lv1ID_port(id,0) for id in lv1_dummy_qubit.x_supp]))
            )
            lv1_obs_z_op = Lv1_Op(
                Gate.MPP_MZ,
                targets=tuple(sorted([Lv1ID_port(lv1_dummy_qubit.circuit_id,0)] 
                              + [Lv1ID_port(id,0) for id in lv1_dummy_qubit.z_supp]))
            )
            if lv1_obs.logical_basis == Pauli.X:
                self.lv1_circuit.patch_2_timeline[lv1_dummy_qubit.circuit_id].add_new_event(
                    lv1_obs_x_op, self.current_lv1_time
                )
                lv1_obs.add_lv1_meas(self.current_lv1_time, lv1_obs_x_op)
            else:
                assert lv1_obs.logical_basis == Pauli.Z
                self.lv1_circuit.patch_2_timeline[lv1_dummy_qubit.circuit_id].add_new_event(
                    lv1_obs_z_op, self.current_lv1_time
                )
                lv1_obs.add_lv1_meas(self.current_lv1_time, lv1_obs_z_op)
                # print(self.lv1_circuit.patch_2_timeline[lv1_obs_qubit.circuit_id].time_2_ops)
        pass

    def _lv1_perf_anc_id_2_mpp_supp(self) -> dict[int,list[tuple[int,Pauli]]]: 
        '''
        Generate the mapping from a level-1 ancilla to the lv-0 logical operator 
        (on data patches) that it measures. 
        '''
        se_group_indices = sorted(self.lv1_code
                                  .SE_group_index_2_anc_qubits.keys())
        anc_id_2_mpp_supp: dict[int,list[tuple[int,Pauli]]] = {}
        for se_group_index in se_group_indices:
            for anc in self.lv1_code.SE_group_index_2_anc_qubits[se_group_index]:
                logical_basis = Pauli.X if anc.core_type == CoreType.XBUS\
                                        else Pauli.Z
                anc_logical_operator = \
                            anc.code_patch.code.logical_X_operator\
                            if anc.core_type == CoreType.XBUS\
                            else anc.code_patch.code.logical_Z_operator
                anc_logical_supp_pos = [(qubit.position,logical_basis) 
                                        for qubit in anc_logical_operator]
                mpp_supp: set[tuple[int,Pauli]] = set()
                for se_spacetime in anc.SEspacetime_2_qubit:
                    port = se_spacetime.port
                    data_qubit = anc.SEspacetime_2_qubit[se_spacetime]
                    assert isinstance(anc.code_patch.code,ShuttleBus_code)
                    anc_pos_2_data_pos = anc.code_patch.code.dock(
                        data_qubit.code_patch.code, port
                        )
                    data_supp_pos = [(anc_pos_2_data_pos[target[0]],logical_basis) 
                                     for target in anc_logical_supp_pos 
                                     if target[0] in anc_pos_2_data_pos]
                    data_supp = [
                        (data_qubit.code_patch.code.
                         position_2_qubit[target[0]].circuit_id, logical_basis)
                        for target in data_supp_pos
                        ]
                    mpp_supp.symmetric_difference_update(data_supp)
                anc_id_2_mpp_supp[anc.patch_id] = sorted(
                    mpp_supp,
                    key=lambda target: target[0]
                )   # supp sorted by circuit id
        return anc_id_2_mpp_supp
    
    def perfect_meas(self):
        lop = LogicalOp.LM_PERF
        for qubit in self.lv1_code.data_qubits:
            self.set_logic_op(qubit.patch_id, lop)
        self.do_next_SE()
        self._lv1_perf_meas()
        # self._lv1_perf_logical_meas()
        self._lv1_perf_logical_auto_meas()
        self.clock_move_forward

    def logical_SE_round(self):
        SE_group_indices = sorted(self.lv1_code
                                  .SE_group_index_2_anc_qubits.keys())
        inter_se_rounds = ceil(self.lv1_code.time_unit*self.inner_distance)
        for SE_group_index in SE_group_indices:
            # anc reset
            for anc in self.lv1_code.SE_group_index_2_anc_qubits[SE_group_index]:
                if anc.core_type == CoreType.XBUS:
                    lop = LogicalOp.LRX
                    lv1_gate = Gate.RX
                else:
                    assert anc.core_type == CoreType.ZBUS
                    lop = LogicalOp.LRZ
                    lv1_gate = Gate.RZ
                self._reset_lv0_qubit_tags_for_lv1_qubit(anc)
                self.set_logic_op(anc.patch_id, lop)
                self._register_anc_patch_reset(anc.patch_id, lv1_gate)
            for data in self.lv1_code.data_qubits:
                self._update_lv0_qubit_tags_for_lv1_qubit(data)
                self.set_logic_op(data.patch_id, LogicalOp.LSE)
            se_group_start_time = CoreInstTime(self.lv1_circuit.lv1_cycle_time,
                                               SE_group_index, 0)
            self.core_time_2_tick[se_group_start_time] = self.current_tick
            if self.external_cores is not None:
                if se_group_start_time in self.external_cores.time_2_instructions:
                    instructions = self.external_cores.time_2_instructions[se_group_start_time]
                    for instruction in instructions:
                        target_qubit = self.external_cores.pos_2_core[instruction.target_pos]
                        if instruction.core_operation == CoreOp.MPP_BELL_RESET:
                            assert instruction.target_type == CoreType.EXTCORE
                            self.set_logic_op(target_qubit.patch_id, LogicalOp.LR_PERF) 
                            self._reset_lv0_qubit_tags_for_lv1_qubit(target_qubit)    
                            # self._register_patch_reset(target_qubit.patch_id, Gate.MPP_RESET)
                        elif instruction.core_operation == CoreOp.MPP_BELL_MEAS:
                            assert instruction.target_type == CoreType.EXTCORE
                            self.set_logic_op(target_qubit.patch_id, LogicalOp.LM_PERF)    
                            self._update_lv0_qubit_tags_for_lv1_qubit(target_qubit) 
                        elif instruction.core_operation in [CoreOp.RZ, CoreOp.RX]:
                            lop = LogicalOp.LRZ if instruction.core_operation == CoreOp.RZ \
                                                else LogicalOp.LRX
                            self.set_logic_op(target_qubit.patch_id, lop)
                            self._reset_lv0_qubit_tags_for_lv1_qubit(target_qubit)
                            reset_gate = Gate.RZ if instruction.core_operation == CoreOp.RZ \
                                                 else Gate.RX
                            if target_qubit.core_type in [CoreType.XBUS, CoreType.ZBUS]:
                                self._register_anc_patch_reset(target_qubit.patch_id, reset_gate, unpaired=True)
                            else:
                                self._register_patch_reset(target_qubit.patch_id, reset_gate)
                        elif instruction.core_operation in [CoreOp.MZ, CoreOp.MX]:
                            lop = LogicalOp.LMZ if instruction.core_operation == CoreOp.MZ \
                                                else LogicalOp.LMX
                            meas_gate = Gate.MX if instruction.core_operation == CoreOp.MX \
                                                else Gate.MZ
                            self.set_logic_op(target_qubit.patch_id, lop)
                            self._update_lv0_qubit_tags_for_lv1_qubit(target_qubit)
                            self._register_patch_meas(target_qubit.patch_id, meas_gate)
                for qubit in self.external_cores.external_cores:
                    if qubit.code_patch.active == True:
                        self.set_logic_op_no_overwrite(qubit.patch_id, LogicalOp.LSE)
                
            self.do_next_SE()
            # handle external core Bell resets
            if self.external_cores is not None:
                if se_group_start_time in self.external_cores.time_2_instructions:
                    for instruction in self.external_cores.time_2_instructions[se_group_start_time]:
                        if instruction.core_operation == CoreOp.MPP_BELL_RESET:
                            target_qubit = self.external_cores.pos_2_core[instruction.target_pos]
                            self._extcore_bell_mpp_reset(target_qubit.patch_id)
                            pass
                if se_group_start_time in self.external_cores.time_2_obs_instructions:
                    for obs_inst in self.external_cores.time_2_obs_instructions[se_group_start_time]:
                        if obs_inst.obs_operation == Lv1ObsOp.ADD:
                            target_qubit = self.external_cores.pos_2_core[obs_inst.target_pos]
                            internal_logical_qubits = [self.lv1_code.logical_qubits[index] 
                                                       for index in obs_inst.internal_logical_qubit_ids]
                            obs_ids = [lq.basis_2_obs_id[obs_inst.basis] for lq in internal_logical_qubits]
                            if self.external_cores is not None:
                                external_logical_qubits = [self.external_cores.external_logical_qubits[index]
                                                            for index in obs_inst.external_logical_qubit_ids]
                                for lq in external_logical_qubits:
                                    obs_ids.append(lq.basis_2_obs_id[obs_inst.basis])
                            new_obs = self.add_obs_from_existing_ones(obs_ids)
                            if target_qubit.core_type in [CoreType.XBUS, CoreType.ZBUS]:
                                self.lv1_circuit.external_anc_add_meas_obs_id(target_qubit.patch_id, new_obs.obs_id)
                            self.lv1_circuit.link_lv1_obs_to_patch(new_obs, target_qubit.patch_id)
                            if obs_inst.basis == Pauli.X:
                                logical_x_supp = [(qubit.circuit_id,Pauli.X) for qubit in 
                                                  target_qubit.code_patch.code.logical_X_operator]
                                new_obs.supp_update(logical_x_supp)
                                new_obs.lv1_supp_update([(target_qubit.patch_id,Pauli.X)])
                            else:
                                assert obs_inst.basis == Pauli.Z
                                logical_z_supp = [(qubit.circuit_id,Pauli.Z) for qubit in 
                                                  target_qubit.code_patch.code.logical_Z_operator]
                                new_obs.supp_update(logical_z_supp)
                                new_obs.lv1_supp_update([(target_qubit.patch_id,Pauli.Z)])
                            self._add_observable_include_from_lv1_obs(new_obs)
                        else:
                            raise NotImplementedError(f'{obs_inst.obs_operation=} not implemented')
            self.clock_move_forward
            # inblock indices
            inblock_indices = self.lv1_code\
                              .SE_group_index_2_inblock_indices[SE_group_index]
            for i in range(max(inblock_indices)+1):
                if i in inblock_indices:
                    for data in self.lv1_code.data_qubits:
                        se_spacetime = data.check_SE_at_time(SE_group_index,i)
                        if se_spacetime is None:
                            continue
                        anc_patch = data.SEspacetime_2_qubit[se_spacetime]
                        if anc_patch.core_type == CoreType.XBUS:
                            ct = 'CT'
                        else:
                            assert anc_patch.core_type == CoreType.ZBUS
                            ct = 'TC'
                        self.set_inter_cnot(anc_patch.patch_id, data.patch_id,
                                            ctrl_target=ct,
                                            port=se_spacetime.port)
                        self._register_inter_cnots(anc_patch.patch_id, data.patch_id,
                                                   ct, se_spacetime.port)
                instruction_time = CoreInstTime(self.lv1_circuit.lv1_cycle_time,
                                                SE_group_index, i)
                if instruction_time not in self.core_time_2_tick:
                    self.core_time_2_tick[instruction_time] = self.current_tick
                if self.external_cores is not None and instruction_time != se_group_start_time:
                    if instruction_time in self.external_cores.time_2_instructions:
                        instructions = self.external_cores.time_2_instructions[instruction_time]
                        for instruction in instructions:
                            if instruction.target_type != CoreType.INTCORE:
                                target_qubit = self.external_cores.pos_2_core[instruction.target_pos]
                            else:
                                target_qubit = self.lv1_code.distinct_pos_2_qubit[instruction.target_pos]
                            if instruction.core_operation == CoreOp.CX_RD:
                                ctrl_qubit = self.external_cores.pos_2_core[instruction.ctrl_pos]
                                if ctrl_qubit.core_type == CoreType.EXTCORE:
                                    assert target_qubit.core_type == CoreType.ZBUS
                                    self.set_inter_reversed_cnot(target_qubit.patch_id,
                                                                 ctrl_qubit.patch_id, 'TC')
                                    self._register_inter_reversed_cnots(target_qubit.patch_id,
                                                                        ctrl_qubit.patch_id, 'TC')
                                else:
                                    assert ctrl_qubit.core_type == CoreType.XBUS
                                    assert target_qubit.core_type == CoreType.EXTCORE
                                    self.set_inter_reversed_cnot(ctrl_qubit.patch_id, 
                                                                 target_qubit.patch_id, 'CT')
                                    self._register_inter_reversed_cnots(ctrl_qubit.patch_id, 
                                                                        target_qubit.patch_id, 'CT')
                            elif instruction.core_operation == CoreOp.CX:
                                if instruction.ctrl_type != CoreType.INTCORE:
                                    ctrl_qubit = self.external_cores.pos_2_core[instruction.ctrl_pos]
                                else:
                                    ctrl_qubit = self.lv1_code.distinct_pos_2_qubit[instruction.ctrl_pos]
                                if ctrl_qubit.core_type == CoreType.INTCORE:
                                    assert target_qubit.core_type == CoreType.ZBUS
                                    self.set_inter_cnot(target_qubit.patch_id, ctrl_qubit.patch_id, 'TC', port=instruction.target_port)
                                    self._register_inter_cnots(target_qubit.patch_id, ctrl_qubit.patch_id, 'TC', port=instruction.target_port)
                                else:
                                    assert ctrl_qubit.core_type == CoreType.XBUS
                                    assert target_qubit.core_type == CoreType.INTCORE
                                    self.set_inter_cnot(ctrl_qubit.patch_id, target_qubit.patch_id, 'CT', port=instruction.ctrl_port)
                                    self._register_inter_cnots(ctrl_qubit.patch_id, target_qubit.patch_id, 'CT', port=instruction.ctrl_port)
                            pass
                self.commit_inter_cnots()
                for clock_offset in range(inter_se_rounds):
                    instruction_time = CoreInstTime(self.lv1_circuit.lv1_cycle_time,
                                                     SE_group_index, i, clock_offset)
                    if clock_offset == 0:
                        self.LSE_rounds_for_all_active_qubits(1)
                        continue
                    if self.external_cores is None:
                        self.LSE_rounds_for_all_active_qubits(1)
                        continue
                    if instruction_time not in self.external_cores.time_2_instructions:
                        self.LSE_rounds_for_all_active_qubits(1)
                        continue
                    instructions = self.external_cores.time_2_instructions[instruction_time]
                    self.handle_external_CNOT_instructions(instructions)
                    self.LSE_rounds_for_all_active_qubits(1)
            # anc meas
            for anc in self.lv1_code.SE_group_index_2_anc_qubits[SE_group_index]:
                if anc.core_type == CoreType.XBUS:
                    lop = LogicalOp.LMX
                else:
                    assert anc.core_type == CoreType.ZBUS
                    lop = LogicalOp.LMZ
                self.set_logic_op(anc.patch_id, lop)
            for data in self.lv1_code.data_qubits:
                self.set_logic_op(data.patch_id, LogicalOp.LSE)
            ext_anc_id_2_meas_gate: dict[int,Gate] = {}
            if self.external_cores is not None:
                se_group_end_time = CoreInstTime(self.lv1_circuit.lv1_cycle_time,
                                                 SE_group_index, -1)
                if se_group_end_time in self.external_cores.time_2_instructions:
                    instructions = self.external_cores.time_2_instructions[se_group_end_time]
                    for instruction in instructions:
                        assert instruction.core_operation in [CoreOp.MX, CoreOp.MZ]
                        target_qubit = self.external_cores.pos_2_core[instruction.target_pos]
                        lop = LogicalOp.LMZ if instruction.core_operation == CoreOp.MZ \
                                            else LogicalOp.LMX
                        meas_gate = Gate.MX if instruction.core_operation == CoreOp.MX \
                                            else Gate.MZ
                        self.set_logic_op(target_qubit.patch_id, lop)
                        assert instruction.target_type in [CoreType.XBUS, CoreType.ZBUS]
                        ext_anc_id_2_meas_gate[target_qubit.patch_id] = meas_gate
                for qubit in self.external_cores.external_cores:
                    if qubit.code_patch.active == True:
                        self.set_logic_op_no_overwrite(qubit.patch_id, LogicalOp.LSE)
            self.do_next_SE()
            for anc in self.lv1_code.SE_group_index_2_anc_qubits[SE_group_index]:
                meas_gate = Gate.MX if anc.core_type == CoreType.XBUS \
                                    else Gate.MZ
                self._register_anc_meas(anc.patch_id, meas_gate)
            for anc_id in ext_anc_id_2_meas_gate:
                meas_gate = ext_anc_id_2_meas_gate[anc_id]
                self._register_anc_meas(anc_id, meas_gate)
            ext_anc_id_2_meas_gate.clear()
            self.clock_move_forward
            # for data in self.lv1_code.data_qubits:
            #     self.set_logic_op(data.patch_id, LogicalOp.LSE)
            # self.do_next_SE()
            # self.clock_move_forward
        self._commit_lv1_obs_gauge_fixing()
        self.lv1_cycle_time_move_forward
            
    @property
    def active_lv1qubits(self) -> list[Lv1Qubit]:
        ret_qubits = []
        for lv1_qubit in self.lv1_code.distinct_pos_2_qubit.values():
            if lv1_qubit.code_patch.active == True:
                ret_qubits.append(lv1_qubit.code_patch)
        if self.external_cores is not None:
            for qubit in self.external_cores.external_cores:
                if qubit.code_patch.active == True:
                    ret_qubits.append(qubit.code_patch)
        return ret_qubits
    
    @property
    def lv1_cycle_time_move_forward(self):
        self.lv1_circuit.lv1_cycle_time += 1

    @property
    def current_lv1_time(self) -> Lv1Time:
        return Lv1Time(self.lv1_circuit.lv1_cycle_time,
                       self.time)

    def _set_LSE_for_all_active_qubits(self):
        for qubit in self.active_lv1qubits:
            self.set_logic_op(qubit.patch_id, LogicalOp.LSE)
    
    def LSE_rounds_for_all_active_qubits(self, num_rounds: int):
        for _ in range(num_rounds):
            self._set_LSE_for_all_active_qubits()
            self.do_next_SE()
            self.clock_move_forward

    def handle_external_CNOT_instructions(self, instructions: list[CoreInstruction]):
        for instruction in instructions:
            if instruction.target_type != CoreType.INTCORE:
                target_qubit = self.external_cores.pos_2_core[instruction.target_pos]
            else:
                target_qubit = self.lv1_code.distinct_pos_2_qubit[instruction.target_pos]
            if instruction.core_operation == CoreOp.CX:
                if instruction.ctrl_type != CoreType.INTCORE:
                    ctrl_qubit = self.external_cores.pos_2_core[instruction.ctrl_pos]
                else:
                    ctrl_qubit = self.lv1_code.distinct_pos_2_qubit[instruction.ctrl_pos]
                if ctrl_qubit.core_type == CoreType.INTCORE:
                    assert target_qubit.core_type == CoreType.ZBUS
                    self.set_inter_cnot(target_qubit.patch_id, ctrl_qubit.patch_id, 'TC', port=instruction.target_port)
                    self._register_inter_cnots(target_qubit.patch_id, ctrl_qubit.patch_id, 'TC', port=instruction.target_port)
                else:
                    assert ctrl_qubit.core_type == CoreType.XBUS
                    assert target_qubit.core_type == CoreType.INTCORE
                    self.set_inter_cnot(ctrl_qubit.patch_id, target_qubit.patch_id, 'CT', port=instruction.ctrl_port)
                    self._register_inter_cnots(ctrl_qubit.patch_id, target_qubit.patch_id, 'CT', port=instruction.ctrl_port)
            else:
                raise NotImplementedError(f'{instruction=} Not implemented.')
        self.commit_inter_cnots()
        pass

    def _register_patch_reset(self, patch_id: int, reset_gate: Gate):
        '''
        Add new reset event to the patch. 
        
        :param patch_id: 
        :type patch_id: int
        :param reset_gate: 
        :type reset_gate: Gate
        '''
        if reset_gate not in ACTIVATE_INSTRUCTIONS:
            raise TypeError(f'{reset_gate=} should be in {ACTIVATE_INSTRUCTIONS=}')
        lv1_qubit = self.patch_2_lv1_qubit[patch_id]
        self.lv1_circuit.patch_2_timeline[patch_id].add_new_event(
            Lv1_Op(reset_gate, 
                   targets=tuple([Lv1ID_port(patch_id,i) 
                                    for i in lv1_qubit.code_patch.code.ports])),
            time=self.current_lv1_time
        )


    def _register_anc_patch_reset(self, patch_id: int, reset_gate: Gate, unpaired: bool = False):
        '''
        An lv-1 ancilla of the lv-1 circuit is reset to perform lv-1 SE.
        1. Add new reset event to the patch; 
        2. Auto create a new lv-1 detector instance.
        '''
        lv1_qubit = self.patch_2_lv1_qubit[patch_id]
        if lv1_qubit.core_type not in [CoreType.XBUS, CoreType.ZBUS]:
            raise TypeError(f'{patch_id=} has type {lv1_qubit.core_type} '
                            f'which should be a bus')
        self.lv1_circuit.patch_2_timeline[patch_id].add_new_event(
            Lv1_Op(reset_gate, 
                   targets=tuple([Lv1ID_port(patch_id,i) 
                                    for i in lv1_qubit.code_patch.code.ports])),
            time=self.current_lv1_time
        )
        if reset_gate not in ACTIVATE_INSTRUCTIONS:
            raise TypeError(f'{reset_gate=} should be in {ACTIVATE_INSTRUCTIONS=}')
        if reset_gate in [Gate.RX, Gate.MPP_RX]:
            supp = set([(qubit.circuit_id, Pauli.X) for qubit in 
                    lv1_qubit.code_patch.code.logical_X_operator])
        elif reset_gate in [Gate.RZ, Gate.MPP_RZ]:
            supp = set([(qubit.circuit_id, Pauli.Z) for qubit in
                    lv1_qubit.code_patch.code.logical_Z_operator])
        else:
            raise RuntimeError(f'Lv1Det auto generation failed. Reset opeartion {reset_gate} '
                               f'is not specified a specific basis.')
        new_det = Lv1Det(self.lv1_circuit.lv1Det_id_new,supp,
                         anc_patch_id=patch_id,
                         pos=lv1_qubit.distinct_position,
                         unpaired=unpaired
                         )
        for lv1_det_id in self.lv1_circuit\
                          .patch_2_generated_active_lv1Det_ids[patch_id]:
            lv1_det = self.lv1_circuit.lv1Det_id_2_lv1Det[lv1_det_id]
            if lv1_det.anc_patch_id != patch_id:
                continue
            lv1_det.supp_update(supp)
            if self._test_support_on_patch(lv1_det.supp, patch_id):
                self.lv1_circuit.add_new_lv1_det_to_patch(lv1_det, patch_id)
        self.lv1_circuit.generate_new_lv1_Det_id
        self.lv1_circuit.lv1Det_id_2_lv1Det[new_det.id] = new_det
        self.lv1_circuit.introduce_new_lv1_det_from_patch(patch_id, new_det)
        self.lv1_circuit.add_new_lv1_det_to_patch(new_det, patch_id)
        
    def _register_inter_reversed_cnots(self, bus_id: int, core_id: int,
                                cnot_direction: Literal['CT','TC'] = 'TC'):
        '''
        Docstring for _register_inter_reversed_cnots
        
        :param bus_id: Description
        :type bus_id: int
        :param core_id: Description
        :type core_id: int
        :param cnot_direction: Description
        :type cnot_direction: Literal['CT', 'TC']
        '''
        bus = self.patch_2_lv1_qubit[bus_id]
        extcore = self.patch_2_lv1_qubit[core_id]
        assert extcore.core_type == CoreType.EXTCORE
        assert isinstance(extcore.code_patch.code, Rotated_surface_code)
        if cnot_direction == 'CT':
            assert bus.core_type == CoreType.XBUS
            lv1_op = Lv1_Op(Gate.CX, (Lv1ID_port(bus_id, 0),
                                      Lv1ID_port(core_id, 0)))
        else:
            assert cnot_direction == 'TC'
            assert bus.core_type == CoreType.ZBUS
            lv1_op = Lv1_Op(Gate.CX, (Lv1ID_port(core_id, 0),
                                      Lv1ID_port(bus_id, 0)))
        self.lv1_circuit.patch_2_timeline[bus_id].add_new_event(lv1_op, self.current_lv1_time)
        self.lv1_circuit.patch_2_timeline[core_id].add_new_event(lv1_op, self.current_lv1_time)
        core_pos_2_bus_pos = extcore.code_patch.code.reverse_dock(bus.code_patch.code)
        bus_pos_2_core_pos = {core_pos_2_bus_pos[core_pos]: core_pos 
                                    for core_pos in core_pos_2_bus_pos}
        core_id_2_bus_id = {}
        for core_pos in core_pos_2_bus_pos:
            core_id_2_bus_id[extcore.code_patch.code
                             .position_2_qubit[core_pos].circuit_id] \
                = bus.code_patch.code.position_2_qubit[
                            core_pos_2_bus_pos[core_pos]].circuit_id
            core_qubit = extcore.code_patch.code.position_2_qubit[core_pos]
            if core_qubit.qubit_type in [QubitType.X_ancilla, QubitType.Z_ancilla]:
                core_qubit.tag_buffer.aftercnot = True
                core_qubit.tag_buffer.cnot = 'C' if cnot_direction == 'TC' else 'T'
            bus_qubit = bus.code_patch.code.position_2_qubit[
                                            core_pos_2_bus_pos[core_pos]]
            if bus_qubit.qubit_type in [QubitType.X_ancilla, QubitType.Z_ancilla]:
                bus_qubit.tag_buffer.aftercnot = True
                bus_qubit.tag_buffer.cnot = 'T' if cnot_direction == 'TC' else 'C'
        bus_id_2_core_id = {core_id_2_bus_id[core_id]: core_id for core_id in core_id_2_bus_id}
        if cnot_direction == 'CT':
            bus_to_core_basis = Pauli.X
            core_to_bus_basis = Pauli.Z
        else:
            bus_to_core_basis = Pauli.Z
            core_to_bus_basis = Pauli.X
        # bus lv1_det propagation to external core
        for lv1_det_id in self.lv1_circuit.patch_2_lv1Det_ids[bus_id]:
            lv1_det = self.lv1_circuit.lv1Det_id_2_lv1Det[lv1_det_id]
            supp_bus = self._extract_lines_of_logical_pauli(lv1_det.supp,
                                                bus, bus_to_core_basis)
            assert len(supp_bus) <= 1
            for logical_line in supp_bus:
                propagated_line = []
                for pauli_source in logical_line:
                    if pauli_source[0] not in bus_id_2_core_id:
                        continue
                    propagated_line.append((bus_id_2_core_id[pauli_source[0]],
                                              bus_to_core_basis))
                reference_line, sign_fixing_meas_temp \
                    = self._convert_to_reference_logical_line(core_id, propagated_line)
                lv1_det.supp_update(reference_line)
                lv1_det.meas_update(sign_fixing_meas_temp)
            if self._test_support_on_patch(lv1_det.supp, core_id):
                self.lv1_circuit.add_new_lv1_det_to_patch(lv1_det, core_id)
            else:
                self.lv1_circuit.remove_lv1_det_from_patch(lv1_det, core_id)
        # bus lv1_obs propagation to external core
        for lv1_obs_id in self.lv1_circuit.patch_2_lv1Obs_ids[bus_id]:
            lv1_obs = self.lv1_circuit.lv1Obs_id_2_lv1Obs[lv1_obs_id]
            supp_bus = self._extract_lines_of_logical_pauli(lv1_obs.supp,
                                                bus, bus_to_core_basis)
            assert len(supp_bus) <= 1
            for logical_line in supp_bus:
                propagated_line = []
                for pauli_source in logical_line:
                    if pauli_source[0] not in bus_id_2_core_id:
                        continue
                    propagated_line.append((bus_id_2_core_id[pauli_source[0]]))
                reference_line, sign_fixing_meas_temp \
                    = self._convert_to_reference_logical_line(core_id, propagated_line)
                lv1_obs.supp_update(reference_line)
                if len(reference_line) != 0:
                    lv1_obs.lv1_supp_update((core_id, bus_to_core_basis))
                lv1_obs.meas_buff_update(sign_fixing_meas_temp)
            if self._test_support_on_patch(lv1_obs.supp, core_id):
                self.lv1_circuit.link_lv1_obs_to_patch(lv1_obs, core_id)
            else:
                self.lv1_circuit.unlink_lv1_obs_to_patch(lv1_obs, core_id)
        # external core lv1_det propagation to bus
        for lv1_det_id in self.lv1_circuit.patch_2_lv1Det_ids[core_id]:
            lv1_det = self.lv1_circuit.lv1Det_id_2_lv1Det[lv1_det_id]
            supp_core = self._extract_lines_of_logical_pauli(lv1_det.supp,
                                                core_id, core_to_bus_basis)
            for logical_line in supp_core:
                propagated_line = []
                for pauli_source in logical_line:
                    if pauli_source[0] not in core_id_2_bus_id:
                        continue
                    propagated_line.append((core_id_2_bus_id[pauli_source[0]],
                                            core_to_bus_basis))
                reference_line, sign_fixing_meas_temp \
                    = self._convert_to_reference_logical_line(bus_id, propagated_line)
                lv1_det.supp_update(reference_line)
                lv1_det.meas_update(sign_fixing_meas_temp)
            if self._test_support_on_patch(lv1_det.supp, bus_id):
                self.lv1_circuit.add_new_lv1_det_to_patch(lv1_det, bus_id)
            else:
                self.lv1_circuit.remove_lv1_det_from_patch(lv1_det, bus_id)
        # external core lv1_obs to bus
        for lv1_obs_id in self.lv1_circuit.patch_2_lv1Obs_ids[core_id]:
            lv1_obs = self.lv1_circuit.lv1Obs_id_2_lv1Obs[lv1_obs_id]
            supp_core = self._extract_lines_of_logical_pauli(lv1_obs.supp,
                                                core_id, core_to_bus_basis)
            for logical_line in supp_core:
                propagated_line = []
                for pauli_source in logical_line:
                    if pauli_source[0] not in core_id_2_bus_id:
                        continue
                    propagated_line.append((core_id_2_bus_id[pauli_source[0]],
                                            core_to_bus_basis))
                reference_line, sign_fixing_meas_temp \
                    = self._convert_to_reference_logical_line(bus_id, propagated_line)
                lv1_obs.supp_update(reference_line)
                if len(reference_line) != 0:
                    lv1_obs.lv1_supp_update([(bus_id, core_to_bus_basis)])
                lv1_obs.meas_buff_update(sign_fixing_meas_temp)
            if self._test_support_on_patch(lv1_obs.supp, bus_id):
                self.lv1_circuit.link_lv1_obs_to_patch(lv1_obs, bus_id)
            else:
                self.lv1_circuit.unlink_lv1_obs_to_patch(lv1_obs, bus_id)

    def _register_inter_cnots(self, anc_patch: int, data_patch: int,
                              ctrl_targ: Literal['CT','TC'], port: int):
        '''
        When an lv-1 data interacts with an lv-1 ancilla via a CNOT gate: 
        1. Add new CNOT event to both ancilla and data patch;
        2. Update lv-1 detectors connected to either the data or the ancilla

            * detecting region update
            * append gauge measurement results from ancilla
        '''
        lv1_anc = self.patch_2_lv1_qubit[anc_patch]
        lv1_data = self.patch_2_lv1_qubit[data_patch]
        if lv1_anc.core_type not in [CoreType.XBUS, CoreType.ZBUS]:
            raise TypeError(f'{anc_patch=} has type {lv1_anc.core_type=} '
                            f'which is not an Ancilla Bus type')
        if lv1_data.core_type != CoreType.INTCORE:
            raise TypeError(f'{data_patch=} has type {lv1_data.core_type=} '
                            f'which is not the Data Core type.')
        if ctrl_targ == 'CT':
            lv1_op = Lv1_Op(Gate.CX, (Lv1ID_port(anc_patch,port),
                                      Lv1ID_port(data_patch,0)))
        else:
            assert ctrl_targ == 'TC'
            lv1_op = Lv1_Op(Gate.CX, (Lv1ID_port(data_patch,0),
                                      Lv1ID_port(anc_patch,port)))
        self.lv1_circuit.patch_2_timeline[anc_patch].add_new_event(
            lv1_op, self.current_lv1_time
        )
        self.lv1_circuit.patch_2_timeline[data_patch].add_new_event(
            lv1_op, self.current_lv1_time
        )
        anc_id_2_data_id: dict[int,int] = {}
        anc_pos_2_data_pos = self.code_patches[anc_patch].code.dock(
            self.code_patches[data_patch].code,
            port
        )
        for anc_pos in anc_pos_2_data_pos:
            anc_id_2_data_id[lv1_anc.code_patch.code
                              .position_2_qubit[anc_pos].circuit_id] \
                = lv1_data.code_patch.code.position_2_qubit[
                                anc_pos_2_data_pos[anc_pos]].circuit_id
            anc_qubit = lv1_anc.code_patch.code.position_2_qubit[anc_pos]
            if anc_qubit.qubit_type in [QubitType.X_ancilla, QubitType.Z_ancilla]:
                anc_qubit.tag_buffer.aftercnot = True
                anc_qubit.tag_buffer.cnot = 'C' if ctrl_targ == 'CT' else 'T'
            data_qubit = lv1_data.code_patch.code.position_2_qubit[
                anc_pos_2_data_pos[anc_pos]
            ]
            if data_qubit.qubit_type in [QubitType.X_ancilla, QubitType.Z_ancilla]:
                data_qubit.tag_buffer.aftercnot = True
                data_qubit.tag_buffer.cnot = 'T' if ctrl_targ == 'CT' else 'C'
        data_id_2_anc_id = {anc_id_2_data_id[anc_id]:anc_id 
                            for anc_id in anc_id_2_data_id}
        if ctrl_targ == 'CT':
            anc_to_data_basis = Pauli.X
            data_to_anc_basis = Pauli.Z
        else:
            assert ctrl_targ == 'TC'
            anc_to_data_basis = Pauli.Z
            data_to_anc_basis = Pauli.X
        # anc to data
        for lv1_det_id in self.lv1_circuit.patch_2_lv1Det_ids[anc_patch]:
            lv1_det = self.lv1_circuit.lv1Det_id_2_lv1Det[lv1_det_id]
            supp_ctrl = self._extract_lines_of_logical_pauli(lv1_det.supp,
                                                anc_patch, anc_to_data_basis)
            propagate_targ: list[tuple[int,Pauli]] = []
            for logical_line in supp_ctrl:
                for target in logical_line:
                    if target[0] not in anc_id_2_data_id:
                        continue
                    propagate_targ.append((anc_id_2_data_id[target[0]],
                                            anc_to_data_basis))
            lv1_det.supp_update(propagate_targ)
            if self._test_support_on_patch(lv1_det.supp, data_patch):
                self.lv1_circuit.add_new_lv1_det_to_patch(lv1_det, data_patch)
            else:
                self.lv1_circuit.remove_lv1_det_from_patch(lv1_det, data_patch)
        # data to anc
        for lv1_det_id in self.lv1_circuit.patch_2_lv1Det_ids[data_patch]:
            lv1_det = self.lv1_circuit.lv1Det_id_2_lv1Det[lv1_det_id]
            supp_targ = self._extract_lines_of_logical_pauli(lv1_det.supp,
                                                data_patch, data_to_anc_basis)
            propagate_ctrl_ref: list[tuple[int,Pauli]] = []
            sign_fixing_meas: list[int] = []
            for logical_line in supp_targ:
                prop_line = []
                for target in logical_line:
                    assert target[0] in data_id_2_anc_id
                    prop_line.append((data_id_2_anc_id[target[0]],
                                        data_to_anc_basis))
                reference_line, sign_fixing_meas_temp \
                    = self._convert_to_reference_logical_line(
                        anc_patch, prop_line
                    )
                propagate_ctrl_ref += reference_line
                sign_fixing_meas += sign_fixing_meas_temp
            lv1_det.supp_update(propagate_ctrl_ref)
            lv1_det.meas_update(sign_fixing_meas)
            if self._test_support_on_patch(lv1_det.supp, anc_patch):
                self.lv1_circuit.add_new_lv1_det_to_patch(lv1_det, anc_patch)
            else:
                self.lv1_circuit.remove_lv1_det_from_patch(lv1_det, anc_patch)
        # data obs to anc
        for lv1_obs_id in self.lv1_circuit.patch_2_lv1Obs_ids[data_patch]:
            lv1_obs = self.lv1_circuit.lv1Obs_id_2_lv1Obs[lv1_obs_id]
            supp_targ = self._extract_lines_of_logical_pauli(lv1_obs.supp,
                                                data_patch, data_to_anc_basis)
            for logical_line in supp_targ:
                prop_line = []
                for target in logical_line:
                    assert target[0] in data_id_2_anc_id
                    prop_line.append((data_id_2_anc_id[target[0]],
                                        data_to_anc_basis))
                reference_line, sign_fixing_meas_temp \
                    = self._convert_to_reference_logical_line(
                        anc_patch, prop_line
                    )
                lv1_obs.supp_update(reference_line)
                if len(reference_line) != 0:
                    lv1_obs.lv1_supp_update([(anc_patch, data_to_anc_basis)])
                lv1_obs.meas_buff_update(sign_fixing_meas_temp)
                sign_fixing_meas += sign_fixing_meas_temp
        # bus obs to core
        for lv1_obs_id in self.lv1_circuit.patch_2_lv1Obs_ids[anc_patch]:
            lv1_obs = self.lv1_circuit.lv1Obs_id_2_lv1Obs[lv1_obs_id]
            supp_core = self._extract_lines_of_logical_pauli(lv1_obs.supp,
                                                anc_patch, anc_to_data_basis)
            for logical_line in supp_core:
                propagated_line = []
                for pauli_source in logical_line:
                    if pauli_source[0] not in anc_id_2_data_id:
                        continue
                    propagated_line.append((anc_id_2_data_id[pauli_source[0]],
                                            anc_to_data_basis))
                reference_line, sign_fixing_meas_temp \
                    = self._convert_to_reference_logical_line(data_patch, propagated_line)
                lv1_obs.supp_update(reference_line)
                if len(reference_line) != 0:
                    lv1_obs.lv1_supp_update([(data_patch, anc_to_data_basis)])
                lv1_obs.meas_buff_update(sign_fixing_meas_temp)
            if self._test_support_on_patch(lv1_obs.supp, data_patch):
                self.lv1_circuit.link_lv1_obs_to_patch(lv1_obs, data_patch)
            else:
                self.lv1_circuit.unlink_lv1_obs_to_patch(lv1_obs, data_patch)


    def _commit_lv1_obs_gauge_fixing(self):
        for lv1_obs in self.lv1_circuit.lv1Obs_id_2_lv1Obs.values():
            self._add_observable_include_from_lv1_obs(lv1_obs)

    def _register_patch_meas(self, patch_id: int, meas_gate: Gate):
        '''
        Add new meas event to the patch
        
        :param patch_id: 
        :type patch_id: int
        :param meas_gate: 
        :type meas_gate: Gate
        '''
        if meas_gate not in DEACTIVATE_INSTRUCTIONS:
            raise TypeError(f'{meas_gate=} should be in {DEACTIVATE_INSTRUCTIONS=}')
        lv1_anc = self.patch_2_lv1_qubit[patch_id]
        meas_op = Lv1_Op(
            meas_gate, 
            targets=tuple([Lv1ID_port(patch_id,i) 
                           for i in lv1_anc.code_patch.code.ports])
            )
        self.lv1_circuit.patch_2_timeline[patch_id].add_new_event(
            meas_op,
            time=self.current_lv1_time
        )
        

    def _register_anc_meas(self, patch_id: int, meas_gate: Gate):
        '''
        When an ancilla patch is measured:
        1. Add new meas event to the patch;
        2. Verify all detecting regions on this patch are terminated.
        3. Generate lv-1 detectors from completely terminated det regions.
        4. Generate lv-1 observables from completely terminated obs regions.
        '''
        lv1_anc = self.patch_2_lv1_qubit[patch_id]
        if lv1_anc.core_type not in [CoreType.XBUS, CoreType.ZBUS]:
            raise TypeError(f'{patch_id=} has type {lv1_anc.core_type} '
                            f'which should be an ancilla')
        meas_op = Lv1_Op(
            meas_gate, 
            targets=tuple([Lv1ID_port(patch_id,i) 
                           for i in lv1_anc.code_patch.code.ports])
            )
        if meas_gate in [Gate.MX, Gate.MZ]:
            meas_time = Lv1Time(self.current_lv1_time.lv1_cycle_time,
                                self.current_lv1_time.clock_time+1)
        else:
            meas_time = self.current_lv1_time
        self.lv1_circuit.patch_2_timeline[patch_id].add_new_event(
            meas_op,
            time=meas_time
        )
        if meas_gate not in DEACTIVATE_INSTRUCTIONS:
            raise TypeError(f'{meas_gate=} should be in {DEACTIVATE_INSTRUCTIONS=}')
        if meas_gate in [Gate.MX, Gate.MPP_MX]:
            meas_basis = Pauli.X
        else:
            assert meas_gate in [Gate.MZ, Gate.MPP_MZ]
            meas_basis = Pauli.Z
        det_ids_connected_to_patch = deepcopy(
                    self.lv1_circuit.patch_2_lv1Det_ids[patch_id])
        for lv1_det_id in det_ids_connected_to_patch:
            lv1_det = self.lv1_circuit.lv1Det_id_2_lv1Det[lv1_det_id]
            supp = self._get_qubit_supp_on_patch(lv1_det.supp, patch_id)
            if supp:
                lv1_det.add_lv1_meas(meas_time, meas_op)
            for target in supp:
                if target[-1] != meas_basis:
                    raise ValueError(f'{lv1_det=} gets annihilated by '
                        f'{meas_basis} basis measurement on patch {patch_id}')
                meas = self.qubit_id_2_meas_records[target[0]]\
                        .time_basis_2_meas_id[(self.time,meas_basis)]
                lv1_det.meas_update([meas])
            lv1_det.supp_update(supp)
            # if lv1_det.anc_patch_id != patch_id:
            self.lv1_circuit.remove_lv1_det_from_patch(lv1_det, patch_id)
            if len(lv1_det.supp) == 0:
                # self.lv1_circuit.remove_lv1_det_from_patch(lv1_det,
                #                                                patch_id)
                self.lv1_circuit.deactivate_lv1_det(lv1_det.id)
                self.circuit.detector(lv1_det.meas_ids,
                                (999,) + lv1_det.pos
                                + (self.current_lv1_time.lv1_cycle_time,))
        obs_ids_connected_to_patch = deepcopy(self.lv1_circuit.patch_2_lv1Obs_ids[patch_id])
        for obs_id in obs_ids_connected_to_patch:
            obs = self.lv1_circuit.lv1Obs_id_2_lv1Obs[obs_id]
            supp = self._get_qubit_supp_on_patch(obs.supp, patch_id)
            if supp:
                obs.add_lv1_meas(meas_time, meas_op)
            for target in supp:
                if target[-1] != meas_basis:
                    raise ValueError(f'{obs=} gets annihilated by '
                        f'{meas_basis} basis measurement on patch {patch_id}')
                meas = self.qubit_id_2_meas_records[target[0]]\
                        .time_basis_2_meas_id[(self.time,meas_basis)]
                obs.meas_buff_update([meas])
            obs.supp_update(supp)
            if supp:
                obs.lv1_supp_update([(patch_id, meas_basis)])
                self.lv1_circuit.remove_lv1_obs_from_patch(obs, patch_id)
                self._add_observable_include_from_lv1_obs(obs)

    def _extract_lines_of_logical_pauli(
            self, 
            qubit_supp: Iterable[tuple[int,Pauli]],
            patch_id: int, 
            logical_basis: Pauli
        ) -> list[list[tuple[int,Pauli]]]:
        '''
        Assuming the ```qubit_supp``` becomes a logical operator 
        with ```logical_basis``` after restricted to patch with ```patch_id```, 
        return a ```list``` of lines of logical operator such that each line 
        is a vertical or horizontal line of qubits that compose a logical operator
        with ```logical_basis```.                
        '''
        if logical_basis not in [Pauli.X, Pauli.Z]:
            raise TypeError(f'{logical_basis=} should be either X or Z.')
        restricted_supp = self._get_qubit_supp_on_patch(qubit_supp, patch_id)
        restricted_basis_supp = [supp for supp in restricted_supp 
                                    if supp[-1]==logical_basis]
        lines: list[list[tuple[int,Pauli]]] = []
        while restricted_basis_supp:
            first_id = restricted_basis_supp[0][0]
            first_pos = self.code_patches[patch_id].qubit_id_2_pos[first_id]
            logical_line = self.code_patches[patch_id].code\
                .logical_pauli_line_through_qubit(first_pos, logical_basis)
            line_supp = [(self.code_patches[patch_id]
                          .code.position_2_qubit[pos].circuit_id, logical_basis)
                          for pos in logical_line]
            lines.append(line_supp)
            for supp in line_supp:
                if supp in restricted_basis_supp:
                    restricted_basis_supp.remove(supp)
                else:
                    raise ValueError(f'{restricted_basis_supp=} is expected to be '
                                     f'a logical operator.')
        return lines
    
    def _test_support_on_patch(self, qubit_supp: Iterable[tuple[int,Pauli]],
                               patch_id: int) -> bool:
        for target in qubit_supp:
            if target[0] in self.dummy_qubits:
                continue
            if patch_id in self.qubit_id_2_patch_ids[target[0]]:
                return True
        return False
    
    def _convert_to_reference_logical_line(
            self, patch_id: int, 
            logical_supp: list[tuple[int,Pauli]]
        ) -> tuple[list[tuple[int,Pauli]],list[int]]:
        '''
        Given a logical line operator of a ShuttleBus along its short direction, 
        we can convert this operator to the reference logical operator along
        the left edge of the ShuttleBus by multiplying stabilizers sandwiched 
        between the input logical line operator and the reference operator.

        :Return: `tuple` of the reference operator and the `list` of stabilizer
        measurement results that fix the sign of the sandwiched stabilizers.  
        
        :param self: Description
        :param patch_id: Description
        :type patch_id: int
        :param logical_supp: Description
        :type logical_supp: list[tuple[int, Pauli]]
        :return: Description
        :rtype: tuple[list[tuple[int, Pauli]], list[int]]
        '''
        lv1_qubit = self.patch_2_lv1_qubit[patch_id]
        if len(logical_supp) == 0:
            raise ValueError(f'{logical_supp=} is empty')
        # first test whether logical_supp is a correct logical line operator 
        pauli_register = logical_supp[0][-1]
        first_pos = lv1_qubit.code_patch.qubit_id_2_pos[logical_supp[0][0]]
        for target in logical_supp:
            if target[-1] != pauli_register:
                raise ValueError(f'{logical_supp=} does not have a unified '
                                 f'Pauli type.')
        extracted_lines = self._extract_lines_of_logical_pauli(logical_supp,
                                                    patch_id, pauli_register)
        if len(extracted_lines) != 1 \
            or set(extracted_lines[0]) != set(logical_supp):
            raise ValueError(f'{logical_supp=} is not a valid logical '
                             f'operator on patch {patch_id}. '
                             f'{extracted_lines=}')
        if pauli_register == Pauli.Z:
            reference_supp = [(qubit.circuit_id,Pauli.Z) for qubit 
                              in lv1_qubit.code_patch.code.logical_Z_operator]
        else:
            assert pauli_register == Pauli.X
            reference_supp = [(qubit.circuit_id,Pauli.X) for qubit 
                              in lv1_qubit.code_patch.code.logical_X_operator]
        sandwiched_stabs: list[Qubit] = []
        stab_type = QubitType.X_ancilla if pauli_register == Pauli.X\
                    else QubitType.Z_ancilla
        horizontal_or_vertical = 0
        if lv1_qubit.core_type == CoreType.XBUS and pauli_register == Pauli.X:
            horizontal_or_vertical = 1
        elif lv1_qubit.core_type == CoreType.ZBUS and pauli_register == Pauli.Z:
            horizontal_or_vertical = 1
        elif lv1_qubit.core_type in [CoreType.INTCORE, CoreType.EXTCORE] \
            and pauli_register == Pauli.Z:
                horizontal_or_vertical = 1
        for qubit in lv1_qubit.code_patch.code.qubits:
            if qubit.qubit_type == stab_type:
                if qubit.position[horizontal_or_vertical] < first_pos[horizontal_or_vertical]:
                    sandwiched_stabs.append(qubit)
        sign_fixing_meas = []
        most_recent_anc_reset_time, reset_gate = self.lv1_circuit\
                                        .patch_2_timeline[lv1_qubit.patch_id]\
                                        .most_recent_activation_time_and_gate
        if reset_gate == Gate.RX and pauli_register == Pauli.X:
            return (reference_supp, [])
        if reset_gate == Gate.RZ and pauli_register == Pauli.Z:
            return (reference_supp, [])
        if most_recent_anc_reset_time is not None:
            for qubit in sandwiched_stabs:
                qubit_reset_basis = Pauli.Z \
                        if qubit.qubit_type == QubitType.Z_ancilla else Pauli.X
                sign_fixing_meas.append(
                    self.qubit_id_2_meas_records[qubit.circuit_id].time_basis_2_meas_id[
                        (most_recent_anc_reset_time.clock_time, qubit_reset_basis)
                    ]
                )
        return (reference_supp, sign_fixing_meas)
        

    
    def _convert_to_reference_logical_line_deprecated(
            self, 
            anc_patch: int, 
            logical_supp: list[tuple[int,Pauli]]
        ) -> tuple[list[tuple[int,Pauli]],list[int]]:
        '''
        Given a logical line operator of a ShuttleBus along its short direction, 
        we can convert this operator to the reference logical operator along
        the left edge of the ShuttleBus by multiplying stabilizers sandwiched 
        between the input logical line operator and the reference operator.

        :Return: `tuple` of the reference operator and the `list` of stabilizer
        measurement results that fix the sign of the sandwiched stabilizers.  
        '''
        lv1_anc = self.patch_2_lv1_qubit[anc_patch]
        if lv1_anc.core_type not in [CoreType.ZBUS, CoreType.XBUS]:
            raise TypeError(f'{anc_patch=} has type {lv1_anc.core_type=} '
                            f'which is not an ancilla type')
        if len(logical_supp) == 0:
            raise ValueError(f'{logical_supp=} is empty')
        # first test whether logical_supp is a correct logical line operator 
        pauli_register = logical_supp[0][-1]
        first_pos = lv1_anc.code_patch.qubit_id_2_pos[logical_supp[0][0]]
        for target in logical_supp:
            if target[-1] != pauli_register:
                raise ValueError(f'{logical_supp=} does not have a unified '
                                 f'Pauli type.')
        extracted_lines = self._extract_lines_of_logical_pauli(logical_supp,
                                                    anc_patch, pauli_register)
        if len(extracted_lines) != 1 \
            or set(extracted_lines[0]) != set(logical_supp):
            raise ValueError(f'{logical_supp=} is not a valid logical '
                             f'operator on patch {anc_patch}. '
                             f'{extracted_lines=}')
        if lv1_anc.core_type == CoreType.XBUS:
            reference_supp = [(qubit.circuit_id,Pauli.Z) for qubit 
                              in lv1_anc.code_patch.code.logical_Z_operator]
            if pauli_register != Pauli.Z:
                raise TypeError(f'{logical_supp=} is not a logical Z operator '
                            f'on an X ancilla patch')
        if lv1_anc.core_type == CoreType.ZBUS:
            reference_supp = [(qubit.circuit_id,Pauli.X) for qubit 
                              in lv1_anc.code_patch.code.logical_X_operator]
            if pauli_register != Pauli.X:
                raise TypeError(f'{logical_supp=} is not a logical X operator '
                            f'on a Z ancilla patch')
        sandwiched_stabs: list[Qubit] = []
        stab_type = QubitType.X_ancilla if pauli_register == Pauli.X\
                    else QubitType.Z_ancilla
        for qubit in lv1_anc.code_patch.code.qubits:
            if qubit.qubit_type == stab_type:
                if qubit.position[0] < first_pos[0]:
                    sandwiched_stabs.append(qubit)
        sign_fixing_meas = []
        most_recent_anc_reset_time = self.lv1_circuit\
                                        .patch_2_timeline[anc_patch]\
                                        .most_recent_activation_time
        reset_basis = Pauli.X if lv1_anc.core_type == CoreType.XBUS \
                        else Pauli.Z
        if most_recent_anc_reset_time is not None:
            for qubit in sandwiched_stabs:
                qubit_reset_basis = Pauli.Z \
                        if qubit.qubit_type == QubitType.Z_ancilla else Pauli.X
                sign_fixing_meas.append(
                    self.qubit_id_2_meas_records[qubit.circuit_id].time_basis_2_meas_id[
                        (most_recent_anc_reset_time.clock_time, qubit_reset_basis)
                    ]
                )
        return (reference_supp, sign_fixing_meas)
    
    def _add_observable_include_from_lv1_obs(self, lv1_obs: Lv1Obs):
        if len(lv1_obs.meas_ids_buff) == 0:
            return 
        self.circuit.observable_include(lv1_obs.meas_ids_buff, lv1_obs.obs_id)
        lv1_obs.meas_history_update()
        lv1_obs.clear_meas_buff()