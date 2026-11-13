from lv1_codes import *
from lv1_utils import *
from circuit_interface import *

@dataclass
class Lv1DummyQubit:
    id: int
    x_supp: list[int] = field(default_factory=list)
    z_supp: list[int] = field(default_factory=list)
    pos: Optional[tuple[int,...]] = None
    circuit_id: Optional[int] = None
    reset: Lv1LogicalRESET = Lv1LogicalRESET.UNDEFINED

class Lv1Circuit:

    def __init__(self):
        self.patch_2_timeline: dict[int,Lv1Timeline] = {}
        self.lv1Det_id_2_lv1Det: dict[int,Lv1Det] = {}
        self.patch_2_lv1Det_ids: dict[int,set[int]] = {}
        self.patch_2_generated_active_lv1Det_ids: dict[int,set[int]] = {}
        self.lv1Det_id_2_patches: dict[int,set[int]] = {}
        self.lv1Obs_id_2_lv1Obs: dict[int, Lv1Obs] = {}
        self.patch_2_lv1Obs_ids: dict[int, set[int]] = {}
        self.patch_2_pos: dict[int,tuple[int,...]] = {}
        self.pos_2_patch: dict[tuple[int,...],int] = {}
        self.patch_2_coretype: dict[int,CoreType] = {}
        self.lv1_cycle_time: int = 0
        self.lv1Det_id_new: int = 0
        self.lv1Obs_id_new: int = 0
        self.lv1_err_const = 0.0001
        self.lv1_circuit = Circuit_helper()
        self.id_2_core: dict[int,Lv1Qubit] = {}
        self.logical_qubit_id_2_dummy_qubit: dict[int,Lv1DummyQubit] = {}
        self.dummy_pos_2_dummy_qubit: dict[int,Lv1DummyQubit] = {}
        self.time_meas_op_2_meas_id: dict[tuple[Lv1Time,Lv1_Op],int] = {}
        self.external_patches: list[int] = []
        self.external_anc_id_2_obs_id: dict[int,int] = {}
        self.lv1_circ_id_2_circ_pos: dict[int,tuple[int,...]] = {}

    @property
    def generate_new_lv1_Det_id(self):
        self.lv1Det_id_new += 1
    
    def generate_new_lv1_Obs_id(self):
        self.lv1Obs_id_new += 1

    @property
    def sorted_time(self) -> list[Lv1Time]:
        all_time: set[Lv1Time] = set()
        for timeline in self.patch_2_timeline.values():
            all_time.update(timeline.sorted_time)
        return sorted(all_time)
    
    @property
    def time_2_ops(self) -> dict[Lv1Time,set[Lv1_Op]]:
        ret: dict[Lv1Time,set[Lv1_Op]] = {}
        for timeline in self.patch_2_timeline.values():
            for time in timeline.time_2_ops:
                for op in timeline.time_2_ops[time]:
                    if time not in ret:
                        ret[time] = set([op])
                    else:
                        ret[time].add(op)
        return ret
    
    @property
    def time_2_lv1_dets(self) -> dict[Lv1Time,list[Lv1Det]]:
        ret: dict[Lv1Time,list[Lv1Det]] = {}
        for lv1_det in self.lv1Det_id_2_lv1Det.values():
            if lv1_det.is_active:
                continue
            if lv1_det.latest_meas_time is not None:
                if lv1_det.latest_meas_time not in ret:
                    ret[lv1_det.latest_meas_time] = [lv1_det]
                else:
                    ret[lv1_det.latest_meas_time].append(lv1_det)
        return ret
    
    @property
    def time_2_lv1_obs(self) -> dict[Lv1Time,list[Lv1Obs]]:
        ret: dict[Lv1Time,list[Lv1Obs]] = {}
        for lv1_obs in self.lv1Obs_id_2_lv1Obs.values():
            involved_times: set[Lv1Time] = set()
            for time, meas_op in lv1_obs.involved_lv1_meas:
                involved_times.add(time)
            for time in involved_times:
                if time not in ret:
                    ret[time] = [lv1_obs]
                else:
                    ret[time].append(lv1_obs)
        return ret

    
    @property
    def active_lv1_dets(self) -> list[Lv1Det]:
        active_lv1_det_ids = set()
        for det_ids in self.patch_2_lv1Det_ids.values():
            active_lv1_det_ids.update(det_ids)
        return [self.lv1Det_id_2_lv1Det[id] for id in active_lv1_det_ids]

    def deactivate_lv1_det(self, det_id: int):
        det = self.lv1Det_id_2_lv1Det[det_id]
        patch_origin = det.anc_patch_id
        self.patch_2_generated_active_lv1Det_ids[patch_origin].remove(det_id)
        for patch in self.lv1Det_id_2_patches[det_id]:
            self.patch_2_lv1Det_ids[patch].remove(det_id)
        self.lv1Det_id_2_patches[det_id].clear()
        pass

    def add_new_core(self, core: Lv1Qubit, external: bool = False):
        if core.patch_id in self.id_2_core:
            raise ValueError(f'logical qubit with patch id '
                             f'{core.patch_id} already added')
        self.id_2_core[core.patch_id] = core
        self.add_new_patch(core.patch_id, core.position, core.core_type)
        if core.core_type in [CoreType.XBUS, CoreType.ZBUS]:
            self.patch_2_generated_active_lv1Det_ids[core.patch_id] = set()
        if external:
            self.external_patches.append(core.patch_id)
            
    def add_new_lv1_dummy_qubit(self, lv1_logical_qubit: Lv1LogicalQubit):
        if lv1_logical_qubit.id in self.logical_qubit_id_2_dummy_qubit:
            raise ValueError(f'observable {lv1_logical_qubit.id} already added.')
        x_supp_ids = [self.pos_2_patch[pos] for pos in lv1_logical_qubit.x_supp]
        z_supp_ids = [self.pos_2_patch[pos] for pos in lv1_logical_qubit.z_supp]
        min_y_pos = min([pos[-1] for pos in self.pos_2_patch])
        max_x_pos = max([pos[0] for pos in self.pos_2_patch])
        dummy_pos = (lv1_logical_qubit.id % (max_x_pos + 1), min_y_pos - 1 - (lv1_logical_qubit.id//(max_x_pos+1)))
        new_dummy_qubit = Lv1DummyQubit(id=lv1_logical_qubit.id, 
                                    x_supp=x_supp_ids,
                                    z_supp=z_supp_ids,
                                    pos=dummy_pos)
        self.dummy_pos_2_dummy_qubit[dummy_pos] = new_dummy_qubit
        self.logical_qubit_id_2_dummy_qubit[lv1_logical_qubit.id] = new_dummy_qubit

    def add_new_patch(self, patch_id: int, patch_pos: tuple[int,...], core_type: CoreType):
        if patch_id in self.patch_2_timeline:
            raise ValueError(f'patch {patch_id} already added')
        self.patch_2_timeline[patch_id] = Lv1Timeline(patch_id)
        self.patch_2_lv1Det_ids[patch_id] = set()
        self.patch_2_lv1Obs_ids[patch_id] = set()
        self.patch_2_pos[patch_id] = patch_pos
        self.pos_2_patch[patch_pos] = patch_id
        self.patch_2_coretype[patch_id] = core_type
    
    def introduce_new_lv1_det_from_patch(
            self,
            patch: int, new_det: Lv1Det
            ):
        self.lv1Det_id_2_lv1Det[new_det.id] = new_det
        if patch not in self.patch_2_generated_active_lv1Det_ids:
            self.patch_2_generated_active_lv1Det_ids[patch] = set([new_det.id])
        else:
            self.patch_2_generated_active_lv1Det_ids[patch].add(new_det.id)
        pass

    def introduce_new_lv1_obs(self, new_obs: Lv1Obs):
        if new_obs.obs_id in self.lv1Obs_id_2_lv1Obs:
            raise RuntimeError(f'new lv1-obs with {new_obs.obs_id=} '
                               f'already added.')
        self.lv1Obs_id_2_lv1Obs[new_obs.obs_id] = new_obs

    def link_lv1_obs_to_patch(self, lv1_obs: Lv1Obs, patch: int):
        self.patch_2_lv1Obs_ids[patch].add(lv1_obs.obs_id)

    def unlink_lv1_obs_to_patch(self, lv1_obs: Lv1Obs, patch: int):
        if lv1_obs.obs_id in self.patch_2_lv1Obs_ids[patch]:
            self.patch_2_lv1Obs_ids[patch].remove(lv1_obs.obs_id)

    def add_new_lv1_det_to_patch(self, lv1_det: Lv1Det, patch: int):
        self.patch_2_lv1Det_ids[patch].add(lv1_det.id)
        if lv1_det.id in self.lv1Det_id_2_patches:
            self.lv1Det_id_2_patches[lv1_det.id].add(patch)
        else:
            self.lv1Det_id_2_patches[lv1_det.id] = set([patch])
    
    def add_new_lv1_obs_to_patch(self, lv1_obs: Lv1Obs, patch: int):
        self.patch_2_lv1Obs_ids[patch].add(lv1_obs.obs_id)

    def remove_lv1_det_from_patch(self, lv1_det: Lv1Det, patch: int):
        if lv1_det.id in self.patch_2_lv1Det_ids[patch]:
            self.patch_2_lv1Det_ids[patch].remove(lv1_det.id)
        if patch in self.lv1Det_id_2_patches[lv1_det.id]:
            self.lv1Det_id_2_patches[lv1_det.id].remove(patch)
    
    def remove_lv1_obs_from_patch(self, lv1_obs: Lv1Obs, patch: int):
        if lv1_obs.obs_id in self.patch_2_lv1Obs_ids[patch]:
            self.patch_2_lv1Obs_ids[patch].remove(lv1_obs.obs_id)

    def external_anc_add_meas_obs_id(self, anc_id: int, obs_id: int):
        assert anc_id in self.external_patches
        assert self.patch_2_coretype[anc_id] in [CoreType.ZBUS, CoreType.XBUS]
        assert anc_id not in self.external_anc_id_2_obs_id
        self.external_anc_id_2_obs_id[anc_id] = obs_id
    
    def initalize_lv1_circuit(self):
        patch_id_sorted = sorted(self.id_2_core.keys())
        assert patch_id_sorted == sorted(range(len(patch_id_sorted)))
        distinct_positions = [self.id_2_core[patch_id]
                              .distinct_position for patch_id in patch_id_sorted]
        self.lv1_circuit.introduce_new_qubits(distinct_positions)
        for pos, cid in zip(distinct_positions, patch_id_sorted):
            self.lv1_circ_id_2_circ_pos[cid] = pos
        logical_qubit_id_sorted = sorted(self.logical_qubit_id_2_dummy_qubit.keys())
        assert logical_qubit_id_sorted == sorted(range(len(logical_qubit_id_sorted)))
        dummy_positions = [self.logical_qubit_id_2_dummy_qubit[log_id].pos for log_id in logical_qubit_id_sorted]
        dummy_pos_2_circuit_id = self.lv1_circuit.introduce_new_qubits(dummy_positions)
        for dummy_pos in dummy_pos_2_circuit_id:
            self.dummy_pos_2_dummy_qubit[dummy_pos].circuit_id = dummy_pos_2_circuit_id[dummy_pos]
            self.lv1_circ_id_2_circ_pos[dummy_pos_2_circuit_id[dummy_pos]] = dummy_pos
        for dummy_qubit in self.logical_qubit_id_2_dummy_qubit.values():
            self.patch_2_timeline[dummy_qubit.circuit_id] = Lv1Timeline(dummy_qubit.circuit_id)
            self.patch_2_coretype[dummy_qubit.circuit_id] = CoreType.OBSDUMMY
    
    def generate_lv1_circuit(self, keep_ext_anc_id: Optional[int] = None):
        if keep_ext_anc_id is not None:
            assert keep_ext_anc_id in self.external_patches
            assert self.patch_2_coretype[keep_ext_anc_id] in [CoreType.ZBUS, CoreType.XBUS]
        exclude_targets = []
        for patch_id in self.external_patches:
            if self.patch_2_coretype[patch_id] not in [CoreType.ZBUS, CoreType.XBUS]:
                continue
            if keep_ext_anc_id is not None:
                if patch_id == keep_ext_anc_id:
                    continue
            exclude_targets.append(patch_id)

        time_2_ops = self.time_2_ops
        time_2_lv1_dets = self.time_2_lv1_dets
        core_type_2_name: dict[CoreType, str] = {
            CoreType.EXTCORE: 'EXTDATA',
            CoreType.INTCORE: 'DATA',
            CoreType.XBUS: 'X_ANC',
            CoreType.ZBUS: 'Z_ANC'
        }

        def so_time_interval(so_start_time: int, so_end_time: int) -> int:
            return 1 + (so_end_time + so_end_time%2 - so_start_time) // 2
        
        for time in self.sorted_time:
            ops = sorted(time_2_ops[time])
            for op in ops:
                should_skip = False
                for lv1_target in op.lv1_targets:
                    if lv1_target not in self.external_patches:
                        continue
                    if self.patch_2_coretype[lv1_target] not in [CoreType.ZBUS, CoreType.XBUS]:
                        continue
                    if keep_ext_anc_id is None:
                        should_skip = True
                    elif keep_ext_anc_id != lv1_target:
                        should_skip = True
                if should_skip:
                    continue
                if op.gate in [Gate.MPP_RX, Gate.MPP_MX, 
                               Gate.MPP_RZ, Gate.MPP_MZ]:
                    target_pauli = Pauli.X \
                                if op.gate in [Gate.MPP_RX, Gate.MPP_MX]\
                                else Pauli.Z
                    targets = []
                    for lv1_target in op.lv1_targets:
                        targets.append(stim.target_pauli(lv1_target,target_pauli.basis))
                        targets.append(stim.target_combiner())
                    targets.pop()
                    self.lv1_circuit.append_Instruction(CircuitInstruction(
                        name=op.gate.instruction,
                        targets=targets
                    ))
                elif op.gate in [Gate.MPP, Gate.MPP_RESET, Gate.MPP_MEAS]:
                    raise NotImplementedError(f'{op.gate=} not supported here.')
                else:
                    # Add pre-gate errors
                    if op.gate == Gate.CX:
                        assert self.patch_2_coretype[op.lv1_targets[0]] != CoreType.OBSDUMMY
                        assert self.patch_2_coretype[op.lv1_targets[1]] != CoreType.OBSDUMMY
                        contrl_qubit = self.id_2_core[op.lv1_targets[0]]
                        target_qubit = self.id_2_core[op.lv1_targets[1]]
                        if contrl_qubit.core_type in [CoreType.INTCORE, CoreType.EXTCORE]:
                            assert target_qubit.core_type in [CoreType.XBUS, CoreType.ZBUS]
                            prev_time, found = self.patch_2_timeline[contrl_qubit.patch_id]\
                                                .find_the_previous_same_operation(time, op, exclude_targets)
                            if not found:
                                so_start_time = prev_time.clock_time
                                so_end_time = time.clock_time
                                self.lv1_circuit.append_Instruction(CircuitInstruction(
                                    name=Noise.X_ERROR.instruction,
                                    targets=[contrl_qubit.patch_id],
                                    gate_args=[self.lv1_err_const],
                                    tag=f'patch_id={contrl_qubit.patch_id},'
                                        f'patch_type={core_type_2_name[contrl_qubit.core_type]},'
                                        f'error_type=X,'
                                        f'lv1cycle_time={prev_time.lv1_cycle_time},'
                                        f'clock_time={prev_time.clock_time},'
                                        f'so_start_time={so_start_time},'
                                        f'so_end_time={so_end_time},'
                                        f'time_interval={so_time_interval(so_start_time, so_end_time)}'
                                )) 
                        else:
                            assert target_qubit.core_type in [CoreType.INTCORE, CoreType.EXTCORE]
                            prev_time, found = self.patch_2_timeline[target_qubit.patch_id]\
                                                    .find_the_previous_same_operation(time, op, exclude_targets)
                            if not found:
                                so_start_time = prev_time.clock_time
                                so_end_time = time.clock_time
                                self.lv1_circuit.append_Instruction(CircuitInstruction(
                                    name=Noise.Z_ERROR.instruction,
                                    targets=[target_qubit.patch_id],
                                    gate_args=[self.lv1_err_const],
                                    tag=f'patch_id={target_qubit.patch_id},'
                                        f'patch_type={core_type_2_name[target_qubit.core_type]},'
                                        f'error_type=Z,'
                                        f'lv1cycle_time={prev_time.lv1_cycle_time},'
                                        f'clock_time={prev_time.clock_time},'
                                        f'so_start_time={so_start_time},'
                                        f'so_end_time={so_end_time},'
                                        f'time_interval={so_time_interval(so_start_time, so_end_time)}'
                                ))
                    if op.gate == Gate.MX:
                        core_type = self.patch_2_coretype[op.lv1_targets[0]]
                        assert core_type in [CoreType.EXTCORE, CoreType.XBUS]
                        prev_reset_time, reset_gate = self.patch_2_timeline[op.lv1_targets[0]]\
                                                          .find_the_previous_activation_time_and_gate(time)
                        if core_type == CoreType.XBUS:
                            assert reset_gate == Gate.RX
                        self.lv1_circuit.append_Instruction(CircuitInstruction(
                            name=Noise.Z_ERROR.instruction,
                            targets=op.lv1_targets,
                            gate_args=[self.lv1_err_const],
                            tag=f'patch_id={op.lv1_targets[0]},'
                                f'patch_type={core_type_2_name[core_type]},'
                                f'error_type=Z,'
                                f'lv1cycle_time={time.lv1_cycle_time},'
                                f'clock_time={time.clock_time},'
                                f'so_start_time={prev_reset_time.clock_time},'
                                f'so_end_time={time.clock_time},'
                                f'time_interval={so_time_interval(prev_reset_time.clock_time,time.clock_time)}'
                        ))
                    if op.gate == Gate.MZ:
                        core_type = self.patch_2_coretype[op.lv1_targets[0]]
                        assert core_type in [CoreType.EXTCORE, CoreType.ZBUS]
                        prev_reset_time, reset_gate = self.patch_2_timeline[op.lv1_targets[0]]\
                                                          .find_the_previous_activation_time_and_gate(time)
                        if core_type == CoreType.ZBUS:
                            assert reset_gate == Gate.RZ
                        self.lv1_circuit.append_Instruction(CircuitInstruction(
                            name=Noise.X_ERROR.instruction,
                            targets=op.lv1_targets,
                            gate_args=[self.lv1_err_const],
                            tag=f'patch_id={op.lv1_targets[0]},'
                                f'patch_type={core_type_2_name[core_type]},'
                                f'error_type=X,'
                                f'lv1cycle_time={time.lv1_cycle_time},'
                                f'clock_time={time.clock_time},'
                                f'so_start_time={prev_reset_time.clock_time},'
                                f'so_end_time={time.clock_time},'
                                f'time_interval={so_time_interval(prev_reset_time.clock_time,time.clock_time)}'
                        ))
                    # Add gate
                    self.lv1_circuit.append_Instruction(CircuitInstruction(
                        name=op.gate.instruction, 
                        targets=op.lv1_targets
                    ))
                if op.gate in DEACTIVATE_INSTRUCTIONS + [Gate.MPP_RX, Gate.MPP_RZ]:
                    new_meas_id = self.lv1_circuit.circuit.num_measurements-1
                    self.time_meas_op_2_meas_id[(time,op)] = new_meas_id
                # Add post-gate errors
                if op.gate == Gate.CX:
                    assert self.patch_2_coretype[op.lv1_targets[0]] != CoreType.OBSDUMMY
                    assert self.patch_2_coretype[op.lv1_targets[1]] != CoreType.OBSDUMMY
                    contrl_qubit = self.id_2_core[op.lv1_targets[0]]
                    target_qubit = self.id_2_core[op.lv1_targets[1]]
                    if contrl_qubit.core_type in [CoreType.INTCORE, CoreType.EXTCORE]:
                        assert target_qubit.core_type in [CoreType.XBUS, CoreType.ZBUS]
                        next_time, found = self.patch_2_timeline[contrl_qubit.patch_id]\
                                            .find_the_next_same_operation(time, op, exclude_targets)
                        if found:
                            so_end_time = next_time.clock_time
                        else:
                            so_end_time = next_time.clock_time
                        self.lv1_circuit.append_Instruction(CircuitInstruction(
                            name=Noise.X_ERROR.instruction,
                            targets=[contrl_qubit.patch_id],
                            gate_args=[self.lv1_err_const],
                            tag=f'patch_id={contrl_qubit.patch_id},'
                                f'patch_type={core_type_2_name[contrl_qubit.core_type]},'
                                f'error_type=X,'
                                f'lv1cycle_time={time.lv1_cycle_time},'
                                f'clock_time={time.clock_time},'
                                f'so_start_time={time.clock_time + 2},'
                                f'so_end_time={so_end_time},'
                                f'time_interval={so_time_interval(time.clock_time + 2, so_end_time)}'
                        )) 
                    else:
                        assert target_qubit.core_type in [CoreType.INTCORE, CoreType.EXTCORE]
                        next_time, found = self.patch_2_timeline[target_qubit.patch_id]\
                                                .find_the_next_same_operation(time, op, exclude_targets)
                        if found:
                            so_end_time = next_time.clock_time
                        else:
                            so_end_time = next_time.clock_time
                        self.lv1_circuit.append_Instruction(CircuitInstruction(
                            name=Noise.Z_ERROR.instruction,
                            targets=[target_qubit.patch_id],
                            gate_args=[self.lv1_err_const],
                            tag=f'patch_id={target_qubit.patch_id},'
                                f'patch_type={core_type_2_name[target_qubit.core_type]},'
                                f'error_type=Z,'
                                f'lv1cycle_time={time.lv1_cycle_time},'
                                f'clock_time={time.clock_time},'
                                f'so_start_time={time.clock_time + 2},'
                                f'so_end_time={so_end_time},'
                                f'time_interval={so_time_interval(time.clock_time + 2, so_end_time)}'
                        ))
            if time in time_2_lv1_dets:
                for lv1_det in time_2_lv1_dets[time]:
                    if lv1_det.anc_patch_id in self.external_patches:
                        if keep_ext_anc_id is None:
                            continue
                        elif keep_ext_anc_id != lv1_det.anc_patch_id:
                            continue
                    meas_ids = [self.time_meas_op_2_meas_id[time_meas] 
                                for time_meas in lv1_det.involved_lv1_meas]
                    self.lv1_circuit.detector(
                        meas_ids,
                        detector_pos=lv1_det.pos+(time.lv1_cycle_time,)
                    )
            if time in self.time_2_lv1_obs:
                for obs in self.time_2_lv1_obs[time]:
                    if obs.obs_id in self.external_anc_id_2_obs_id.values():
                        if keep_ext_anc_id is None:
                            continue
                        elif self.external_anc_id_2_obs_id[keep_ext_anc_id] != obs.obs_id:
                            continue
                    time_op_indices = []
                    for index, time_op in enumerate(obs.involved_lv1_meas):
                        if time_op[0] == time:
                            time_op_indices.append(index)
                    if not time_op_indices:
                        continue
                    meas_ids = [self.time_meas_op_2_meas_id[obs.involved_lv1_meas[time_op_index]]
                                     for time_op_index in time_op_indices]
                    self.lv1_circuit.observable_include(meas_ids, obs.obs_id)
            self.lv1_circuit.tick
        pass

    def regenerate_lv1_circuit_w_ext_anc(self, ext_anc_id: Optional[int] = None):
        self.lv1_circuit.clear()
        circ_ids = sorted(self.lv1_circ_id_2_circ_pos.keys())
        positions = [self.lv1_circ_id_2_circ_pos[cid] for cid in circ_ids]
        self.lv1_circuit.introduce_new_qubits(positions)
        self.generate_lv1_circuit(keep_ext_anc_id=ext_anc_id)

    def draw(self, svg_filename: str, *, type = 'timeslice-svg'):
        '''
        **Args**

            type: recommend 'timeslice-svg' (default) or 'detslice-with-ops-svg'
        '''
        with open(svg_filename,'w') as file:
            out = self.lv1_circuit.circuit.diagram(type)
            file.write(out.__str__()) 
