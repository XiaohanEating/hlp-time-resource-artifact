from __future__ import annotations

from typing import Union, Optional, Iterable
from pathlib import Path
from warnings import warn
from enum import Enum, unique

import stim

from low_level_utils import Noise, Pauli, CircuitInstruction, \
                            is_instruction_empty
from gate_types import TagBuffer


class Clock():

    pass


        
class Circuit_helper:
    # Wraper around stim.Circuit to facilitate circuit generation. 
    def __init__(self) -> None:
        self.circuit = stim.Circuit()
        self.position_2_qubit_id: dict[tuple[int,...],int] = {}
        self.header_max_index: int = -1
        self.qubit_id_op_index_2_meas_id: dict[tuple[int,int],int] = {}
        pass

    def clear(self):
        self.circuit.clear()
        self.position_2_qubit_id.clear()
        self.header_max_index = -1
        self.qubit_id_op_index_2_meas_id.clear()

    def append_noise_channel(self, noise: Optional[tuple[Noise,float]],
                      qubit_ids: list[int], *, tag = "") -> None:
        if noise is None:
            return
        noise_instruction = noise[0].instruction
        noise_rate = noise[1]
        self.circuit.append(noise_instruction, qubit_ids, noise_rate, tag=tag) 
    
    def append_Instruction(self, circuit_Instruction: CircuitInstruction):
        if is_instruction_empty(circuit_Instruction):
            return
        self.circuit.append(circuit_Instruction)

    def detector(self, meas_ids: list[int],
                 detector_pos: Optional[tuple[int,...]],
                 tag_buffer: Optional[TagBuffer] = None) -> None:
        meas_rec_ids = sorted([self.measure_id_recollect(meas_id) for meas_id in meas_ids])
        rec_targets = [stim.target_rec(rec_id) for rec_id in meas_rec_ids]
        if tag_buffer is not None:
            self.circuit.append('DETECTOR', rec_targets, detector_pos, tag=tag_buffer.tag)
        else:
            self.circuit.append('DETECTOR', rec_targets, detector_pos)
    
    def observable_include(self, meas_ids: Iterable[int], obs_id: int):
        if len(list(meas_ids)) == 0:
            return
        meas_rec_ids = sorted([self.measure_id_recollect(meas_id) for meas_id in meas_ids])
        rec_targets = [stim.target_rec(rec_id) for rec_id in meas_rec_ids]
        self.circuit.append('OBSERVABLE_INCLUDE', rec_targets, [obs_id])

    def introduce_new_qubits(self, qubit_positions: list[tuple[int,...]]) -> \
                                                        dict[tuple[int,...],int]:
        '''
        New qubits are introduced by their positions. You can introduce new
        qubits mid-circuit. If an input qubit position coincide with an 
        already added qubit position, no new qubit is introduced.
        '''
        pos_2_id: dict[tuple[int,...],int] = {}
        for pos in qubit_positions:
            if pos in self.position_2_qubit_id:
                pos_2_id[pos] = self.position_2_qubit_id[pos]
            else:
                new_id = self.num_qubits
                circInstruction = stim.CircuitInstruction('QUBIT_COORDS',
                                                          [new_id], pos)
                self.position_2_qubit_id[pos] = new_id
                pos_2_id[pos] = new_id
                self.circuit.insert(self.header_max_index+1, circInstruction)
                self.header_max_index += 1
        return pos_2_id
    
    @property
    def length(self) -> int:
        return len(self.circuit.flattened())
    
    def measure(self, meas_basis: Pauli, qubit_ids: list[int],
                noise: Optional[tuple[Noise,float]] = None,
                *, tag = ""):
        self.append_noise_channel(noise, qubit_ids, tag=tag)
        self.circuit.append(f'M{meas_basis.basis}', qubit_ids, tag=tag)
    
    def measure_id_recollect(self, measure_id: int) -> int:
        assert measure_id < self.num_measurements and measure_id >= 0
        return measure_id - self.num_measurements
    
    @property
    def num_qubits(self) -> int:
        return self.circuit.num_qubits
    
    @property
    def num_detectors(self) -> int:
        return self.circuit.num_detectors
    
    @property
    def num_measurements(self) -> int:
        return self.circuit.num_measurements

    def reset(self, reset_basis: Pauli, qubit_ids: list[int],
              noise: Optional[tuple[Noise,float]] = None,
              *, tag = ""):
        self.circuit.append(f'R{reset_basis.basis}', qubit_ids, tag=tag)
        self.append_noise_channel(noise, qubit_ids, tag=tag)
    
    def save_circuit_stim_file_to(self, file_path: Union[str, Path]) -> None:
        self.circuit.to_file(file_path)

    def single_qubit_gate(self, gateName: str, qubit_ids: list[int],
                          noise: Optional[tuple[Noise,float]] = None,
                          *, tag = ""):
        self.circuit.append(gateName, qubit_ids, tag = tag)
        self.append_noise_channel(noise, qubit_ids, tag=tag)

    @property
    def tick(self):
        self.circuit.append('TICK')

    def two_qubit_gate(self, gateName: str, ctrl_qubit_ids: list[int],
                       targ_qubit_ids: list[int],
                       noise: Optional[tuple[Noise,float]] = None,
                       *, tag=""):
        if len(ctrl_qubit_ids) != len(targ_qubit_ids):
            raise ValueError(f'{len(ctrl_qubit_ids)} control qubits -- '
                             f'{len(targ_qubit_ids)} target qubits mismatch.')
        if len(ctrl_qubit_ids) == 0:
            return
        qubit_ids = []
        for ctrl_id, targ_id in zip(ctrl_qubit_ids, targ_qubit_ids):
            qubit_ids.append(ctrl_id)
            qubit_ids.append(targ_id)
        self.circuit.append(gateName, qubit_ids, tag=tag)
        self.append_noise_channel(noise, qubit_ids, tag=tag)

    def coordinate_span(self, coordinate_index: int) -> tuple[int,int]:
        coordinates = set()
        for pos in self.position_2_qubit_id:
            assert(len(pos)>coordinate_index)
            coordinates.add(pos[coordinate_index])
        return min(coordinates), max(coordinates)
        