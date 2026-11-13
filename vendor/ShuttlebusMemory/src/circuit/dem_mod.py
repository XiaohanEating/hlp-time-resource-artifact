from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from enum import Enum

import stim

from gate_types import Pauli
from surface_code import QubitType

class HierarchyType(Enum):
    LEVEL0 = 0
    LEVEL1 = 1
    UNDEFINED = 2

class PatchType(Enum):
    DATA = 0
    X_ANC = 1
    Z_ANC = 2

@dataclass
class ErrorLocation:
    error_type: Pauli
    lv1_time: int
    clock_time: int
    err_index: int

    def __lt__(self, other: ErrorLocation) -> bool:
        if self.error_type.name < other.error_type.name:
            return True
        elif self.error_type.name > other.error_type.name:
            return False
        if self.lv1_time < other.lv1_time:
            return True
        elif self.lv1_time > other.lv1_time:
            return False
        if self.clock_time < other.clock_time:
            return True
        elif self.clock_time > other.clock_time:
            return False
        if self.err_index < other.err_index:
            return True
        else:
            return False
        
    def __repr__(self) -> str:
        info = 'Error Location: {'
        info += f'error type: {self.error_type}, '
        info += f'lv1 time: {self.lv1_time}, '
        info += f'clock time: {self.clock_time}, '
        info += f'error index: {self.err_index}'
        info += '}\n'
        return info

@dataclass
class rich_det:
    det_id: int
    patch_id: Optional[int] = None
    lv0_pos: tuple[Optional[int],Optional[int]] = (None,None)
    clock_time: Optional[int] = None
    hierarchy: HierarchyType = HierarchyType.UNDEFINED

    def __str__(self) -> str:
        info = f'DET-{self.det_id}: '
        info += f'patch_id: {self.patch_id}, lv0 position: {self.lv0_pos}, '
        info += f'clock_time {self.clock_time}, hierarchy: {self.hierarchy}'
        info += '}\n'
        return info
    

@dataclass
class lv1_det:
    det_id: int
    patch_id: int

class DemHandler:

    def __init__(self, dem: stim.DetectorErrorModel):
        self.dem = dem.flattened()
        self.dets: list[rich_det] = []
        pass

    def _get_dets(self):
        for inst in self.dem:
            if inst.type != 'detector':
                continue
            det_targets = inst.targets_copy()
            assert(len(det_targets)==1)
            assert(det_targets[0].is_relative_detector_id())
            tag = inst.tag
            det = rich_det(det_id=det_targets[0].val)
            if len(inst.args_copy()) == 4:
                assert(inst.args_copy()[0]==999)
                det.hierarchy = HierarchyType.LEVEL1
            else:
                assert(len(inst.args_copy())==3)
                det.hierarchy = HierarchyType.LEVEL0
                det.lv0_pos = (inst.args_copy()[0], inst.args_copy()[1])
                det.clock_time = inst.args_copy()[2]
            self.dets.append(det)


class SOTaggedDem:

    def __init__(self, lv1_dem: stim.DetectorErrorModel):
        self.lv1_dem = lv1_dem.flattened()
        self.err_instructions: list[stim.DemInstruction] = []
        self.err_instructions_so_tagged: list[stim.DemInstruction] = []
        self.detector_instructions: list[stim.DemInstruction] = []
        for inst in self.lv1_dem:
            if inst.type == 'error':
                self.err_instructions.append(inst)
            else:
                assert(inst.type=='detector')
                self.detector_instructions.append(inst)
        self.data_patches = []
        self.data_patch_2_ordered_err_locations: dict[int,list[ErrorLocation]] = {}
        self.data_patch_err_type_2_init_err_inst: dict[tuple[int,Pauli],int] = {}
        self.data_patch_err_type_2_end_err_inst: dict[tuple[int,Pauli],int] = {}
        self.data_patch_err_type_2_duration_template: dict[tuple[int,Pauli],list[int]] = {}
        self.err_index_2_duration: dict[int,int] = {}
        self.anc_patch_lv1_time_2_clock_period: dict[tuple[int,int],tuple[int,int]] = {}
        pass

    def _so_tagging_on_data(self):
        for index, inst in enumerate(self.err_instructions):
            key_2_val = tag_parser(inst.tag)
            if key_2_val['patch_type'] != PatchType.DATA:
                continue
            if key_2_val['patch_id'] not in self.data_patches:
                self.data_patches.append(key_2_val['patch_id'])
            err_location = ErrorLocation(error_type=key_2_val['error_type'],
                                         lv1_time=key_2_val['lv1cycle_time'],
                                         clock_time=key_2_val['clock_time'],
                                         err_index=index)
            if key_2_val['patch_id'] not in self.data_patch_2_ordered_err_locations:
                self.data_patch_2_ordered_err_locations[key_2_val['patch_id']] = [err_location]
            else:
                self.data_patch_2_ordered_err_locations[key_2_val['patch_id']].append(err_location)
                pass
        for patch_id in self.data_patch_2_ordered_err_locations:
            self.data_patch_2_ordered_err_locations[patch_id].sort()
            for err_type in [Pauli.X, Pauli.Z]:
                lv1_duration_template = []
                lv1_round_template = []
                locations: list[ErrorLocation] = []
                for err_location in self.data_patch_2_ordered_err_locations[patch_id]:
                    if err_location.error_type != err_type:
                        continue
                    locations.append(err_location)
                self.data_patch_err_type_2_init_err_inst[(patch_id,err_type)] = locations[0].err_index
                self.data_patch_err_type_2_end_err_inst[(patch_id,err_type)] = locations[-1].err_index
                for err_location in locations:
                    if err_location.lv1_time == 0:
                        if err_location.err_index == locations[0].err_index:
                            continue
                        lv1_round_template.append(err_location.clock_time)
                    if err_location.lv1_time == 1:
                        lv1_round_template.append(err_location.clock_time)
                        break
                for i in range(len(lv1_round_template)-1):
                    lv1_duration_template.append(lv1_round_template[i+1]-lv1_round_template[i])
                self.data_patch_err_type_2_duration_template[(patch_id,err_type)] = [_ for _ in lv1_duration_template]
                for i, err_location in enumerate(locations):
                    if i == 0 or i == len(locations)-1:
                        continue
                    self.err_index_2_duration[err_location.err_index] = (locations[i+1].clock_time - err_location.clock_time) // 2
                self.err_index_2_duration[locations[0].err_index] = lv1_duration_template[-1] // 2
                self.err_index_2_duration[locations[-1].err_index] = lv1_duration_template[-1] // 2

    def _so_tagging_on_anc(self, lv0_dem: stim.DetectorErrorModel):
        for inst in lv0_dem:
            if inst.type != 'detector':
                continue
            if len(inst.args_copy()) != 3:
                continue
            key_2_val = tag_parser(inst.tag)
            if key_2_val['patch_type'] not in [PatchType.X_ANC, PatchType.Z_ANC]:
                continue
            anc_id = key_2_val['patch_id']
            anc_patch_lv1_time = (anc_id,key_2_val['lv1_time'])
            clock_time = inst.args_copy()[-1]
            if anc_patch_lv1_time not in self.anc_patch_lv1_time_2_clock_period:
                self.anc_patch_lv1_time_2_clock_period[anc_patch_lv1_time] = (clock_time,clock_time)
            else:
                current_start_end_time = self.anc_patch_lv1_time_2_clock_period[anc_patch_lv1_time]
                updated_start_time = current_start_end_time[0]
                updated_end_time = current_start_end_time[1]
                if current_start_end_time[0] > clock_time:
                    updated_start_time = clock_time
                if current_start_end_time[1] < clock_time:
                    updated_end_time = clock_time
                self.anc_patch_lv1_time_2_clock_period[anc_patch_lv1_time] = (updated_start_time, updated_end_time)
        for index, err_inst in enumerate(self.err_instructions):
            key_2_val = tag_parser(err_inst.tag)
            if key_2_val['patch_type'] not in [PatchType.X_ANC, PatchType.Z_ANC]:
                continue
            anc_patch_lv1_time = (key_2_val['patch_id'],key_2_val['lv1cycle_time'])
            start_end_times = self.anc_patch_lv1_time_2_clock_period[anc_patch_lv1_time]
            duration = int((start_end_times[1] - start_end_times[0]) // 2 + 1)
            self.err_index_2_duration[index] = duration

    def so_tagged_dem(self) -> stim.DetectorErrorModel:
        for index, inst in enumerate(self.err_instructions):
            new_inst = stim.DemInstruction(type=inst.type, 
                                           args=inst.args_copy(),
                                           targets=inst.targets_copy(),
                                           tag=inst.tag + f',time_interval={self.err_index_2_duration[index]}')
            self.err_instructions_so_tagged.append(new_inst)
            pass
        ret_dem = stim.DetectorErrorModel()
        for inst in self.err_instructions_so_tagged:
            ret_dem.append(inst)
        for inst in self.detector_instructions:
            ret_dem.append(inst)
        return ret_dem
    

def tag_parser(tag: str) -> dict:
    key_vals = tag.split(',')
    key_2_val = {}
    for key_val in key_vals:
        pair = key_val.split('=')
        if pair[0] == 'patch_id':
            key_2_val[pair[0]] = int(pair[1])
        elif pair[0] == 'patch_type':
            if pair[1] == 'DATA':
                key_2_val[pair[0]] = PatchType.DATA
            elif pair[1] == 'X_ANC':
                key_2_val[pair[0]] = PatchType.X_ANC
            else:
                assert(pair[1]=='Z_ANC')
                key_2_val[pair[0]] = PatchType.Z_ANC
        elif pair[0] == 'lv1cycle_time' or pair[0] == 'lv1_time':
            key_2_val[pair[0]] = int(pair[1])
        elif pair[0] == 'clock_time':
            key_2_val[pair[0]] = int(pair[1])
        elif pair[0] == 'error_type':
            if pair[1] == 'X':
                key_2_val[pair[0]] = Pauli.X
            else:
                assert(pair[1]=='Z')
                key_2_val[pair[0]] = Pauli.Z
        elif pair[0] == 'time_interval':
            key_2_val[pair[0]] = int(pair[1])
    return key_2_val

# error[patch_id=1,patch_type=X_ANC,error_type=Z,lv1cycle_time=1,clock_time=64](0.0001000000000000000048) D3 D5
# detector[patch_id=3,lv1_posx=1,lv1_posy=-1,pauli=X,boundary=True,boundary_index=0,lv1_time=0,aftercnot=False,cnot=UNDEF,patch_type=DATA](10, -6, 6) D213
