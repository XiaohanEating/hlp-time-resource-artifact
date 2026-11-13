from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import Any, Union, TypeVar, Tuple, Literal, Optional
from warnings import warn

from numpy import ndarray
import numpy as np
import stim

from gate_types import Pauli, TagBuffer

@unique
class QubitType(Enum):
    Data = 'Data'
    X_ancilla = 'X_ancilla'
    Z_ancilla = 'Z_ancilla'
    Unspecified = 'Unspecified'

    @staticmethod
    def from_name(name: str) -> QubitType:
        admissible_types: list[QubitType] = [] 
        for qubittype in QubitType:
            if qubittype.value in name:
                admissible_types.append(qubittype)
        if not admissible_types:
            raise ValueError(f'name {name} does not correspond to any available'
                             'qubit types.')
        if len(admissible_types) >= 2:
            raise ValueError(f'name {name} may correspond to at least two '
                             'available qubit types')
        return admissible_types[0]


class Qubit:
    
    def __init__(self, 
                 position: Optional[tuple[int,...]],
                 qubit_type: QubitType = QubitType.Unspecified) -> None:
        self.position = position
        self.qubit_type = qubit_type
        self.SEIndex_2_neighbor_qubit: dict[int,Qubit] = {}
        self.circuit_id = None
        self.tag_buffer: Optional[TagBuffer] = None
    
    def add_SE_neighbor_qubit(self, SEIndex: int, qubit: Qubit):
        if SEIndex in self.SEIndex_2_neighbor_qubit:
            raise ValueError('Neighboring qubit with SEIndex '
                             f'{SEIndex} is already added.')
        assert SEIndex not in self.SEIndex_2_neighbor_qubit
        self.SEIndex_2_neighbor_qubit[SEIndex] = qubit

    def set_circuit_id(self, circuit_id: int) -> None:
        if self.circuit_id is not None:
            raise RuntimeError(f'attempt to reset the circuit id {self.circuit_id}'
                               f' as {circuit_id}.')
        self.circuit_id = circuit_id
    
    def __repr__(self) -> str:
        ret_str = f'<QubitType: {self.qubit_type}, '
        ret_str += f'position: {self.position}, circuit_id: {self.circuit_id}\n'
        ret_str += f'(SE_index, ancilla position): '
        for se_index in self.SEIndex_2_neighbor_qubit:
            ret_str += f'({se_index},'\
                       f'{self.SEIndex_2_neighbor_qubit[se_index].position}), '
        ret_str += '>\n'
        return ret_str


class RectGrid:
    '''
    2D Rectangular grid as a collection of vertices. Upperleft vertex is set 
    to be located at (0,0) by default. Each vertex should have integer coordinates.
    x: short for x_incr (increment in the x direction)
    y: short for y_incr (increment in the y direction)
    (0,0) --- (x,0) ---> +x direction
      |         |
      |         |
    (0,y) --- (x,y)
      |
      v
     +y direction
    '''
    def __init__(self, x_incr: int, x_points: int,
                 y_incr: int, y_points: int,
                 x_offset: int = 0,
                 y_offset: int = 0, 
                 name: str = QubitType.Unspecified.value):
        if x_incr <= 0:
            raise ValueError(f'{x_incr=} should be positive')
        if y_incr <= 0:
            raise ValueError(f'{y_incr=} should be positive')
        if x_points <= 0:
            raise ValueError(f'{x_points=} should be positive')
        if y_points <= 0:
            raise ValueError(f'{y_points=} should be positive')
        self.x_incr = x_incr
        self.x_points = x_points
        self.y_incr = y_incr
        self.y_points = y_points
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.x_shifted = 0  # register x shifts
        self.y_shifted = 0  # register y shifts
        self.name = name
        pass

    def __eq__(self, value: RectGrid) -> bool:
        if self.generate_grid_points() == value.generate_grid_points():
            return True
        else:
            return False
    
    @staticmethod
    def from_grid_points(grid_points: set[tuple[int,int]]) -> RectGrid:
        # first check if each grid point is an integral point
        x_components: set[int] = set()
        y_components: set[int] = set()
        for grid_point in grid_points:
            if not isinstance(grid_point[0], int) or \
               not isinstance(grid_point[1], int):
                raise ValueError(f'grid point {grid_point} is not intgral')
            x_components.add(grid_point[0])
            y_components.add(grid_point[1])
        # Generate RectGrid
        x_sorted = sorted(x_components)
        y_sorted = sorted(y_components)
        x_offset = x_sorted[0]
        x_incr = x_sorted[1] - x_sorted[0]
        y_offset = y_sorted[0]
        y_incr = y_sorted[1] - y_sorted[0]
        rectgrid = RectGrid(x_incr, len(x_sorted), y_incr, len(y_sorted),
                            x_offset, y_offset)
        # Verify
        if set(rectgrid.generate_grid_points()) == grid_points:
            return rectgrid
        else:
            raise ValueError('input grid_points do not constitute a '
                             'RectGrid.')

    def shift(self, x_shift: int, y_shift: int) -> None:
        if not isinstance(x_shift, int):
            raise ValueError(f'x_shift {x_shift} is not an integer')
        if not isinstance(y_shift, int):
            raise ValueError(f'y_shift {y_shift} is not an integer')
        self.x_offset += x_shift
        self.y_offset += y_shift
        self.x_shifted += x_shift
        self.y_shifted += y_shift
    
    def get_position_from_index(self, xy_index: tuple[int,int]) -> tuple[int,int]:
        assert xy_index[0] >= 0 and xy_index[1] >= 0
        if xy_index[0] > self.x_points-1:
            raise IndexError(f'x_index {xy_index[0]} out of range '
                             f'0 -- {self.x_points-1}.')
        if xy_index[1] > self.y_points-1:
            raise IndexError(f'y_index {xy_index[1]} out of range '
                             f'0 -- {self.y_points-1}.')
        return (self.x_offset + self.x_incr*xy_index[0],
                self.y_offset + self.y_incr*xy_index[1])
    
    def get_index_from_position(self, position: tuple[int,int]) -> tuple[int,int]:
        x_index = (position[0]-self.x_offset) // self.x_incr
        y_index = (position[1]-self.y_offset) // self.y_incr
        if (position[0]-self.x_offset) % self.x_incr != 0 or \
           (position[1]-self.y_offset) % self.y_incr != 0:
            raise ValueError(f'position {position} is not on the rectgrid.')
        assert position[0] == x_index*self.x_incr + self.x_offset
        assert position[1] == y_index*self.y_incr + self.y_offset
        return (x_index, y_index)

    def generate_grid_points(self) -> set[tuple[int,int]]:
        grid_points: set[tuple[int,int]] = set()
        for i in range(self.x_points):
            for j in range(self.y_points):
                grid_points.add(self.get_position_from_index((i,j)))
        return grid_points
    
    def generate_grid_points_sorted(self) -> list[tuple[int,int]]:
        return sorted(self.generate_grid_points())
    
    @property
    def x_min(self) -> int:
        return self.x_offset
    
    @property
    def x_max(self) -> int:
        return self.x_offset + self.x_incr*(self.x_points-1)
    
    @property
    def y_min(self) -> int:
        return self.y_offset
    
    @property
    def y_max(self) -> int:
        return self.y_offset + self.y_incr*(self.y_points-1)


class Lattice:
    '''
    Composition of multiple RectGrids.
    '''
    def __init__(self):
        self.rectgrids: list[RectGrid] = []
        self.name_2_rectgrid: dict[str,RectGrid] = {}
        self.lattice_points: set[tuple[int,int]] = set()
        pass
    
    def add_rectgrid(self, rectgrid: RectGrid) -> None:
        if rectgrid.generate_grid_points().intersection(self.lattice_points):
            raise ValueError(f'rectgrid {rectgrid.name} intersects nontrivially'
                             'with the existing lattice.')
        assert rectgrid.name not in self.name_2_rectgrid
        self.rectgrids.append(rectgrid)
        self.name_2_rectgrid[rectgrid.name] = rectgrid
        self.lattice_points.update(rectgrid.generate_grid_points())

    def shift(self, x_shift: int, y_shift: int):
        # rectgrids in Lattice should have the same off-set
        for rectgrid in self.rectgrids:
            rectgrid.shift(x_shift, y_shift)

    def intersection(self, lattice: Lattice) -> Lattice:
        pass

    def sublattice(self, x_bounds: tuple[int,int],
                         y_bounds: tuple[int,int]) -> Lattice:
        pass

    @staticmethod
    def rotate_clockwise(fixed_point: tuple[int,int],
                         rotated_point: tuple[int,int]) -> tuple[int,int]:
        rel_pos = (rotated_point[0]-fixed_point[0],
                   rotated_point[1]-fixed_point[1])
        return (fixed_point[0]-rel_pos[1],
                fixed_point[0]+rel_pos[0])

    @staticmethod
    def rotate_counter_clockwise(fixed_point: tuple[int,int],
                                 rotated_point: tuple[int,int]) -> tuple[int,int]:
        rel_pos = (rotated_point[0]-fixed_point[0],
                   rotated_point[1]-fixed_point[1])
        return (fixed_point[0]+rel_pos[1],
                fixed_point[1]-rel_pos[0])
    
    @staticmethod
    def reflect_along_vertical_axis(fixed_point: tuple[int,int],
                            reflected_point: tuple[int,int]) -> tuple[int,int]:
        rel_pos = (reflected_point[0]-fixed_point[0],
                   reflected_point[1]-fixed_point[1])
        return (fixed_point[0]-rel_pos[0], fixed_point[1]+rel_pos[1])

    @property
    def x_min(self) -> int:
        return min([rectgrid.x_min for rectgrid in self.rectgrids])
    
    @property
    def x_max(self) -> int:
        return max(rectgrid.x_max for rectgrid in self.rectgrids)
    
    @property
    def y_min(self) -> int:
        return min(rectgrid.y_min for rectgrid in self.rectgrids)
    
    @property
    def y_max(self) -> int:
        return max(rectgrid.y_max for rectgrid in self.rectgrids)
    
    @property
    def x_span(self) -> int:
        return self.x_max - self.x_min
    
    @property
    def y_span(self) -> int:
        return self.y_max - self.y_min
    

class CSS_code_2D(ABC):

    def __init__(self):
        self.qubit_lattice = Lattice()
        self.generate_qubit_lattice()
        self.qubits: list[Qubit] = []
        self.position_2_qubit: dict[tuple[int,int],Qubit] = {}
        self._qubit_generation_from_lattice()
        self.generate_SE_gate_ordering()
        self.logical_X_operator = self.logical_X_support()
        self.logical_Z_operator = self.logical_Z_support()

    def _qubit_generation_from_lattice(self) -> None:
        for rectname in self.qubit_lattice.name_2_rectgrid:
            rectgrid = self.qubit_lattice.name_2_rectgrid[rectname]
            qubit_type = QubitType.from_name(rectname)
            if qubit_type == QubitType.Unspecified:
                warn(f'The rectgrid `{rectgrid.name}` is not assigned any concrete '
                     'qubit type and will be ignored here.')
                continue
            for position in rectgrid.generate_grid_points_sorted():
                qubit = Qubit(position, qubit_type)
                assert qubit not in self.position_2_qubit
                self.qubits.append(qubit)
                self.position_2_qubit[position] = qubit
    
    def shift(self, x_shift: int, y_shift: int):
        self.qubit_lattice.shift(x_shift, y_shift)
        self.position_2_qubit.clear()
        for qubit in self.qubits:
            qubit.position = (qubit.position[0]+x_shift, qubit.position[1]+y_shift)
            self.position_2_qubit[qubit.position] = qubit

    @property
    def x_shifted(self) -> int:
        ret_x_shifted: Optional[int] = None
        for rect_grid in self.qubit_lattice.rectgrids:
            if ret_x_shifted is None:
                ret_x_shifted = rect_grid.x_shifted
            elif ret_x_shifted != rect_grid.x_shifted:
                raise ValueError(f'{rect_grid.name=} has x_shifted ' 
                                 f'{rect_grid.x_shifted} different from another '
                                 f'x_shifted {ret_x_shifted} in the lattice')
        return ret_x_shifted
    
    @property
    def y_shifted(self) -> int:
        ret_y_shifted: Optional[int] = None
        for rect_grid in self.qubit_lattice.rectgrids:
            if ret_y_shifted is None:
                ret_y_shifted = rect_grid.y_shifted
            elif ret_y_shifted != rect_grid.y_shifted:
                raise ValueError(f'{rect_grid.name=} has y_shifted ' 
                                 f'{rect_grid.y_shifted} different from another '
                                 f'y_shifted {ret_y_shifted} in the lattice')
        return ret_y_shifted

    @property
    def x_span(self) -> int:
        return self.qubit_lattice.x_span
    
    @property
    def y_span(self) -> int:
        return self.qubit_lattice.y_span
    
    @property
    def x_min(self) -> int:
        return self.qubit_lattice.x_min
    
    @property
    def x_max(self) -> int:
        return self.qubit_lattice.x_max
    
    @property
    def y_min(self) -> int:
        return self.qubit_lattice.y_min
    
    @property
    def y_max(self) -> int:
        return self.qubit_lattice.y_max

    @abstractmethod
    def generate_qubit_lattice(self) -> None:
        pass

    @abstractmethod
    def generate_SE_gate_ordering(self) -> None:
        pass

    @abstractmethod
    def logical_Z_support(self) -> list[Qubit]:
        pass

    @abstractmethod
    def logical_X_support(self) -> list[Qubit]:
        pass

    @abstractmethod
    def dock(self):
        raise NotImplementedError(f'dock method not implemented for {self}')
    
    @property
    @abstractmethod
    def center(self) -> tuple[int,int]:
        raise NotImplementedError(f'center method not implemented for {self}')
    
    def logical_pauli_line_through_qubit(
            self, position:tuple[int,int],
            logical_basis: Pauli
        ) -> list[tuple[int,int]]:
        raise NotImplementedError(f'logical pauli line not supported for {self}')
    
    @property
    @abstractmethod
    def ports(self) -> list[int]:
        pass

    def is_boundary_stabilizer(self, qubit: Qubit) -> tuple[bool,Literal['0','1','UNDEF']]:
        raise NotImplementedError(f'boundary checking not implemented for {self}')


class Rotated_surface_code(CSS_code_2D):

    def __init__(self, distance: int):
        '''
        --
        '''
        if distance <= 2:
            raise ValueError(f'{distance=} should be at least 3.')
        self.distance = distance
        super().__init__()
        pass

    def generate_qubit_lattice(self):
        data_incr = 2
        anc_incr = 4
        data_grid = RectGrid(data_incr, self.distance, 
                             data_incr, self.distance,
                             name = 'Data_Block')
        self.qubit_lattice.add_rectgrid(data_grid)
        bulk_anc_pos_max = (self.distance-1)*data_incr - 1

        def max_num_points_from_length(length: int, incr: int) -> int:
            assert length >= 0 and incr >= 0
            return 1 + (length // incr)

        # X, Z ancilla qubits in the bulk
        bulk_ancilla_grid_indices = ['00', '01', '10', '11']
        grid_index_2_offset = {
            '00': (1,1),
            '01': (1,3),
            '10': (3,1),
            '11': (3,3)
        }
        if self.distance % 2 == 1:
            grid_index_2_qubit_type = {
                '00': QubitType.X_ancilla,
                '01': QubitType.Z_ancilla,
                '10': QubitType.Z_ancilla,
                '11': QubitType.X_ancilla
            }
        else:
            grid_index_2_qubit_type = {
                '00': QubitType.Z_ancilla,
                '01': QubitType.X_ancilla,
                '10': QubitType.X_ancilla,
                '11': QubitType.Z_ancilla
            }
        for grid_index in bulk_ancilla_grid_indices:
            x_length_bound = bulk_anc_pos_max - grid_index_2_offset[grid_index][0]
            y_length_bound = bulk_anc_pos_max - grid_index_2_offset[grid_index][1]
            bulk_anc_rectgrid = RectGrid(
                                anc_incr,
                                max_num_points_from_length(x_length_bound,
                                                           anc_incr),
                                anc_incr,
                                max_num_points_from_length(y_length_bound,
                                                           anc_incr),
                                x_offset=grid_index_2_offset[grid_index][0],
                                y_offset=grid_index_2_offset[grid_index][1],
                                name = grid_index_2_qubit_type[grid_index].value + \
                                       '_Block_' + grid_index)
            self.qubit_lattice.add_rectgrid(bulk_anc_rectgrid)
        # X, Z ancilla qubits on the boundary

        if self.distance % 2 == 1:
            num_z_anc = max_num_points_from_length(bulk_anc_pos_max-1,anc_incr)
            left_y_offset = 1
            right_y_offset = 3
            num_x_anc = max_num_points_from_length(bulk_anc_pos_max-1,anc_incr)
            up_x_offset = 3
            down_x_offset = 1
        else:
            num_z_anc = max_num_points_from_length(bulk_anc_pos_max-3,anc_incr)
            left_y_offset = 3
            right_y_offset = 3
            num_x_anc = max_num_points_from_length(bulk_anc_pos_max-1,anc_incr)
            up_x_offset = 1
            down_x_offset = 1
        left_bd_Z_block = RectGrid(anc_incr, 1, anc_incr, num_z_anc,
                                   x_offset = -1, y_offset = left_y_offset,
                                   name = QubitType.Z_ancilla.value + '_left_bd_Block')
        right_bd_Z_block = RectGrid(anc_incr, 1, anc_incr, num_z_anc,
                                    x_offset = self.distance*data_incr - 1,
                                    y_offset = right_y_offset,
                                    name = QubitType.Z_ancilla.value + '_right_bd_Block')
        up_bd_X_block = RectGrid(anc_incr, num_x_anc, anc_incr, 1,
                                 x_offset = up_x_offset, y_offset = -1,
                                 name = QubitType.X_ancilla.value + '_up_bd_Block')
        down_bd_X_block = RectGrid(anc_incr, num_x_anc, anc_incr, 1,
                                   x_offset = down_x_offset, 
                                   y_offset = self.distance*data_incr - 1,
                                   name = QubitType.X_ancilla.value + '_down_bd_Block')
        self.qubit_lattice.add_rectgrid(left_bd_Z_block)
        self.qubit_lattice.add_rectgrid(right_bd_Z_block)
        self.qubit_lattice.add_rectgrid(up_bd_X_block)
        self.qubit_lattice.add_rectgrid(down_bd_X_block)
    
    def generate_SE_gate_ordering(self):
        position_shift_pm1 = [(x_shift, y_shift) for y_shift in [-1,1] 
                                                 for x_shift in [-1,1]] 
        position_shift_2_XSE_order = {(-1,-1): 1, (1,-1): 2, 
                                      (-1,1): 3, (1,1): 4}
        position_shift_2_ZSE_order = {(-1,-1): 1, (-1,1): 2,
                                      (1,-1): 3, (1,1): 4}
        for qubit in self.qubits:
            if qubit.qubit_type == QubitType.Data or \
               qubit.qubit_type == QubitType.Unspecified:
                continue
            if qubit.qubit_type == QubitType.X_ancilla:
                shift_2_SE_order = position_shift_2_XSE_order
            elif qubit.qubit_type == QubitType.Z_ancilla:
                shift_2_SE_order = position_shift_2_ZSE_order
            anc_pos = qubit.position
            for x_shift, y_shift in position_shift_pm1:
                data_pos = (anc_pos[0]+x_shift, anc_pos[1]+y_shift)
                if data_pos in self.position_2_qubit:
                    data_qubit = self.position_2_qubit[data_pos]
                    SE_order = shift_2_SE_order[(x_shift, y_shift)]
                    qubit.add_SE_neighbor_qubit(SE_order, data_qubit)
                    data_qubit.add_SE_neighbor_qubit(SE_order, qubit)
    
    def logical_X_support(self) -> list[Qubit]:
        ret_list = []
        for qubit in self.qubits:
            if qubit.position[0] == 0:
                ret_list.append(qubit)
        return ret_list
    
    def logical_Z_support(self) -> list[Qubit]:
        ret_list = []
        for qubit in self.qubits:
            if qubit.position[1] == 0:
                ret_list.append(qubit)
        return ret_list
    
    def logical_pauli_line_through_qubit(
            self, position:tuple[int,int],
            logical_basis: Pauli
        ) -> list[tuple[int,int]]:
        if logical_basis not in [Pauli.X, Pauli.Z]:
            raise TypeError(f'{logical_basis=} should be either X or Z.')
        if position not in self.position_2_qubit:
            raise ValueError(f'{position=} is not a valid qubit position')
        if self.position_2_qubit[position].qubit_type != QubitType.Data:
            raise ValueError(f'{position=} points to a qubit type '
                             f'{self.position_2_qubit[position].qubit_type} '
                             f'which is not the Data type.')
        line_of_qubit_pos = []
        for qubit in self.qubits:
            if logical_basis == Pauli.X:
                if qubit.position[0] == position[0]:
                    line_of_qubit_pos.append(qubit.position)
            else:
                assert logical_basis == Pauli.Z
                if qubit.position[1] == position[1]:
                    line_of_qubit_pos.append(qubit.position)
        return line_of_qubit_pos

    def dock(self, code2: Rotated_surface_code) -> dict[tuple[int,int],
                                                        tuple[int,int]]:
        if not isinstance(code2,Rotated_surface_code):
            raise TypeError(f'{code2} is not a Rotated Surface Code.')
        if code2.distance != self.distance:
            raise ValueError(f'{code2.distance=} mismatches with '
                             f'{self.distance=}.')
        ret_dict = {}
        for qubit in self.qubits:
            unshifted_pos = (qubit.position[0]-self.x_shifted,
                             qubit.position[1]-self.y_shifted)
            pos_on_code_2 = (unshifted_pos[0]+code2.x_shifted,
                             unshifted_pos[1]+code2.y_shifted)
            ret_dict[qubit.position] = pos_on_code_2
        return ret_dict
    
    def reverse_dock(self, code_2: ShuttleBus_code) -> dict[tuple[int,int],
                                                            tuple[int,int]]:
        if not isinstance(code_2, ShuttleBus_code):
            raise TypeError(f'{code_2} is not a Shuttle bus Code.')
        if code_2.longBasis != Pauli.Z:
            raise NotImplementedError(f'{code_2} has long basis {code_2.longBasis}, '
                                      f'which is currently not supported for reverse docking.')
        if code_2.sc_distance > self.distance:
            raise ValueError(f'The X distance of {code_2=} ({code_2.sc_distance}) should be no larger'
                             f' than that of the surface code distance ({self.distance}).')
        if code_2.sc_distance*code_2.length < self.distance:
            raise ValueError(f'The Z distance of {code_2=} ({code_2.sc_distance*code_2.length})'
                             f' should be no smaller than that of the surface code ({self.distance}).')
        ret_dict = {}
        lb_unshifted_pos = (0, 2*(self.distance-1))
        lb_code_2_unshifted_pos = (0, 2*(code_2.sc_distance-1))
        x_rel_range = (self.x_min-self.x_shifted, self.x_max-self.x_shifted)
        y_rel_range = (code_2.y_min-code_2.y_shifted-lb_code_2_unshifted_pos[1],
                       code_2.y_max-code_2.y_shifted-lb_code_2_unshifted_pos[1])
        assert x_rel_range[0] == -1
        assert x_rel_range[1] == 2*self.distance-1
        assert y_rel_range[0] == -(2*code_2.sc_distance-1)
        assert y_rel_range[1] == 1
        for qubit in self.qubits:
            unshifted_pos = (qubit.position[0]-self.x_shifted,
                             qubit.position[1]-self.y_shifted)
            rel_lb_pos = (unshifted_pos[0]-lb_unshifted_pos[0],
                          unshifted_pos[1]-lb_unshifted_pos[1])
            if rel_lb_pos[0] < x_rel_range[0] or rel_lb_pos[0] > x_rel_range[1]:
                continue
            if rel_lb_pos[1] < y_rel_range[0] or rel_lb_pos[1] > y_rel_range[1]:
                continue
            pos_on_code_2 = (lb_code_2_unshifted_pos[0]+rel_lb_pos[0]+code_2.x_shifted,
                             lb_code_2_unshifted_pos[1]+rel_lb_pos[1]+code_2.y_shifted)
            if pos_on_code_2 not in code_2.position_2_qubit:
                continue
            ret_dict[qubit.position] = pos_on_code_2
        return ret_dict
    
    @property
    def center(self) -> tuple[int,int]:
        x_center = round((self.qubit_lattice.x_min+self.qubit_lattice.x_max)/2)
        y_center = round((self.qubit_lattice.y_min+self.qubit_lattice.y_max)/2)
        return (x_center, y_center)
    
    @property
    def ports(self) -> list[int]:
        return [0]
    
    def is_boundary_stabilizer(self, qubit) -> tuple[bool,Literal['0','1','UNDEF']]:
        if qubit.position not in self.position_2_qubit:
            raise ValueError(f'{qubit.position=} which is not contained in the code.')
        if qubit.qubit_type == QubitType.X_ancilla:
            if qubit.position[0]-self.x_min == 2:
                return [True,'0']
            elif self.x_max-qubit.position[0] == 2:
                return [True,'1']
            else:
                return [False,'UNDEF']
        elif qubit.qubit_type == QubitType.Z_ancilla:
            if qubit.position[1]-self.y_min == 2:
                return [True,'0']
            elif self.y_max-qubit.position[1] == 2:
                return [True,'1']
            else:
                return [False,'UNDEF']
        else:
            return [False,'UNDEF']        


class ShuttleBus_code(CSS_code_2D):
    '''
    Single-deck bus
    '''
    def __init__(self, sc_distance: int, length: int,
                 longBasis: Pauli):
        if sc_distance <= 2:
            raise ValueError(f'{sc_distance=} should be at least 3.')
        if length < 1:
            raise ValueError(f'{length=} should be at least 1.')
        if longBasis.basis not in ['X','Z']:
            raise NotImplementedError(f'{longBasis=} should be either '
                                      '\'X\' or \'Z\'.') 
        self.sc_distance = sc_distance
        self.length = length
        self.longBasis = longBasis
        super().__init__()
        pass

    def generate_qubit_lattice(self):
        data_incr = 2
        anc_incr = 4
        data_grid = RectGrid(data_incr, self.sc_distance*self.length, 
                             data_incr, self.sc_distance,
                             name = 'Data_Block')
        self.qubit_lattice.add_rectgrid(data_grid)
        bulk_anc_x_max = ((self.sc_distance*self.length)-1)*data_incr - 1
        bulk_anc_y_max = (self.sc_distance-1)*data_incr - 1

        def max_num_points_from_length(length: int, incr: int) -> int:
            assert length >= 0 and incr >= 0
            return 1 + (length // incr)

        # X, Z ancilla qubits in the bulk
        bulk_ancilla_grid_indices = ['00', '01', '10', '11']
        grid_index_2_offset = {
            '00': (1,1),
            '01': (1,3),
            '10': (3,1),
            '11': (3,3)
        }
        if self.sc_distance % 2 == 1 and self.longBasis == Pauli.Z:
            grid_index_2_qubit_type = {
                '00': QubitType.X_ancilla,
                '01': QubitType.Z_ancilla,
                '10': QubitType.Z_ancilla,
                '11': QubitType.X_ancilla
            }
        else:
            grid_index_2_qubit_type = {
                '00': QubitType.Z_ancilla,
                '01': QubitType.X_ancilla,
                '10': QubitType.X_ancilla,
                '11': QubitType.Z_ancilla
            }
        for grid_index in bulk_ancilla_grid_indices:
            x_length_bound = bulk_anc_x_max - grid_index_2_offset[grid_index][0]
            y_length_bound = bulk_anc_y_max - grid_index_2_offset[grid_index][1]
            bulk_anc_rectgrid = RectGrid(
                                anc_incr,
                                max_num_points_from_length(x_length_bound,
                                                           anc_incr),
                                anc_incr,
                                max_num_points_from_length(y_length_bound,
                                                           anc_incr),
                                x_offset=grid_index_2_offset[grid_index][0],
                                y_offset=grid_index_2_offset[grid_index][1],
                                name = grid_index_2_qubit_type[grid_index].value + \
                                       '_Block_' + grid_index)
            self.qubit_lattice.add_rectgrid(bulk_anc_rectgrid)
        # X, Z ancilla qubits on the boundary

        if self.sc_distance % 2 == 1:
            left_y_offset = 1
            if self.length % 2 == 1:
                right_y_offset = 3
            else:
                right_y_offset = 1
            up_x_offset = 3
            down_x_offset = 1
        elif self.longBasis == Pauli.Z:
            left_y_offset = 3
            right_y_offset = 3
            up_x_offset = 1
            down_x_offset = 1
        else:
            assert self.longBasis == Pauli.X and self.sc_distance % 2 == 0
            left_y_offset = 1
            right_y_offset = 1
            up_x_offset = 3
            down_x_offset = 3
        num_l_anc = max_num_points_from_length(bulk_anc_y_max-left_y_offset,
                                               anc_incr)
        num_r_anc = max_num_points_from_length(bulk_anc_y_max-right_y_offset,
                                               anc_incr)
        num_u_anc = max_num_points_from_length(bulk_anc_x_max-up_x_offset,
                                               anc_incr)
        num_d_anc = max_num_points_from_length(bulk_anc_x_max-down_x_offset,
                                               anc_incr)
        if self.longBasis == Pauli.Z:
                lr_anc_type = QubitType.Z_ancilla
                ud_anc_type = QubitType.X_ancilla
        else:
                lr_anc_type = QubitType.X_ancilla
                ud_anc_type = QubitType.Z_ancilla
        left_bd_block = RectGrid(anc_incr, 1, anc_incr, num_l_anc,
                                 x_offset = -1, y_offset = left_y_offset,
                                 name = lr_anc_type.value + '_left_bd_Block')
        right_bd_block = RectGrid(anc_incr, 1, anc_incr, num_r_anc,
                                    x_offset = self.sc_distance*self.length*data_incr - 1,
                                    y_offset = right_y_offset,
                                    name = lr_anc_type.value + '_right_bd_Block')
        up_bd_block = RectGrid(anc_incr, num_u_anc, anc_incr, 1,
                                 x_offset = up_x_offset, y_offset = -1,
                                 name = ud_anc_type.value + '_up_bd_Block')
        down_bd_block = RectGrid(anc_incr, num_d_anc, anc_incr, 1,
                                   x_offset = down_x_offset, 
                                   y_offset = self.sc_distance*data_incr - 1,
                                   name = ud_anc_type.value + '_down_bd_Block')
        self.qubit_lattice.add_rectgrid(left_bd_block)
        self.qubit_lattice.add_rectgrid(right_bd_block)
        self.qubit_lattice.add_rectgrid(up_bd_block)
        self.qubit_lattice.add_rectgrid(down_bd_block)
    
    def generate_SE_gate_ordering(self):
        position_shift_pm1 = [(x_shift, y_shift) for y_shift in [-1,1] 
                                                 for x_shift in [-1,1]]
        if self.longBasis == Pauli.Z: 
            position_shift_2_XSE_order = {(-1,-1): 1, (1,-1): 2, 
                                        (-1,1): 3, (1,1): 4}
            position_shift_2_ZSE_order = {(-1,-1): 1, (-1,1): 2,
                                        (1,-1): 3, (1,1): 4}
        else:
            position_shift_2_XSE_order = {(-1,-1): 1, (-1,1): 2,
                                          (1,-1): 3, (1,1): 4}
            position_shift_2_ZSE_order = {(-1,-1): 1, (1,-1): 2,
                                          (-1,1): 3, (1,1): 4}
        for qubit in self.qubits:
            if qubit.qubit_type == QubitType.Data or \
               qubit.qubit_type == QubitType.Unspecified:
                continue
            if qubit.qubit_type == QubitType.X_ancilla:
                shift_2_SE_order = position_shift_2_XSE_order
            elif qubit.qubit_type == QubitType.Z_ancilla:
                shift_2_SE_order = position_shift_2_ZSE_order
            anc_pos = qubit.position
            for x_shift, y_shift in position_shift_pm1:
                data_pos = (anc_pos[0]+x_shift, anc_pos[1]+y_shift)
                if data_pos in self.position_2_qubit:
                    data_qubit = self.position_2_qubit[data_pos]
                    SE_order = shift_2_SE_order[(x_shift, y_shift)]
                    qubit.add_SE_neighbor_qubit(SE_order, data_qubit)
                    data_qubit.add_SE_neighbor_qubit(SE_order, qubit)
    
    def logical_X_support(self):
        ret_list = []
        if self.longBasis == Pauli.Z:
            horizontal_or_vertical = 0
        else:
            horizontal_or_vertical = 1
        for qubit in self.qubits:
            if qubit.position[horizontal_or_vertical] == 0:
                ret_list.append(qubit)
        return ret_list
    
    def logical_Z_support(self):
        ret_list = []
        if self.longBasis == Pauli.X:
            horizontal_or_vertical = 0
        else:
            horizontal_or_vertical = 1
        for qubit in self.qubits:
            if qubit.position[horizontal_or_vertical] == 0:
                ret_list.append(qubit)
        return ret_list
    
    def logical_pauli_line_through_qubit(
            self, position:tuple[int,int],
            logical_basis: Pauli
        ) -> list[tuple[int,int]]:
        if logical_basis not in [Pauli.X, Pauli.Z]:
            raise TypeError(f'{logical_basis=} should be either X or Z.')
        if position not in self.position_2_qubit:
            raise ValueError(f'{position=} is not a valid qubit position')
        if self.position_2_qubit[position].qubit_type != QubitType.Data:
            raise ValueError(f'{position=} points to a qubit type '
                             f'{self.position_2_qubit[position].qubit_type} '
                             f'which is not the Data type.')
        if self.longBasis == Pauli.X:
            if logical_basis == Pauli.X:
                horizontal_or_vertical = 1
            else:
                assert logical_basis == Pauli.Z
                horizontal_or_vertical = 0
        else:
            if logical_basis == Pauli.X:
                horizontal_or_vertical = 0
            else:
                assert logical_basis == Pauli.Z
                horizontal_or_vertical = 1
        line_of_qubit_pos = []
        for qubit in self.qubits:
            if qubit.position[horizontal_or_vertical] \
                == position[horizontal_or_vertical]:
                line_of_qubit_pos.append(qubit.position)
        return line_of_qubit_pos
    
    def dock(self, code2: Rotated_surface_code, port: int) -> dict[tuple[int,int],
                                                                   tuple[int,int]]:
        if not isinstance(code2, Rotated_surface_code):
            raise TypeError(f'{code2} is not a Rotated Surface Code.')
        if self.sc_distance != code2.distance:
            raise ValueError(f'{code2.distance=} mismatches with '
                             f'{self.sc_distance=}.')
        ret_dict = {}
        for qubit in code2.qubits:
            unshifted_pos = (qubit.position[0]-code2.x_shifted,
                             qubit.position[1]-code2.y_shifted)
            unshifted_center = (code2.center[0]-code2.x_shifted,
                                    code2.center[1]-code2.y_shifted)
            if self.longBasis == Pauli.X:
                # rotation
                unshifted_pos = Lattice.rotate_counter_clockwise(unshifted_center,
                                                                 unshifted_pos)
            if self.sc_distance % 2 == 1 and port % 2 == 1:
                # reflection
                unshifted_pos = Lattice.reflect_along_vertical_axis(unshifted_center,
                                                                    unshifted_pos)
            pos_on_bus = (unshifted_pos[0]+port*code2.x_span+self.x_shifted,
                          unshifted_pos[1]+self.y_shifted)
            ret_dict[pos_on_bus] = qubit.position
        return ret_dict
    
    def _short_basis_stabs_to_the_left(self, 
                                       vertical_line_pos: int,
                                       ) -> list[tuple[int,int]]:
        short_basis_anc_type = QubitType.X_ancilla if self.longBasis==Pauli.Z \
                               else QubitType.Z_ancilla
        stab_pos_to_left = []
        for qubit in self.qubits:
            if qubit.qubit_type == short_basis_anc_type:
                if qubit.position[0] < vertical_line_pos:
                    stab_pos_to_left.append(qubit.position)
        return stab_pos_to_left
    
    @property
    def center(self) -> tuple[int,int]:
        return super().center

    @property
    def ports(self) -> list[int]:
        return list(range(self.length))
    
    def is_boundary_stabilizer(self, qubit) -> tuple[bool,Literal['0','1','UNDEF']]:
        if qubit.position not in self.position_2_qubit:
            raise ValueError(f'{qubit.position=} which is not contained in the code.')
        if self.longBasis == Pauli.Z:
            if qubit.qubit_type == QubitType.X_ancilla:
                if qubit.position[0]-self.x_min == 2:
                    return [True,'0']
                elif self.x_max-qubit.position[0] == 2:
                    return [True,'1']
                else:
                    return [False,'UNDEF']
            elif qubit.qubit_type == QubitType.Z_ancilla:
                if qubit.position[1]-self.y_min == 2:
                    return [True,'0']
                elif self.y_max-qubit.position[1] == 2:
                    return [True,'1']
                else:
                    return [False,'UNDEF']
            else:
                return [False,'UNDEF']      
        else:
            assert self.longBasis == Pauli.X
            if qubit.qubit_type == QubitType.Z_ancilla:
                if qubit.position[0]-self.x_min == 2:
                    return [True,'0']
                elif self.x_max-qubit.position[0] == 2:
                    return [True,'1']
                else:
                    return [False,'UNDEF']
            elif qubit.qubit_type == QubitType.X_ancilla:
                if qubit.position[1]-self.y_min == 2:
                    return [True,'0']
                elif self.y_max-qubit.position[1] == 2:
                    return [True,'1']
                else:
                    return [False,'UNDEF']
            else:
                return [False,'UNDEF']  
        

class Duble_Decker_code(CSS_code_2D):

    def __init__(self):
        super().__init__()
    
    

