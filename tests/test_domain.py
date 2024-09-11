import pytest
import numpy as np
from numpy import zeros

from splitfxm.boundary import Boundary
from splitfxm.cell import Cell
from splitfxm.constants import btype
from splitfxm.domain import Domain
from splitfxm.error import SFXM


@pytest.fixture
def example_cells():
    # Simple cells with dummy values
    return [Cell(i, zeros(2)) for i in range(5)]


@pytest.fixture
def example_left_boundaries():
    # 2 left boundaries
    return [Boundary(-i - 1, btype.LEFT, zeros(2)) for i in range(2)]


@pytest.fixture
def example_right_boundaries():
    # 2 right boundaries
    return [Boundary(5 + i, btype.RIGHT, zeros(2)) for i in range(2)]


@pytest.fixture
def components():
    return ['component_1', 'component_2']


@pytest.fixture
def domain(example_cells, example_left_boundaries, example_right_boundaries, components):
    return Domain(example_cells, example_left_boundaries, example_right_boundaries, components)


def test_domain_init(domain, example_cells, example_left_boundaries, example_right_boundaries, components):
    assert len(domain._domain) == len(example_cells) + \
        len(example_left_boundaries) + len(example_right_boundaries)
    assert domain._components == components
    assert domain._nb[btype.LEFT] == len(example_left_boundaries)
    assert domain._nb[btype.RIGHT] == len(example_right_boundaries)
    assert domain._nx == len(example_cells)


def test_domain_from_size():
    nx, nb_left, nb_right = 5, 2, 2
    components = ['component_1', 'component_2']
    domain = Domain.from_size(nx, nb_left, nb_right, components)
    assert len(domain._domain) == nx + nb_left + nb_right
    assert len(domain._components) == 2
    assert domain._nb[btype.LEFT] == nb_left
    assert domain._nb[btype.RIGHT] == nb_right


def test_ilo(domain):
    assert domain.ilo() == 2  # The number of left boundaries


def test_ihi(domain):
    assert domain.ihi() == 6  # Left boundaries + number of cells - 1


def test_nb(domain):
    assert domain.nb(btype.LEFT) == 2
    assert domain.nb(btype.RIGHT) == 2


def test_cells(domain):
    all_cells = domain.cells()
    assert len(all_cells) == 9  # Total of boundaries and interior cells


def test_cells_interior(domain):
    interior_cells = domain.cells(interior=True)
    assert len(interior_cells) == 5  # Only interior cells


def test_boundaries(domain):
    left, right = domain.boundaries()
    assert len(left) == 2
    assert len(right) == 2


def test_interior(domain):
    interior_cells = domain.interior()
    assert len(interior_cells) == 5  # Same as the number of interior cells


def test_set_interior(domain):
    new_cells = [Cell(i + 10, zeros(2))
                 for i in range(6)]  # New set of 6 interior cells
    domain.set_interior(new_cells)
    assert len(domain.interior()) == 6  # Updated number of interior cells


def test_num_components(domain):
    assert domain.num_components() == 2  # There are two components in the fixture


def test_component_index(domain):
    assert domain.component_index('component_1') == 0
    assert domain.component_index('component_2') == 1


def test_component_name(domain):
    assert domain.component_name(0) == 'component_1'
    assert domain.component_name(1) == 'component_2'


def test_positions(domain):
    positions = domain.positions()
    expected_positions = [-1, -2, 0, 1, 2, 3, 4, 5, 6]
    assert np.allclose(positions, expected_positions)


def test_values(domain):
    values = domain.values()
    expected_values = [np.array([0., 0.]) * 9]
    assert np.allclose(values, expected_values)


def test_listify_interior(domain):
    # Test with no split
    flat_values = domain.listify_interior(split=False, split_loc=None)
    assert len(flat_values) == 10  # 5 cells * 2 values each

    # Test with split
    split_values = domain.listify_interior(split=True, split_loc=1)
    assert len(split_values) == 10  # Each cell is split into 2 sets


def test_listify_interior_split_error(domain):
    with pytest.raises(SFXM):
        # Should raise an error without split_loc
        domain.listify_interior(split=True, split_loc=None)


def test_update(domain):
    # Initial interior cell values before the update
    initial_values = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0]),
                      np.array([7.0, 8.0]), np.array([9.0, 10.0])]

    # Set these values for the interior cells
    for i, cell in enumerate(domain.interior()):
        cell.set_values(initial_values[i])

    # Define a simple residual block with finite values
    residual_block = [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6]),
                      np.array([0.7, 0.8]), np.array([0.9, 1.0])]

    # Time step
    dt = 0.5

    # Expected values after the update: initial_value + dt * residual
    expected_values = [initial_values[i] + dt * residual_block[i]
                       for i in range(len(initial_values))]

    # Apply the update
    domain.update(dt, residual_block)

    # Check that the values of the interior cells have been updated correctly
    for i, cell in enumerate(domain.interior()):
        assert np.allclose(cell.values(
        ), expected_values[i]), f"Expected {expected_values[i]}, but got {cell.values()}"
