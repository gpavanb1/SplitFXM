import pytest
import numpy as np
from splitfxm.cell import Cell  # Replace with the correct import


def test_cell_initialization():
    # Test default initialization
    cell = Cell()
    assert cell._x is None
    assert isinstance(cell._value, np.ndarray)
    assert cell._value.size == 0
    assert not cell.to_delete

    # Test initialization with values
    value = np.array([1.0, 2.0, 3.0])
    cell = Cell(x=0.5, value=value)
    assert cell._x == 0.5
    assert np.array_equal(cell._value, value)
    assert not cell.to_delete


def test_cell_equality():
    # Test equality operator
    cell1 = Cell(x=0.2)
    cell2 = Cell(x=0.2)
    cell3 = Cell(x=0.8)
    assert cell1 == cell2
    assert cell1 != cell3


def test_cell_lt_operator():
    # Test less-than operator
    cell1 = Cell(x=0.3)
    cell2 = Cell(x=0.7)
    assert cell1 < cell2
    assert not (cell2 < cell1)


def test_cell_x():
    # Test x() method
    cell = Cell(x=0.9)
    assert cell.x() == 0.9


def test_cell_values():
    # Test values() method
    value = np.array([1.0, 2.0, 3.0])
    cell = Cell(value=value)
    assert np.array_equal(cell.values(), value)


def test_cell_value():
    # Test value() method
    value = np.array([1.0, 2.0, 3.0])
    cell = Cell(value=value)
    assert cell.value(1) == 2.0


def test_cell_set_value():
    # Test set_value() method
    value = np.array([1.0, 2.0, 3.0])
    cell = Cell(value=value)
    cell.set_value(1, 5.0)
    assert cell.value(1) == 5.0


def test_cell_set_values():
    # Test set_values() method
    new_values = np.array([4.0, 5.0, 6.0])
    cell = Cell()
    cell.set_values(new_values)
    assert np.array_equal(cell.values(), new_values)


def test_cell_update():
    # Test update() method
    value = np.array([1.0, 2.0, 3.0])
    residual = np.array([0.1, 0.2, 0.3])
    cell = Cell(value=value)
    cell.update(0.5, residual)
    assert np.allclose(cell.values(), np.array([1.05, 2.1, 3.15]))
