import numpy as np
import pytest
from splitfxm.boundary import Boundary
from splitfxm.constants import btype
from splitfxm.error import SFXM


# Test initialization of Boundary with correct arguments
def test_boundary_initialization():
    b = Boundary(x=0.0, _btype=btype.LEFT)
    assert b.x() == 0.0
    assert b.values().size == 0

    # Test with value array
    value = np.array([1.0, 2.0])
    b = Boundary(x=1.0, _btype=btype.RIGHT, value=value)
    assert b.values().size == 2
    assert b.values()[0] == 1.0


# Test raising SFXM for inappropriate boundary type
def test_boundary_initialization_invalid_type():
    with pytest.raises(SFXM):
        Boundary(x=-0.1, _btype=btype.RIGHT)

    with pytest.raises(SFXM):
        Boundary(x=1.1, _btype=btype.LEFT)


# Test the x() method
def test_boundary_x():
    b = Boundary(x=0.5, _btype=btype.LEFT)
    assert b.x() == 0.5


# Test the values() method
def test_boundary_values():
    value = np.array([1.0, 2.0])
    b = Boundary(x=0.5, _btype=btype.LEFT, value=value)
    assert np.array_equal(b.values(), value)


# Test the value() method for retrieving specific index
def test_boundary_value():
    value = np.array([1.0, 2.0, 3.0])
    b = Boundary(x=0.5, _btype=btype.LEFT, value=value)
    assert b.value(1) == 2.0
    assert b.value(2) == 3.0


# Test set_x() method and raise errors for invalid x
def test_boundary_set_x():
    b = Boundary(x=0.0, _btype=btype.LEFT)

    # Valid set
    b.set_x(-0.1)
    assert b.x() == -0.1

    # Test raising SFXM for interior x value
    with pytest.raises(SFXM):
        b.set_x(0.5)

    # Test raising SFXM for inappropriate boundary type
    with pytest.raises(SFXM):
        b.set_x(1.1)


# Test set_value() method
def test_boundary_set_value():
    value = np.array([1.0, 2.0, 3.0])
    b = Boundary(x=0.5, _btype=btype.LEFT, value=value)

    # Set new value for index 1
    b.set_value(1, 4.0)
    assert b.value(1) == 4.0

    # Set new value for index 2
    b.set_value(2, 5.0)
    assert b.value(2) == 5.0


# Edge cases: Empty array
def test_boundary_empty_value():
    b = Boundary(x=0.0, _btype=btype.LEFT)
    assert b.values().size == 0
