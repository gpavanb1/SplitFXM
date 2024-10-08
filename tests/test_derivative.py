import pytest
import numpy as np

from splitfxm.cell import Cell
from splitfxm.derivatives import Dx, dx, D2x, d2x
from splitfxm.schemes import FDSchemes
from splitfxm.error import SFXM


def array(*args, **kwargs):
    # Overriding np.array dtype locally
    """Wrapper for np.array to default to float64."""
    kwargs.setdefault('dtype', np.float64)  # Set default dtype to float64
    return np.array(*args, **kwargs)


def test_Dx_central():
    # Test first derivative with CENTRAL scheme
    cell_sub = [Cell(0, array([1.])), Cell(
        1, array([2.])), Cell(2, array([3.]))]
    result = Dx(lambda v: v, cell_sub, FDSchemes.CENTRAL)
    assert np.allclose(result, array([1.]))


def test_Dx_right_bias():
    # Test first derivative with RIGHT_BIAS scheme
    cell_sub = [Cell(0, array([1.])), Cell(1, array([2.])),
                Cell(2, array([3.])), Cell(3, array([4.]))]
    result = Dx(lambda v: v, cell_sub,
                FDSchemes.RIGHT_BIAS)
    assert np.allclose(result, array([1.]))


def test_D2x_central():
    # Test second derivative with CENTRAL scheme
    cell_sub = [Cell(0, array([1.])), Cell(
        1, array([2.])), Cell(2, array([3.]))]
    result = D2x(lambda v: v, cell_sub, FDSchemes.CENTRAL)
    assert result == 0.0


def test_D2x_right_bias():
    # Test second derivative with RIGHT_BIAS scheme
    cell_sub = [Cell(0, array([1])), Cell(1, array([2])),
                Cell(2, array([3])), Cell(3, array([4]))]
    result = D2x(lambda v: v, cell_sub,
                 FDSchemes.RIGHT_BIAS)
    assert result == 0.0


def test_dx_first_central():
    # Test dx for precomputed values with CENTRAL scheme
    cell_sub = [Cell(0, array([1])), Cell(
        1, array([2])), Cell(2, array([3]))]
    values = array([1, 2, 3])
    result = dx(values, cell_sub, FDSchemes.CENTRAL)
    assert result == 1.


def test_dx_first_right_bias():
    # Test dx for precomputed values with RIGHT_BIAS scheme
    cell_sub = [Cell(0, array([1])), Cell(1, array([2])),
                Cell(2, array([3])), Cell(3, array([4]))]
    values = array([1, 2, 3, 4])
    result = dx(values, cell_sub, FDSchemes.RIGHT_BIAS)
    assert result == 1.


def test_d2x_second_central():
    # Test d2x for precomputed values with CENTRAL scheme
    cell_sub = [Cell(0, array([1])), Cell(
        1, array([2])), Cell(2, array([3]))]
    values = array([1, 2, 3])
    result = d2x(values, cell_sub, FDSchemes.CENTRAL)
    assert result == 0.


def test_d2x_second_right_bias():
    # Test d2x for precomputed values with RIGHT_BIAS scheme
    cell_sub = [Cell(0, array([1])), Cell(1, array([2])),
                Cell(2, array([3])), Cell(3, array([4]))]
    values = array([1, 2, 3, 4])
    result = d2x(values, cell_sub, FDSchemes.RIGHT_BIAS)
    assert result == 0.


def test_improper_stencil_size():
    # Test for improper stencil size that raises SFXM
    cell_sub = [Cell(0, array([1])), Cell(1, array([2]))]
    with pytest.raises(SFXM):
        Dx(lambda v: v, cell_sub, FDSchemes.CENTRAL)
