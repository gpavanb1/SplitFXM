import pytest
import numpy as np

from splitfxm.cell import Cell
from splitfxm.derivatives import derivative, Dx, dx, D2x, d2x
from splitfxm.schemes import FDSchemes
from splitfxm.error import SFXM


def test_derivative_first_central():
    # Test first derivative with CENTRAL scheme
    cell_sub = [Cell(0, np.array([1])), Cell(
        1, np.array([2])), Cell(2, np.array([3]))]
    result = derivative(lambda v: v[0], cell_sub, FDSchemes.CENTRAL, order=1)
    assert result == 1.0


def test_derivative_first_right_bias():
    # Test first derivative with RIGHT_BIAS scheme
    cell_sub = [Cell(0, np.array([1])), Cell(1, np.array([2])),
                Cell(2, np.array([3])), Cell(3, np.array([4]))]
    result = derivative(lambda v: v[0], cell_sub,
                        FDSchemes.RIGHT_BIAS, order=1)
    assert np.allclose(result, np.array([1.]))


def test_derivative_second_central():
    # Test second derivative with CENTRAL scheme
    cell_sub = [Cell(0, np.array([1])), Cell(
        1, np.array([2])), Cell(2, np.array([3]))]
    result = derivative(lambda v: v[0], cell_sub, FDSchemes.CENTRAL, order=2)
    assert result == 0.0


def test_derivative_second_right_bias():
    # Test second derivative with RIGHT_BIAS scheme
    cell_sub = [Cell(0, np.array([1])), Cell(1, np.array([2])),
                Cell(2, np.array([3])), Cell(3, np.array([4]))]
    result = derivative(lambda v: v[0], cell_sub,
                        FDSchemes.RIGHT_BIAS, order=2)
    assert result == 0.0


def test_dx_first_central():
    # Test dx for precomputed values with CENTRAL scheme
    cell_sub = [Cell(0, np.array([1])), Cell(
        1, np.array([2])), Cell(2, np.array([3]))]
    values = [np.array([1]), np.array([2]), np.array([3])]
    result = dx(values, cell_sub, FDSchemes.CENTRAL)
    assert result == 1.0


def test_dx_first_right_bias():
    # Test dx for precomputed values with RIGHT_BIAS scheme
    cell_sub = [Cell(0, np.array([1])), Cell(1, np.array([2])),
                Cell(2, np.array([3])), Cell(3, np.array([4]))]
    values = [np.array([1]), np.array([2]), np.array([3]), np.array([4])]
    result = dx(values, cell_sub, FDSchemes.RIGHT_BIAS)
    assert np.allclose(result, np.array([1.]))


def test_d2x_second_central():
    # Test d2x for precomputed values with CENTRAL scheme
    cell_sub = [Cell(0, np.array([1])), Cell(
        1, np.array([2])), Cell(2, np.array([3]))]
    values = [np.array([1]), np.array([2]), np.array([3])]
    result = d2x(values, cell_sub, FDSchemes.CENTRAL)
    assert result == 0.0


def test_d2x_second_right_bias():
    # Test d2x for precomputed values with RIGHT_BIAS scheme
    cell_sub = [Cell(0, np.array([1])), Cell(1, np.array([2])),
                Cell(2, np.array([3])), Cell(3, np.array([4]))]
    values = [np.array([1]), np.array([2]), np.array([3]), np.array([4])]
    result = d2x(values, cell_sub, FDSchemes.RIGHT_BIAS)
    assert result == 0.0


def test_improper_stencil_size():
    # Test for improper stencil size that raises SFXM
    cell_sub = [Cell(0, np.array([1])), Cell(1, np.array([2]))]
    with pytest.raises(SFXM):
        derivative(lambda v: v[0], cell_sub, FDSchemes.CENTRAL)


def test_unsupported_scheme_or_order():
    # Test for unsupported scheme or order that raises SFXM
    cell_sub = [Cell(0, np.array([1])), Cell(
        1, np.array([2])), Cell(2, np.array([3]))]
    with pytest.raises(SFXM):
        derivative(lambda v: v[0], cell_sub, FDSchemes.CENTRAL, order=3)


def test_Dx_first_central():
    # Test Dx function with CENTRAL scheme
    cell_sub = [Cell(0, np.array([1])), Cell(
        1, np.array([2])), Cell(2, np.array([3]))]
    result = Dx(lambda v: v[0], cell_sub, FDSchemes.CENTRAL)
    assert result == pytest.approx(1.0)


def test_Dx_first_right_bias():
    # Test Dx function with RIGHT_BIAS scheme
    cell_sub = [Cell(0, np.array([1])), Cell(1, np.array([2])),
                Cell(2, np.array([3])), Cell(3, np.array([4]))]
    result = Dx(lambda v: v[0], cell_sub, FDSchemes.RIGHT_BIAS)
    assert result == pytest.approx(1.0)


def test_D2x_second_central():
    # Test D2x function with CENTRAL scheme
    cell_sub = [Cell(0, np.array([1])), Cell(
        1, np.array([4])), Cell(2, np.array([9]))]
    result = D2x(lambda v: v[0], cell_sub, FDSchemes.CENTRAL)
    assert result == pytest.approx(2.0)


def test_D2x_second_right_bias():
    # Test D2x function with RIGHT_BIAS scheme
    cell_sub = [Cell(0, np.array([1])), Cell(1, np.array([4])),
                Cell(2, np.array([9])), Cell(3, np.array([16]))]
    result = D2x(lambda v: v[0], cell_sub, FDSchemes.RIGHT_BIAS)
    assert result == pytest.approx(2.0)
