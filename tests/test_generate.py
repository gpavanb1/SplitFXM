import pytest
from sympy import symbols
from splitfxm.fdm_schemes.generate import compact_scheme
from splitfxm.error import SFXM


def test_valid_compact_scheme_first_derivative():
    # Test the first derivative with a balanced stencil
    left_stencil_size = 1
    right_stencil_size = 1
    derivative_order = 1

    result = compact_scheme(
        left_stencil_size, right_stencil_size, derivative_order)


def test_valid_compact_scheme_second_derivative():
    # Test the second derivative with a balanced stencil
    left_stencil_size = 1
    right_stencil_size = 1
    derivative_order = 2

    result = compact_scheme(
        left_stencil_size, right_stencil_size, derivative_order)


def test_invalid_stencil_size():
    # Test with invalid stencil sizes
    with pytest.raises(SFXM, match="Invalid stencil size or derivative order"):
        compact_scheme(-1, 1, 1)

    with pytest.raises(SFXM, match="Invalid stencil size or derivative order"):
        compact_scheme(1, -1, 1)


def test_invalid_derivative_order():
    # Test with invalid derivative order
    with pytest.raises(SFXM, match="Invalid stencil size or derivative order"):
        compact_scheme(1, 1, -1)
