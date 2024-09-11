import pytest
import numpy as np

from splitfxm.cell import Cell
from splitfxm.domain import Domain
from splitfxm.initialize import set_initial_condition  # Adjust import as needed


@pytest.fixture
def domain():
    # Create a domain using Domain.from_size with boundaries and cells
    nx = 5
    nb_left = 1
    nb_right = 1
    components = ["comp1"]
    return Domain.from_size(nx, nb_left, nb_right, components)


def test_set_initial_condition_tophat(domain):
    set_initial_condition(domain, "comp1", "tophat")
    for cell in domain.interior():
        if 0.333 <= cell.x() <= 0.666:
            assert cell.value(0) == 1.0
        else:
            assert cell.value(0) == 0.0


def test_set_initial_condition_sine(domain):
    set_initial_condition(domain, "comp1", "sine")
    for cell in domain.interior():
        if 0.333 <= cell.x() <= 0.666:
            expected_value = 1.0 + 0.5 * \
                np.sin(2.0 * np.pi * (cell.x() - 0.333) / 0.333)
            assert np.isclose(cell.value(0), expected_value)
        else:
            assert np.isclose(cell.value(0), 1.0)


def test_set_initial_condition_rarefaction(domain):
    set_initial_condition(domain, "comp1", "rarefaction")
    for cell in domain.interior():
        assert cell.value(0) == (2.0 if cell.x() > 0.5 else 1.0)


def test_set_initial_condition_gaussian(domain):
    set_initial_condition(domain, "comp1", "gaussian")
    for cell in domain.interior():
        expected_value = np.exp(-200 * (cell.x() - 0.25) ** 2.0)
        assert np.isclose(cell.value(0), expected_value)


def test_set_initial_condition_invalid_type(domain):
    with pytest.raises(NotImplementedError):
        set_initial_condition(domain, "comp1", "invalid_type")
