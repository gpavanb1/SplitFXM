import pytest
import numpy as np

from splitfxm.domain import Domain
from splitfxm.derivatives import Dx, FDSchemes
from splitfxm.system import System


class MockEquation:
    def residuals(self, cell_sub, scheme):
        # Simple mock behavior for testing purposes
        return Dx(lambda x: x, cell_sub, scheme)


class MockModel:
    def __init__(self, equations):
        self._equations = equations

    def equations(self):
        return self._equations


@pytest.fixture
def mock_domain():
    return Domain.from_size(5, 1, 1, ["comp1"])


def test_system_initialization():
    model = MockModel([MockEquation()])
    scheme = FDSchemes.CENTRAL
    system = System(model, scheme)
    assert system._model == model
    assert system._scheme == scheme


def test_residuals(mock_domain):
    model = MockModel([MockEquation()])
    scheme = FDSchemes.CENTRAL
    system = System(model, scheme)

    rhs_list = system.residuals(mock_domain)

    # Check that rhs_list is the correct length
    assert len(rhs_list) == 5

    expected_residual = np.array([0.0] * 5)
    assert np.allclose(rhs_list[0], expected_residual)
