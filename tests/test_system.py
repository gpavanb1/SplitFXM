import pytest
import numpy as np

from splitfxm.domain import Domain
from splitfxm.derivatives import Dx
from splitfxm.flux import fluxes
from splitfxm.schemes import FDSchemes, FVSchemes
from splitfxm.system import System


class MockFDEquation:
    def residuals(self, cell_sub, scheme, scheme_opts={}):
        # Simple mock behavior for testing purposes
        return Dx(lambda x: x, cell_sub, scheme)


class MockFVEquation:
    def __init__(self):
        # Mock behavior for testing purposes
        self.c = 0.1
        self.F = lambda u: np.array([self.c * x for x in u])
        self.dFdU = lambda x: np.diag([self.c] * len(x))

    def residuals(self, cell_sub, scheme, scheme_opts={}):
        # Simple mock behavior for testing purposes
        limiter = scheme_opts.get("limiter")
        Fw, Fe = fluxes(self.F, cell_sub, scheme, self.dFdU, limiter)
        dx = cell_sub[1].x() - cell_sub[0].x()
        return (Fe - Fw) / (dx)


class MockModel:
    def __init__(self, equations):
        self._equations = equations

    def equations(self):
        return self._equations


@pytest.fixture
def mock_domain():
    return Domain.from_size(5, 1, 1, ["comp1"])


def test_system_initialization():
    model = MockModel([MockFDEquation()])
    scheme = FDSchemes.CENTRAL
    system = System(model, scheme, scheme_opts={})
    assert system._model == model
    assert system._scheme == scheme


def test_fd_residuals(mock_domain):
    model = MockModel([MockFDEquation()])
    fd_scheme = FDSchemes.CENTRAL

    # Check for FD
    system = System(model, fd_scheme, scheme_opts={})
    rhs_list = system.residuals(mock_domain)

    # Check that rhs_list is the correct length
    assert len(rhs_list) == 5

    expected_residual = np.array([0.0] * 5)
    assert np.allclose(rhs_list[0], expected_residual)


def test_fv_residuals(mock_domain):
    model = MockModel([MockFVEquation()])
    fv_scheme = FVSchemes.LAX_FRIEDRICHS

    # Check for FV
    system = System(model, fv_scheme, scheme_opts={})
    rhs_list = system.residuals(mock_domain)

    # Check that rhs_list is the correct length
    assert len(rhs_list) == 5

    expected_residual = np.array([0.0] * 5)
    assert np.allclose(rhs_list[0], expected_residual)
