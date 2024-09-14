import pytest
import numpy as np

from splitfxm.bc import apply_BC
from splitfxm.derivatives import Dx, FDSchemes
from splitfxm.domain import Domain
from splitfxm.refine import Refiner
from splitfxm.simulation import Simulation


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
    return Domain.from_size(5, 1, 1, ["u", "v"])


@pytest.fixture
def mock_model():
    return MockModel([MockEquation()])


@pytest.fixture
def mock_ic():
    return {"u": "gaussian", "v": "rarefaction"}


@pytest.fixture
def mock_bc():
    return {"u": {"left": "periodic",  "right": "periodic"}, "v": {"left": "periodic", "right": "periodic"}}


@pytest.fixture
def mock_scheme():
    return FDSchemes.CENTRAL


@pytest.fixture
def simulation(mock_domain, mock_model, mock_scheme, mock_ic, mock_bc):
    return Simulation(
        d=mock_domain,
        m=mock_model,
        ics=mock_ic,
        bcs=mock_bc,
        scheme=mock_scheme
    )


def test_simulation_initialization(mock_domain, mock_model, mock_scheme, mock_ic, mock_bc):
    simulation = Simulation(
        d=mock_domain,
        m=mock_model,
        ics=mock_ic,
        bcs=mock_bc,
        scheme=mock_scheme
    )

    assert simulation._d == mock_domain
    assert simulation._s._model == mock_model
    assert simulation._s._scheme == mock_scheme
    assert isinstance(simulation._r, Refiner)
    assert simulation._bcs == mock_bc
    assert simulation._ss == {}


def test_evolve(simulation):
    simulation.evolve(dt=0.1, refinement=False)


def test_initialize_from_list(simulation):
    # Initialize with a simple list of values
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    simulation.initialize_from_list(values)


def test_get_residuals_from_list(simulation):
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    residuals = simulation.get_residuals_from_list(values)


def test_steady_state(simulation):
    num_iterations = simulation.steady_state(
        split=False, sparse=True, dt0=0.0, dtmax=1.0, armijo=False)

    assert num_iterations >= 0  # Ensure the number of iterations is non-negative
    # Additional assertions can be added based on the specific implementation of steady_state


def test_initialize_from_list_split(simulation):
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    split_loc = 1
    simulation.initialize_from_list(values, split=True, split_loc=split_loc)
