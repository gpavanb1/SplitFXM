import pytest
import numpy as np
import numdifftools as nd

from splitfxm.bc import apply_BC
from splitfxm.derivatives import Dx
from splitfxm.domain import Domain
from splitfxm.error import SFXM
from splitfxm.refine import Refiner
from splitfxm.schemes import default_scheme, FDSchemes
from splitfxm.simulation import Simulation
from splitfxm.models.advection_diffusion import AdvectionDiffusion


class MockEquation:
    def residuals(self, cell_sub, scheme, scheme_opts={}):
        # Simple mock behavior for testing purposes
        return Dx(lambda x: x, cell_sub, scheme)


class MockModel:
    def __init__(self, equation):
        self._equation = equation

    def equation(self):
        return self._equation


@pytest.fixture
def mock_domain():
    return Domain.from_size(5, 1, 1, ["u", "v"])


@pytest.fixture
def mock_model():
    return MockModel(MockEquation())


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


def test_improper_stencil_size():
    # Test for improper stencil size that raises SFXM
    d = Domain.from_size(5, 1, 2, ["u", "v"])
    m = MockModel(MockEquation())
    with pytest.raises(SFXM):
        s = Simulation(d, m, {}, {}, FDSchemes.CENTRAL)


def test_evolve(simulation):
    simulation.evolve(t_diff=0.1, split=True, split_locs=[1], method='Radau')


def test_initialize_from_list(simulation):
    # Initialize with a simple list of values
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    simulation.initialize_from_list(values)


def test_initialize_from_list_wrong_length(simulation):
    # Initialize with a simple list of values
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    with pytest.raises(SFXM):
        simulation.initialize_from_list(values)


def test_get_residuals_from_list(simulation):
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    residuals = simulation.get_residuals_from_list(values)


def test_steady_state(simulation):
    num_iterations = simulation.steady_state(
        split=False, sparse=True, dt0=0.0, dtmax=1.0, armijo=False)

    assert num_iterations >= 0  # Ensure the number of iterations is non-negative
    # Additional assertions can be added based on the specific implementation of steady_state


def test_steady_state_split(simulation):
    num_iterations = simulation.steady_state(
        split=True, split_locs=[1], sparse=True, dt0=0.0, dtmax=1.0, armijo=False)

    assert num_iterations >= 0  # Ensure the number of iterations is non-negative
    # Additional assertions can be added based on the specific implementation of steady_state


def test_initialize_from_list_split(simulation):
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    split_locs = [1]
    simulation.initialize_from_list(values, split=True, split_locs=split_locs)


def test_initialize_from_list_split_wrong_loc(simulation):
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    split_locs = [4]
    with pytest.raises(SFXM):
        simulation.initialize_from_list(
            values, split=True, split_locs=split_locs)


def test_extend_bounds_no_split(simulation):
    bounds = [[1, 2], [3, 4]]
    nv = 2
    num_points = 3
    result = simulation.extend_bounds(bounds, num_points, nv, split=False)
    assert result == [[1, 2] * num_points, [3, 4] *
                      num_points], "Bounds should extend without split"


def test_extend_bounds_split(simulation):
    bounds = [[1, 2], [3, 4]]
    nv = 2
    num_points = 2
    split_locs = [1]
    result = simulation.extend_bounds(
        bounds, num_points, nv, split=True, split_locs=split_locs)
    expected_result = [[1] * num_points + [2] *
                       num_points, [3] * num_points + [4] * num_points]
    assert result == expected_result, "Bounds should extend with split at the given location"


def test_extend_bounds_invalid_size(simulation):
    bounds = [[1], [2, 3]]  # Mismatched sizes
    nv = 2
    num_points = 2
    with pytest.raises(SFXM):
        simulation.extend_bounds(bounds, num_points, nv)


def test_extend_bounds_invalid_input(simulation):
    bounds = [[1], [2], [3]]  # Mismatched sizes
    nv = 2
    num_points = 2
    with pytest.raises(SFXM):
        simulation.extend_bounds(bounds, num_points, nv)


def test_extend_bounds_invalid_split_loc(simulation):
    bounds = [[1, 2], [3, 4]]
    nv = 2
    num_points = 3
    with pytest.raises(SFXM):
        simulation.extend_bounds(bounds, num_points, nv, split=True)


def test_bounds_exceed_evolve():
    method = 'FDM'
    m = AdvectionDiffusion(c=0.2, nu=0.0, method=method)
    ics = {"u": "gaussian"}
    bcs = {"u": {"left": "periodic", "right": "periodic"}}
    d = Domain.from_size(10, 1, 1, ["u"])
    s = Simulation(d, m, ics, bcs, default_scheme(method))
    bounds = [[0.1], [0.5]]
    split = False
    split_loc = None
    t = 0.1
    # Apply bounds for both kinds of methods
    s.evolve(t, split, split_loc, method='RK45', max_step=0.1, bounds=bounds)
    s.evolve(t, split, split_loc, method='Euler', max_step=0.1, bounds=bounds)


def test_dense_sparse_jac_comparison_steady_state():
    method = 'FDM'
    m = AdvectionDiffusion(c=0.2, nu=0.001, method=method)
    d = Domain.from_size(15, 1, 1, ["u", "v", "w"])
    ics = {"u": "gaussian", "v": "rarefaction"}
    bcs = {
        "u": {
            "left": "periodic",
            "right": "periodic"
        },
        "v": {
            "left": {"dirichlet": 3},
            "right": {"dirichlet": 4}
        },
        "w": {
            "left": {"dirichlet": 2},
            "right": "periodic"
        }
    }
    s = Simulation(d, m, ics, bcs, default_scheme(method))
    split = True
    split_locs = [1]

    # Construct initial vector
    x0 = d.listify_interior(split, split_locs)

    # Construct dense Jacobian
    def _f(u): return s.get_residuals_from_list(u, split, split_locs)
    jac_dense = nd.Jacobian(_f, method='forward', step=1e-8)(x0)

    # Construct Jacobian with no split location
    with pytest.raises(SFXM):
        jac_sparse = s.jacobian(x0, split)

    # Construct sparse Jacobian
    jac_sparse = s.jacobian(x0, split, split_locs)

    # Show timing results in prompt in pytest
    assert np.allclose(jac_sparse.toarray(), jac_dense, atol=1e-7)

    # Repeat the same with FVM
    method = 'FVM'
    m = AdvectionDiffusion(c=0.2, nu=0.001, method=method)
    s = Simulation(d, m, ics, bcs, default_scheme(method))
    split = True
    split_locs = [1]

    # Construct initial vector
    x0 = d.listify_interior(split, split_locs)

    # Construct dense Jacobian
    def _f(u): return s.get_residuals_from_list(u, split, split_locs)
    jac_dense = nd.Jacobian(_f, method='forward', step=1e-8)(x0)

    # Construct Jacobian with no split location
    with pytest.raises(SFXM):
        jac_sparse = s.jacobian(x0, split)

    # Construct sparse Jacobian
    jac_sparse = s.jacobian(x0, split, split_locs)

    # Show timing results in prompt in pytest
    assert np.allclose(jac_sparse.toarray(), jac_dense, atol=1e-7)
