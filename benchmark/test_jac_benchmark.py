import numdifftools as nd
import numpy as np
import pytest
import time


from splitfxm.domain import Domain
from splitfxm.schemes import FDSchemes
from splitfxm.simulation import Simulation
from splitfxm.schemes import default_scheme
from splitfxm.models.advection_diffusion import AdvectionDiffusion


def test_dense_sparse_jac_comparison():
    method = 'FDM'
    m = AdvectionDiffusion(c=0.2, nu=0.001, method=method)
    d = Domain.from_size(250, 1, 1, ["u", "v", "w"])
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

    # Some output
    print('\n' + '*' * 10)
    print('SYMMETRIC STENCIL')
    print('*' * 10)
    print('Computing...')

    # Construct dense Jacobian
    def _f(u): return s.get_residuals_from_list(u, split, split_locs)
    start_time = time.perf_counter()
    jac_dense = nd.Jacobian(_f, method='forward', step=1e-8)(x0)
    dense_time = time.perf_counter() - start_time
    print(f'Dense Jacobian took {dense_time} s')

    # Construct sparse Jacobian
    start_time = time.perf_counter()
    jac_sparse = s.jacobian(x0, split, split_locs)
    sparse_time = time.perf_counter() - start_time
    print(f'Sparse Jacobian took {sparse_time} s')

    # Show timing results in prompt in pytest
    assert sparse_time < dense_time
    assert np.allclose(jac_sparse.toarray(), jac_dense, atol=1e-7)


def test_dense_sparse_jac_comparison_asym():
    method = 'FDM'
    m = AdvectionDiffusion(c=0.2, nu=0.001, method=method)
    d = Domain.from_size(250, 1, 2, ["u", "v", "w"])
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
    s = Simulation(d, m, ics, bcs, FDSchemes.RIGHT_BIAS)
    split = True
    split_locs = [1]

    # Construct initial vector
    x0 = d.listify_interior(split, split_locs)

    # Some output
    print('\n' + '*' * 10)
    print('ASYMMETRIC STENCIL')
    print('*' * 10)
    print('Computing...')

    # Construct dense Jacobian
    def _f(u): return s.get_residuals_from_list(u, split, split_locs)
    start_time = time.perf_counter()
    jac_dense = nd.Jacobian(_f, method='forward', step=1e-8)(x0)
    dense_time = time.perf_counter() - start_time
    print(f'Dense Jacobian took {dense_time} s')

    # Construct sparse Jacobian
    start_time = time.perf_counter()
    jac_sparse = s.jacobian(x0, split, split_locs)
    sparse_time = time.perf_counter() - start_time
    print(f'Sparse Jacobian took {sparse_time} s')

    # Show timing results in prompt in pytest
    assert sparse_time < dense_time
    assert np.allclose(jac_sparse.toarray(), jac_dense, atol=1e-7)


# Run the tests
if __name__ == "__main__":
    pytest.main()
