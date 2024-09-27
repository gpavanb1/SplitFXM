import numpy as np
import pytest
from copy import deepcopy
import matplotlib.pyplot as plt

from splitfxm.domain import Domain
from splitfxm.error import SFXM
from splitfxm.initialize import set_initial_condition
from splitfxm.models.advection_diffusion import AdvectionDiffusion
from splitfxm.simulation import Simulation
from splitfxm.schemes import default_scheme, FVSchemes, stencil_sizes
from splitfxm.visualize import draw


def test_invalid_method():
    with pytest.raises(SFXM):
        m = AdvectionDiffusion(c=0.1, nu=0.0, method="INVALID_METHOD")


def test_validation_all_schemes():
    THRESHOLD = 2e-2
    method = 'FVM'
    m = AdvectionDiffusion(c=0.1, nu=0.0, method=method)

    schemes = [FVSchemes.LAX_FRIEDRICHS, FVSchemes.UPWIND, FVSchemes.CENTRAL, FVSchemes.LAX_WENDROFF,
               FVSchemes.QUICK, FVSchemes.BQUICK, FVSchemes.MUSCL, FVSchemes.ENO, FVSchemes.WENO]

    for scheme in schemes:

        nb = stencil_sizes[scheme] // 2

        d = Domain.from_size(200, nb, nb, ["u"])
        d_copy = deepcopy(d)
        ics = {"u": "tophat"}
        bcs = {
            "u": {
                "left": "periodic",
                "right": "periodic"
            },
        }

        s = Simulation(d, m, ics, bcs, scheme)
        split = False
        split_loc = None

        s.evolve(10., split, split_loc, method='Euler', max_step=0.05)
        set_initial_condition(d_copy, "u", "tophat")

        # Extract the expected and actual solution
        expected_u = np.array([x[0] for x in d_copy.values(interior=True)])
        actual_u = np.array([x[0] for x in s._d.values(interior=True)])

        # Calculate the absolute difference
        abs_diff = np.abs(expected_u - actual_u)

        # Compute the area under the curve using the trapezoidal rule
        area = np.trapezoid(abs_diff, dx=1/len(expected_u))

        # Check if the area is below the threshold
        assert area < THRESHOLD, {"scheme": scheme, "area": area}


def test_validation():
    # Check Advection-Diffusion using FDM and FVM
    method = 'FVM'
    m = AdvectionDiffusion(c=0.1, nu=0.0, method=method)
    d = Domain.from_size(100, 1, 1, ["u", "v"])
    d_copy = deepcopy(d)
    ics = {"u": "sine", "v": "gaussian"}
    bcs = {
        "u": {
            "left": "periodic",
            "right": "periodic"
        },
        "v": {
            "left": "periodic",
            "right": "periodic"
        }
    }
    s = Simulation(d, m, ics, bcs, default_scheme(method))
    split = False
    split_loc = None

    # Evolve over time using Euler
    dt = 10./100
    for i in range(101):
        s.evolve(dt, split, split_loc, method='Euler', max_step=0.1)

    # Construct expected solution
    set_initial_condition(d_copy, "u", "sine")
    set_initial_condition(d_copy, "v", "gaussian")

    # Check that the solution is valid
    expected_u = np.array([x[0] for x in d_copy.values(interior=True)])
    expected_v = np.array([x[1] for x in d_copy.values(interior=True)])

    actual_u = np.array([x[0] for x in s._d.values(interior=True)])
    actual_v = np.array([x[1] for x in s._d.values(interior=True)])

    draw(s._d, "Actual", interior=True)
    draw(d_copy, "Expected", interior=True)
    plt.legend()
    plt.show()

    assert np.allclose(expected_u, actual_u, atol=1e-1)
    assert np.allclose(expected_v, actual_v, atol=1e-1)
