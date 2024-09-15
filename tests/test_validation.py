import numpy as np
import pytest
from copy import deepcopy
import matplotlib.pyplot as plt

from splitfxm.domain import Domain
from splitfxm.error import SFXM
from splitfxm.initialize import set_initial_condition
from splitfxm.models.advection_diffusion import AdvectionDiffusion
from splitfxm.simulation import Simulation
from splitfxm.schemes import default_scheme
from splitfxm.visualize import draw


def test_invalid_method():
    with pytest.raises(SFXM):
        m = AdvectionDiffusion(c=0.1, nu=0.0, method="INVALID_METHOD")


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

    # Evolve over time
    # Keep CFL as close to 1 to avoid dissipation
    for i in range(101):
        s.evolve(10./101)

    # Construct expected solution
    set_initial_condition(d_copy, "u", "sine")
    set_initial_condition(d_copy, "v", "gaussian")

    # Check that the solution is valid
    expected_u = d_copy.values(interior=True)[0]
    expected_v = d_copy.values(interior=True)[1]

    actual_u = s._d.values(interior=True)[0]
    actual_v = s._d.values(interior=True)[1]

    draw(s._d, "actual", interior=True)
    draw(d_copy, "expected", interior=True)
    plt.legend()
    plt.show()

    assert np.allclose(expected_u, actual_u, atol=1e-2)
    assert np.allclose(expected_v, actual_v, atol=1e-2)
