from splitfxm.domain import Domain
from splitfxm.simulation import Simulation
from splitfxm.schemes import default_scheme
from splitfxm.visualize import draw
from matplotlib.pyplot import legend, show

from examples.advection_diffusion import AdvectionDiffusion

import argparse
import logging
from copy import deepcopy

# Set logging level
parser = argparse.ArgumentParser()
parser.add_argument(
    "--log",
    dest="loglevel",
    help="Set the loglevel for your solver  (DEBUG, INFO, WARNING, CRITICAL, ERROR)",
    type=str,
    default="WARNING",
)
args = parser.parse_args()
loglevel = getattr(logging, args.loglevel.upper())
logging.basicConfig(level=loglevel)

# Define the problem
method = 'FDM'
m = AdvectionDiffusion(c=0.2, nu=0.001, method=method)
d = Domain.from_size(20, 1, 1, ["u", "v", "w"])
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

# Initial domain
d_init = deepcopy(d)

# Advance in time
s.evolve(0.01)
bounds = [[-1., -2., 0.], [5., 4., 3.]]
iter = s.steady_state(split=True, split_loc=1, bounds=bounds)
print(f"Took {iter} iterations")

# Show plot
draw(d_init, "l1")
draw(d, "l2")
legend()
show()
