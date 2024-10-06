from splitfxm.domain import Domain
from splitfxm.simulation import Simulation
from splitfxm.models.advection_diffusion import AdvectionDiffusion
from splitfxm.schemes import default_scheme
from splitfxm.visualize import draw
from matplotlib import pyplot as plt

# Define the problem
method = 'FDM'
m = AdvectionDiffusion(c=0.2, nu=0.001, method=method)
d = Domain.from_size(100, 1, 1, ["u", "v", "w"])
ics = {"u": "gaussian", "v": "rarefaction", "w": "tophat"}
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

# Advance in time or to steady state
s.evolve(t_diff=0.1)

# Visualize
draw(d, "label")
plt.legend(["$u$", "$v$", "$w$"])
plt.xlabel("x")
plt.show()
