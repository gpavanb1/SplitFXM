from splitfxm.domain import Domain
from splitfxm.simulation import Simulation
from splitfxm.models.advection_diffusion import AdvectionDiffusion
from splitfxm.schemes import default_scheme
from splitfxm.visualize import draw
import matplotlib.pyplot as plt

# Define the problem
method = 'FVM'
m = AdvectionDiffusion(c=0.2, nu=0.001, method=method)
# nx, nb_left, nb_right, variables
d = Domain.from_size(20, 1, 1, ["u", "v", "w"])
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
bounds = [[-1., -2., -2.], [5., 4., 3.]]
iter = s.steady_state(split=True, split_loc=1, bounds=bounds)

# Visualize
draw(d, "label")
plt.show()
