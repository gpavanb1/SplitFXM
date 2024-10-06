from splitfxm.domain import Domain
from splitfxm.simulation import Simulation
from euler1d import Euler1DConservative
from splitfxm.schemes import default_scheme, FVSchemes
from splitfxm.visualize import draw
import matplotlib.pyplot as plt

# Define the problem
method = 'FVM'
# Euler equations with gamma for ideal gas
m = Euler1DConservative(gamma=1.4, method=method)

# Create a domain for the shock tube
# nx = number of cells, nb_left = 1 ghost cell, nb_right = 1 ghost cell
# Variables: ["rho", "u", "p"] (density, velocity, pressure)
d = Domain.from_size(100, 1, 1, ["rho", "rhou", "E"])

# Boundary conditions (transmissive for both sides)
ics = {}
bcs = {"rho": {"left": {"neumann": 0.}, "right": {"neumann": 0.}},
       "rhou": {"left": {"neumann": 0.}, "right": {"neumann": 0.}},
       "E": {"left": {"neumann": 0.}, "right": {"neumann": 0.}}}

# Create a simulation object
s = Simulation(d, m, ics, bcs, default_scheme(method))

# Initial conditions
for cell in s._d.interior():
    cell.set_value(s._d.component_index("rho"),
                   1.0 if cell.x() < 0.5 else 0.125)
    cell.set_value(s._d.component_index("rhou"), 0.0)
    cell.set_value(s._d.component_index("E"), 2.5 if cell.x() <
                   0.5 else 0.25)  # p/(gamma - 1) + 0.5*U2**2/U1

# Evolve in time (using a small time step due to the shock)
s.evolve(split=False, split_loc=1, t_diff=0.2)

# Visualize the results at a specific time step
values = [m.conservative_to_primitive(cell_values)
          for cell_values in s._d.values(interior=True)]
plt.plot(d.positions(interior=True), values, "-o")

# Properly formatted labels and legend entries
plt.legend([r"$\rho/\rho_{0}$", r"$u$", r"$p/p_{0}$"])
plt.xlabel("x")
plt.ylabel(r"Normalized $\rho$, $u$, Normalized $p$")

plt.show()
