from splitfxm.domain import Domain
from splitfxm.simulation import Simulation
from euler1d import Euler1DConservative
from splitfxm.schemes import default_scheme, FVSchemes
from splitfxm.visualize import draw
from verification_data import *
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
s.evolve(split=True, split_locs=[1], t_diff=0.2)

# Visualize the results at a specific time step
values = [m.conservative_to_primitive(cell_values)
          for cell_values in s._d.values(interior=True)]
plt.plot(d.positions(interior=True), values, "-")

# Plot the verification data
plt.gca().set_prop_cycle(None)
plt.plot(density_x, density_y, "s", markersize=3)
plt.plot(velocity_x, velocity_y, "s", markersize=3)
plt.plot(pressure_x, pressure_y, "s", markersize=3)

# Properly formatted labels and legend entries
plt.legend([r"$\rho/\rho_{0}$", r"$u$", r"$p/p_{0}$",
           r"$\rho/\rho_{0}$ Exact", r"$u$ Exact", r"$p/p_{0}$ Exact"])
plt.xlabel("x")
plt.ylabel(r"Normalized $\rho$, $u$, Normalized $p$")

plt.show()
