import sympy as sp

# Define primitive variables
rho, u, p = sp.symbols('rho u p')
gamma = sp.Symbol('gamma', positive=True)

# Conservative variables
rhoU = rho * u
E = p / (gamma - 1) + 0.5 * rho * u**2

# Flux in terms of primitive variables
F_primitive = sp.Matrix([
    rho * u,              # Mass flux
    rho * u**2 + p,        # Momentum flux
    u * (E + p)            # Energy flux
])

# Define conservative variables U1 = rho, U2 = rho*u, U3 = E
U1, U2, U3 = sp.symbols('U1 U2 U3')

# Express primitive variables in terms of U1, U2, U3
rho_expr = U1           # U1 = rho
u_expr = U2 / U1        # U2 = rho * u
# U3 = E, p = (gamma - 1)*(E - 0.5 * rho * u^2)
p_expr = (gamma - 1) * (U3 - 0.5 * U2**2 / U1)

# Substitute primitive variables with expressions in terms of U1, U2, U3
F_conservative = F_primitive.subs(
    {rho: U1, u: U2 / U1, p: (gamma - 1) * (U3 - 0.5 * U2**2 / U1)})

# Simplify the flux expressions
F_conservative = sp.simplify(F_conservative)
sp.pprint(F_conservative)

# Verify with the analytical expression
F2 = 0.5*(3-gamma)*U2**2/U1 + (gamma-1)*U3
F3 = gamma*U2*U3/U1 - 0.5*(gamma-1)*(U2**3/U1**2)

assert sp.simplify(F_conservative[0] - U2) == 0
assert sp.simplify(F_conservative[1] - F2) == 0
assert sp.simplify(F_conservative[2] - F3) == 0

# Calculate the Jacobian dF/dU
dF_dU = F_conservative.jacobian([U1, U2, U3])

# Display the result
sp.pprint(sp.simplify(dF_dU))

# Verify with the analytical expression
# First row
assert sp.simplify(dF_dU[0, :] - sp.Matrix([[0, 1, 0]])
                   ) == sp.Matrix([[0, 0, 0]])

# Second row
D4 = -0.5*(3 - gamma)*(U2**2/U1**2)
D5 = (3 - gamma)*U2/U1
D6 = gamma - 1
assert sp.simplify(dF_dU[1, :] - sp.Matrix([[D4, D5, D6]])) == sp.Matrix(
    [[0, 0, 0]])

# Third row
D7 = -gamma*U2*U3/U1**2 + (gamma - 1)*(U2**3/U1**3)
D8 = gamma*U3/U1 - 1.5*(gamma - 1)*(U2**2/U1**2)
D9 = gamma * U2/U1
assert sp.simplify(dF_dU[2, :] - sp.Matrix([[D7, D8, D9]])) == sp.Matrix(
    [[0, 0, 0]])
