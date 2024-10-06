import sympy as sp

# Define symbols
rho, u, p, gamma = sp.symbols('rho u p gamma')

# Define the energy term
E = p / (gamma - 1) + 0.5 * rho * u**2  # Assuming gamma = 1.4

# Define the flux function F
F = sp.Matrix([
    rho * u,                # F1
    rho * u**2 + p,        # F2
    u * (E + p)            # F3
])

# Calculate the Jacobian dF/dU
dF_dU = F.jacobian([rho, u, p])

# Display the result
sp.pprint(dF_dU)
