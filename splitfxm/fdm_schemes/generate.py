import numpy as np
from sympy import symbols, Eq, solve, factorial, sign


def compact_scheme(left_stencil, right_stencil, derivative_order):
    """
    Generates a compact finite difference scheme for uneven grid spacing.

    Parameters
    ----------
    left_stencil : int
        The number of points on the left of the reference point in the stencil.
    right_stencil : int
        The number of points on the right of the reference point in the stencil.
    derivative_order : int
        The order of the derivative to approximate (1 for first derivative, 2 for second derivative, etc.).

    Returns
    -------
    solution : dict
        A dictionary containing the coefficients of the finite difference scheme.

    Notes
    -----
    This function creates Taylor series expansions for the stencil points using uneven grid spacing, where
    the grid spacing is denoted by symbols (h_j). The central point is set with a spacing of zero, and the
    coefficients are determined by solving a system of linear equations.

    The sign function from sympy is used to handle the directionality (positive/negative) of the grid spacing.

    Example
    -------
    ```python
    >>> left_stencil_size = 1
    >>> right_stencil_size = 1
    >>> order = 1  # First derivative
    >>> coefficients = compact_scheme(left_stencil_size, right_stencil_size, order)
    >>> print(f"Coefficients for f^({order}):", coefficients)
    ```
    """

    # Create symbols for the coefficients of the scheme
    coefficients = symbols(f'a0:{left_stencil + right_stencil + 1}')

    # Define the uneven grid spacing symbols
    # Set the zero element grid distance to zero
    # Note that h_j denotes distance from the zero element
    grid_spacing = list(symbols(f'h:{left_stencil + right_stencil + 1}'))
    grid_spacing[left_stencil] = 0
    grid_spacing = tuple(grid_spacing)

    # Taylor series expansions for points in the stencil
    equations = []

    # Loop through the orders of the Taylor expansion (0, 1, 2,..., derivative_order)
    for k in range(derivative_order + 1):
        # Generate the k-th order Taylor expansion
        taylor_expansion = sum(((grid_spacing[i + left_stencil] * sign(i)) ** k / factorial(k)) *
                               coefficients[i + left_stencil] for i in range(-left_stencil, right_stencil + 1))

        # Append the equation: 1 for the target derivative, 0 otherwise
        expected_value = 1 if k == derivative_order else 0
        equations.append(Eq(taylor_expansion, expected_value))

    # Solve the system of equations for the coefficients
    solution = solve(equations, coefficients)

    return solution
