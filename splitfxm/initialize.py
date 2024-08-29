import numpy as np
from .domain import Domain

# https://github.com/comp-physics/WENO-scalar/blob/master/params.py
def set_initial_condition(d: Domain, v: str, type="gaussian"):
    """
    Set the initial condition of a given component in a domain.

    Parameters
    ----------
    d : Domain
        The domain to set the initial condition in.
    v : str
        The component to set the initial condition for.
    type : str, optional
        The type of initial condition to set. Options are: 'tophat', 'sine', 'rarefaction', 'gaussian'. Default is 'gaussian'.

    Raises
    ------
    NotImplementedError
        If the specified type of initial condition is not implemented.
    """

    # Find index of component
    idx = d.component_index(v)

    # Some initial conditions
    if type == "tophat":
        for cell in d.interior():
            cell.set_value(
                idx, 1.0 if (cell.x() >= 0.333 and cell.x() <= 0.666) else 0.0
            )
    elif type == "sine":
        for cell in d.interior():
            cell.set_value(
                idx,
                1.0 + 0.5 * np.sin(2.0 * np.pi * (cell.x() - 0.333) / 0.333)
                if (cell.x() >= 0.333 and cell.x() <= 0.666)
                else 1.0,
            )
    elif type == "rarefaction":
        for cell in d.interior():
            cell.set_value(idx, 2.0 if cell.x() > 0.5 else 1.0)
    elif type == "gaussian":
        for cell in d.interior():
            cell.set_value(idx, np.exp(-200 * (cell.x() - 0.25) ** 2.0))
    else:
        raise NotImplementedError
