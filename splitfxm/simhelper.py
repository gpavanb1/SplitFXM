import numpy as np
from scipy.optimize import OptimizeResult

# Basic class to assign solution for Euler timestep


class Solution(OptimizeResult):
    pass


def array_list_reshape(l, shape):
    """
    Reshape a list into a list of 1D NumPy arrays.

    Parameters
    ----------
    l : list
        The list to reshape.
    shape : tuple
        The shape to reshape the list into.

    Returns
    -------
    reshaped_list : list of ndarray
        The reshaped list.
    """
    return np.array(l).reshape(shape)


# Evolve event helper methods
def create_bound_events(y0, bounds):
    """
    Create event functions for detecting bound violations during ODE solving.

    Parameters
    ----------
    y0 : array-like
        Initial state of the system variables.
    bounds : tuple of array-like
        Tuple containing lower and upper bounds for the system variables.

    Returns
    -------
    list
        A list of event functions that trigger when a bound is violated.
    """
    events = []
    if bounds is not None:
        lower_bounds, upper_bounds = bounds[0], bounds[1]
        for i in range(len(y0)):
            events.append(create_bound_event(
                i, lower_bounds[i], upper_bounds[i]))
    return events


def create_bound_event(index, lower_bound, upper_bound):
    """
    Create an event function for checking if a system variable exceeds its bounds.

    Parameters
    ----------
    index : int
        Index of the variable to check.
    lower_bound : float
        The lower bound for the variable.
    upper_bound : float
        The upper bound for the variable.

    Returns
    -------
    function
        An event function that triggers when the variable at `index` exceeds its bounds.
        Prints a warning if a bound is exceeded and returns the difference from the bound.
        The event is non-terminal, allowing the integration to continue.
    """
    def event(t, y):
        if y[index] < lower_bound:
            print(
                f"Warning: At t={t}, bounds exceeded for variable index {index}, value = {y[index]}, lower bound = {lower_bound}")
            return y[index] - lower_bound
        elif y[index] > upper_bound:
            print(
                f"Warning: At t={t}, bounds exceeded for variable index {index}, value = {y[index]}, upper bound = {upper_bound}")
            return upper_bound - y[index]
        return 1  # No bounds exceeded, continue integration
    event.terminal = False  # Continue integration after event
    event.direction = 0     # Detect bounds violations in both directions
    return event


def apply_bounds(y, bounds, t_current):
    """
    Check if the system variables exceed specified bounds during Euler integration.

    Parameters
    ----------
    y : array-like
        Current values of the system variables.
    bounds : tuple of array-like
        Tuple containing the lower and upper bounds for the variables.
    t_current : float
        Current time in the integration.

    Returns
    -------
    None
        Prints warnings if any variable exceeds its bounds.
    """
    lower_bounds, upper_bounds = bounds
    for idx, (val, lb, ub) in enumerate(zip(y, lower_bounds, upper_bounds)):
        if val < lb:
            print(
                f"Warning: At t={t_current}, bounds exceeded for variable index {idx}, value = {val}, lower bound = {lb}")
        elif val > ub:
            print(
                f"Warning: At t={t_current}, bounds exceeded for variable index {idx}, value = {val}, upper bound = {ub}")
