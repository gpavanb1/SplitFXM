from .error import SFXM
from .schemes import stencil_sizes
cimport numpy as np

# Function for handling callable functions
# Enum is not used as it is not supported by Cython
cdef np.ndarray derivative_callable(
    F,  # Function pointer type for callable functions
    list cell_sub,  # List of Cell, the stencil points
    int scheme,  # FDSchemes.value: the finite difference scheme to use
    int stencil_size,  # Size of the stencil
    int order=1,  # Order of the derivative (1 for first, 2 for second)
):
    """
    Calculate the derivative using a callable function.

    Parameters
    ----------
    F : callable
        A function that takes the values at the stencil points.
    cell_sub : list of Cell
        The stencil, a list of cells around the point of interest.
    scheme : FDSchemes.value
        The finite difference scheme to use (1 or 2).
    stencil_size : int
        The size of the stencil.
    order : int, optional
        The order of the derivative (1 for first derivative, 2 for second derivative). Default is 1.

    Returns
    -------
    np.ndarray
        The computed derivative based on the selected scheme and order.

    Raises
    ------
    SFXM
        If an improper stencil size is provided or an unsupported scheme is used.
    """
    cdef np.ndarray Fl, Fr, Dl, Dr, Dw, De
    cdef double dx, dx_w, dx_e

    if len(cell_sub) != stencil_size:
        raise SFXM(f"Improper stencil size. Expected {stencil_size}, got {len(cell_sub)}")

    if order == 1:
        if scheme == 1:  # FDSchemes.CENTRAL
            Fl = F(cell_sub[0].values())
            Fr = F(cell_sub[2].values())
            dx = cell_sub[2].x() - cell_sub[0].x()
            return (Fr - Fl) / dx
        elif scheme == 2:
            return (-F(cell_sub[0].values()) - F(cell_sub[1].values()) + 
                    F(cell_sub[2].values()) + F(cell_sub[3].values())) / (4 * (cell_sub[1].x() - cell_sub[0].x()))
        else:
            raise SFXM("Unsupported scheme")

    elif order == 2:
        if scheme == 1:  # FDSchemes.CENTRAL
            Dl = F(cell_sub[0].values())
            Dr = F(cell_sub[1].values())
            dx_w = cell_sub[1].x() - cell_sub[0].x()
            Dw = (Dr - Dl) / dx_w

            Dl = F(cell_sub[1].values())
            Dr = F(cell_sub[2].values())
            dx_e = cell_sub[2].x() - cell_sub[1].x()
            De = (Dr - Dl) / dx_e

            return (De - Dw) / ((dx_w + dx_e) / 2)
        elif scheme == 2:
            return (F(cell_sub[0].values()) - F(cell_sub[1].values()) - 
                    F(cell_sub[2].values()) + F(cell_sub[3].values())) / (2 * (cell_sub[1].x() - cell_sub[0].x())**2)
        else:
            raise SFXM("Unsupported scheme")
    else:
        raise SFXM("Unsupported order")

# Function for handling precomputed values
cdef double derivative_values(
    np.ndarray values,  # Precomputed values of the function at the stencil points
    list cell_sub,  # List of Cell, the stencil points
    int scheme,  # FDSchemes: the finite difference scheme to use
    int stencil_size, # Size of the stencil
    int order=1  # Order of the derivative (1 for first, 2 for second)
):
    """
    Calculate the derivative using precomputed values.

    Parameters
    ----------
    values : np.ndarray
        Precomputed values of the function at the stencil points.
    cell_sub : list of Cell
        The stencil, a list of cells around the point of interest.
    scheme : FDSchemes
        The finite difference scheme to use (CENTRAL or RIGHT_BIAS).
    order : int, optional
        The order of the derivative (1 for first derivative, 2 for second derivative). Default is 1.

    Returns
    -------
    np.ndarray
        The computed derivative based on the selected scheme and order.

    Raises
    ------
    SFXM
        If an improper stencil size is provided or an unsupported scheme is used.
    """
    cdef double dx

    if len(cell_sub) != stencil_size:
        raise SFXM(f"Improper stencil size. Expected {stencil_size}, got {len(cell_sub)}")

    if order == 1:
        if scheme == 1:  # FDSchemes.CENTRAL
            dx = cell_sub[2].x() - cell_sub[0].x()
            return (values[2] - values[0]) / dx
        elif scheme == 2:  # FDSchemes.RIGHT_BIAS
            dx = cell_sub[1].x() - cell_sub[0].x()
            return (-values[0] - values[1] + values[2] + values[3]) / (4 * dx)
        else:
            raise SFXM("Unsupported scheme")

    elif order == 2:
        if scheme == 1:  # FDSchemes.CENTRAL
            dx_w = cell_sub[1].x() - cell_sub[0].x()
            Dw = (values[1] - values[0]) / dx_w

            dx_e = cell_sub[2].x() - cell_sub[1].x()
            De = (values[2] - values[1]) / dx_e

            return (De - Dw) / ((dx_w + dx_e) / 2)
        elif scheme == 2:  # FDSchemes.RIGHT_BIAS
            dx = cell_sub[1].x() - cell_sub[0].x()
            return (values[0] - values[1] - values[2] + values[3]) / (2 * dx**2)
        else:
            raise SFXM("Unsupported scheme")
    else:
        raise SFXM("Unsupported scheme or order")

# First derivative functions
def Dx(F, cell_sub, scheme):
    """
    Calculate the first derivative using a given function and stencil.

    Parameters
    ----------
    F : function
        A function that takes the values at the stencil points.
    cell_sub : list of Cell
        The stencil, a list of cells around the point of interest.
    scheme : FDSchemes
        The finite difference scheme to use (CENTRAL or RIGHT_BIAS).

    Returns
    -------
    numpy.ndarray
        The computed first derivative based on the selected scheme.
    """
    return derivative_callable(F, cell_sub, scheme.value, stencil_sizes[scheme], order=1)

def dx(values, cell_sub, scheme):
    """
    Calculate the first derivative using precomputed function values at grid points.

    Parameters
    ----------
    values : np.ndarray
        Precomputed values of the function at the stencil points.
    cell_sub : list of Cell
        The stencil, a list of cells around the point of interest.
    scheme : FDSchemes
        The finite difference scheme to use (CENTRAL or RIGHT_BIAS).

    Returns
    -------
    numpy.ndarray
        The computed first derivative based on the selected scheme.
    """
    return derivative_values(values, cell_sub, scheme.value, stencil_sizes[scheme], order=1)

# Second derivative functions
def D2x(D, cell_sub, scheme):
    """
    Calculate the second derivative using a given function and stencil.

    Parameters
    ----------
    D : function
        A function that takes the values at the stencil points.
    cell_sub : list of Cell
        The stencil, a list of cells around the point of interest.
    scheme : FDSchemes
        The finite difference scheme to use (CENTRAL or RIGHT_BIAS).

    Returns
    -------
    numpy.ndarray
        The computed second derivative based on the selected scheme.
    """
    return derivative_callable(D, cell_sub, scheme.value, stencil_sizes[scheme], order=2)

def d2x(values, cell_sub, scheme):
    """
    Calculate the second derivative using precomputed function values at grid points.

    Parameters
    ----------
    values : np.ndarray
        Precomputed values of the function at the stencil points.
    cell_sub : list of Cell
        The stencil, a list of cells around the point of interest.
    scheme : FDSchemes
        The finite difference scheme to use (CENTRAL or RIGHT_BIAS).

    Returns
    -------
    numpy.ndarray
        The computed second derivative based on the selected scheme.
    """
    return derivative_values(values, cell_sub, scheme.value, stencil_sizes[scheme], order=2)
