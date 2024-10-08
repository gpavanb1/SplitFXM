import cython
import numpy as np
from .error import SFXM
from .schemes import stencil_sizes
cimport numpy as np
from libc.math cimport pow

# Function for handling callable functions
# Enum is not used as it is not supported by Cython
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray derivative_callable(
    double[:, :] F,  # Function pointer type for callable functions
    double[:] cell_sub_x,  # List of Cell, the stencil points
    int scheme,  # FDSchemes.value: the finite difference scheme to use
    int stencil_size,  # Size of the stencil
    int order=1,  # Order of the derivative (1 for first, 2 for second)
):
    """
    Calculate the derivative using a callable function.

    Parameters
    ----------
    F_values : double[:, :]
        Precomputed function values at the stencil points (evaluated at corresponding points).
    cell_sub_x : double[:]
        The x-coordinates of the stencil points.
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
    cdef double[:] Fll, Fl, Fc, Fr, Fw, Fe, Fans
    cdef double dx, dx_w, dx_e
    cdef int i, n

    # Initialize memoryviews
    n = F.shape[1]
    Fw = memoryview(np.zeros(n, dtype=np.double))
    Fe = memoryview(np.zeros(n, dtype=np.double))
    Fans = memoryview(np.zeros(n, dtype=np.double))

    if len(cell_sub_x) != stencil_size:
        raise SFXM(f"Improper stencil size. Expected {stencil_size}, got {len(cell_sub_x)}")

    if order == 1:
        if scheme == 1:  # FDSchemes.CENTRAL
            Fl = F[0, :]
            Fr = F[2, :]
            dx = cell_sub_x[2] - cell_sub_x[0]

            for i in range(n):
                Fans[i] = (Fr[i] - Fl[i]) / dx
            return np.asarray(Fans)
        elif scheme == 2:
            Fll = F[0, :]
            Fl = F[1, :]
            Fc = F[2, :]
            Fr = F[3, :]

            for i in range(n):
                Fans[i] = (-Fll[i] - Fl[i] + Fc[i] + Fr[i]) / (4 * (cell_sub_x[1] - cell_sub_x[0]))
            return np.asarray(Fans)
        else:
            raise SFXM("Unsupported scheme")

    elif order == 2:
        if scheme == 1:  # FDSchemes.CENTRAL
            Fl = F[0, :]
            Fr = F[1, :]
            dx_w = cell_sub_x[1] - cell_sub_x[0]

            for i in range(n):
                Fw[i] = (Fr[i] - Fl[i]) / dx_w

            Fl = F[1, :]
            Fr = F[2, :]
            dx_e = cell_sub_x[2] - cell_sub_x[1]
            
            for i in range(n):
                Fe[i] = (Fr[i] - Fl[i]) / dx_e

            for i in range(n):
                Fans[i] = (Fe[i] - Fw[i]) / ((dx_w + dx_e) / 2)
            return np.asarray(Fans)
        elif scheme == 2:
            Fll = F[0, :]
            Fl = F[1, :]
            Fc = F[2, :]
            Fr = F[3, :]

            for i in range(n):
                Fans[i] = (Fll[i] - Fl[i] - Fc[i] + Fr[i]) / (2 * (cell_sub_x[1] - cell_sub_x[0])**2)
            return np.asarray(Fans)
        else:
            raise SFXM("Unsupported scheme")
    else:
        raise SFXM("Unsupported order")

# Function for handling precomputed values
cdef double derivative_values(
    double[:] values,  # Memoryview for precomputed values at stencil points
    double[:] cell_sub_x,  # Memoryview for cell positions (x-coordinates)
    int scheme,  # FDSchemes: the finite difference scheme to use
    int stencil_size,  # Size of the stencil
    int order=1  # Order of the derivative (1 for first, 2 for second)
):
    """
    Calculate the derivative using precomputed values and cell positions.
    
    Parameters
    ----------
    values : double[:]
        Precomputed values of the function at the stencil points.
    cell_sub_x : double[:]
        The x-coordinates of the stencil points.
    scheme : int
        The finite difference scheme to use (CENTRAL or RIGHT_BIAS).
    order : int, optional
        The order of the derivative (1 for first derivative, 2 for second derivative). Default is 1.
        
    Returns
    -------
    double
        The computed derivative based on the selected scheme and order.
    
    Raises
    ------
    ValueError
        If an improper stencil size is provided or an unsupported scheme is used.
    """
    cdef double dx, Dw, De
    
    if stencil_size != values.shape[0] or stencil_size != cell_sub_x.shape[0]:
        raise SFXM(f"Improper stencil size. Expected {stencil_size}, got {values.shape[0]}")

    if order == 1:
        if scheme == 1:  # FDSchemes.CENTRAL
            dx = cell_sub_x[2] - cell_sub_x[0]
            return (values[2] - values[0]) / dx
        elif scheme == 2:  # FDSchemes.RIGHT_BIAS
            dx = cell_sub_x[1] - cell_sub_x[0]
            return (-values[0] - values[1] + values[2] + values[3]) / (4 * dx)
        else:
            raise SFXM("Unsupported scheme")

    elif order == 2:
        if scheme == 1:  # FDSchemes.CENTRAL
            dx_w = cell_sub_x[1] - cell_sub_x[0]
            Dw = (values[1] - values[0]) / dx_w

            dx_e = cell_sub_x[2] - cell_sub_x[1]
            De = (values[2] - values[1]) / dx_e

            return (De - Dw) / ((dx_w + dx_e) / 2)
        elif scheme == 2:  # FDSchemes.RIGHT_BIAS
            dx = cell_sub_x[1] - cell_sub_x[0]
            return (values[0] - values[1] - values[2] + values[3]) / (2 * pow(dx, 2))
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
    cell_sub_x = memoryview(np.array([cell.x() for cell in cell_sub], dtype=np.double))
    F_values = memoryview(np.array([F(cell.values()) for cell in cell_sub], dtype=np.double))
    return derivative_callable(F_values, cell_sub_x, scheme.value, stencil_sizes[scheme], order=1)

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
    cell_sub_x = memoryview(np.array([cell.x() for cell in cell_sub], dtype=np.double))
    return derivative_values(values, cell_sub_x, scheme.value, stencil_sizes[scheme], order=1)

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
    cell_sub_x = memoryview(np.array([cell.x() for cell in cell_sub], dtype=np.double))
    D_values = memoryview(np.array([D(cell.values()) for cell in cell_sub], dtype=np.double))
    return derivative_callable(D_values, cell_sub_x, scheme.value, stencil_sizes[scheme], order=2)

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
    cell_sub_x = memoryview(np.array([cell.x() for cell in cell_sub], dtype=np.double))
    return derivative_values(values, cell_sub_x, scheme.value, stencil_sizes[scheme], order=2)
