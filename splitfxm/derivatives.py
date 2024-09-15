from .error import SFXM
from .schemes import stencil_sizes, FDSchemes


def derivative(F, cell_sub, scheme, order=1, is_values=False):
    """
    General function to calculate the first or second derivative based on a given function or precomputed values.

    Parameters
    ----------
    F : function or list/numpy.ndarray
        If `is_values` is False, this is a function that takes the values at the stencil points.
        If `is_values` is True, this is the precomputed values of the function at the stencil points.
    cell_sub : list of Cell
        The stencil, a list of cells around the point of interest.
    scheme : FDSchemes
        The finite difference scheme to use (CENTRAL or RIGHT_BIAS).
    order : int, optional
        The order of the derivative (1 for first derivative, 2 for second derivative). Default is 1.
    is_values : bool, optional
        If True, `F` is treated as precomputed values of the function at the stencil points. If False, `F` is treated as a function. Default is False.

    Returns
    -------
    numpy.ndarray
        The computed derivative based on the selected scheme and order.

    Raises
    ------
    SFXM
        If an improper stencil size is provided or an unsupported scheme is used.
    """
    if len(cell_sub) != stencil_sizes.get(scheme):
        raise SFXM(
            f"Improper stencil size. Expected {stencil_sizes.get(scheme)}, got {len(cell_sub)}")

    if order == 1:
        if scheme == FDSchemes.CENTRAL:
            Fl = F[0] if is_values else F(
                cell_sub[0].values())
            Fr = F[2] if is_values else F(
                cell_sub[2].values())
            dx = cell_sub[2].x() - cell_sub[0].x()
            return (Fr - Fl) / dx
        elif scheme == FDSchemes.RIGHT_BIAS:
            F_vals = F if is_values else [
                F(cell_sub[i].values()) for i in range(4)]
            dx = cell_sub[1].x() - cell_sub[0].x()
            return (-F_vals[0] - F_vals[1] + F_vals[2] + F_vals[3]) / (4 * dx)

    elif order == 2:
        if scheme == FDSchemes.CENTRAL:
            Dl = F[0] if is_values else F(
                cell_sub[0].values())
            Dr = F[1] if is_values else F(
                cell_sub[1].values())
            dx_w = cell_sub[1].x() - cell_sub[0].x()
            Dw = (Dr - Dl) / dx_w

            Dl = F[1] if is_values else F(
                cell_sub[1].values())
            Dr = F[2] if is_values else F(
                cell_sub[2].values())
            dx_e = cell_sub[2].x() - cell_sub[1].x()
            De = (Dr - Dl) / dx_e

            return (De - Dw) / ((dx_w + dx_e) / 2)
        elif scheme == FDSchemes.RIGHT_BIAS:
            D_vals = F if is_values else [
                F(cell_sub[i].values()) for i in range(4)]
            dx = cell_sub[1].x() - cell_sub[0].x()
            return (D_vals[0] - D_vals[1] - D_vals[2] + D_vals[3]) / (2 * dx**2)

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
    return derivative(F, cell_sub, scheme, order=1)


def dx(values, cell_sub, scheme):
    """
    Calculate the first derivative using precomputed function values at grid points.

    Parameters
    ----------
    values : list or numpy.ndarray
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
    return derivative(values, cell_sub, scheme, order=1, is_values=True)

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
    return derivative(D, cell_sub, scheme, order=2)


def d2x(values, cell_sub, scheme):
    """
    Calculate the second derivative using precomputed function values at grid points.

    Parameters
    ----------
    values : list or numpy.ndarray
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
    return derivative(values, cell_sub, scheme, order=2, is_values=True)
