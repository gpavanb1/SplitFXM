from enum import Enum
from .error import SFXM

# TODO: Add more schemes
FDSchemes = Enum("FDSchemes", "CENTRAL")
Directions = Enum("Directions", "WEST EAST")


def Dx(F, cell_sub, scheme):
    """
    Calculate the first derivative of a given stencil.

    Parameters
    ----------
    F : function
        The function.
    cell_sub : list of Cell
        The stencil.
    scheme : Schemes
        The scheme to use.

    Returns
    -------
    numpy.ndarray
        The first derivative.
    """

    if scheme == FDSchemes.CENTRAL:
        if len(cell_sub) != 3:
            raise SFXM("Improper stencil size for central difference scheme")

        ul = cell_sub[0].values()
        ur = cell_sub[2].values()
        Fl = F(ul)
        Fr = F(ur)
        dx = cell_sub[2].x() - cell_sub[0].x()

        return (Fr - Fl) / dx
    else:
        raise SFXM("Unsupported scheme")


def D2x(D, cell_sub, scheme):
    """
    Calculate the second derivative of a given stencil.

    Parameters
    ----------
    D : function
        The function.
    cell_sub : list of Cell
        The stencil.

    Returns
    -------
    numpy.ndarray
        The second derivative.
    """

    # Only central scheme
    if scheme == FDSchemes.CENTRAL:
        if len(cell_sub) != 3:
            raise SFXM("Improper stencil size for central difference scheme")

        # West derivative
        ul = cell_sub[0].values()
        uc = cell_sub[1].values()
        Dl = D(ul)
        Dr = D(uc)
        dxw = cell_sub[1].x() - cell_sub[0].x()
        Dw = (Dr - Dl) / dxw

        # East derivative
        uc = cell_sub[1].values()
        ur = cell_sub[2].values()
        Dl = D(uc)
        Dr = D(ur)
        dxe = cell_sub[2].x() - cell_sub[1].x()
        De = (Dr - Dl) / dxe

        dx = 0.5 * (dxw + dxe)
        return (De - Dw) / dx
    else:
        raise SFXM("Unsupported scheme")


def dx(values, cell_sub, scheme):
    """
    Calculate the first derivative of a given stencil.

    Parameters
    ----------
    values : values of a function at the grid points
        The function.
    cell_sub : list of Cell
        The stencil.
    scheme : Schemes
        The scheme to use.

    Returns
    -------
    numpy.ndarray
        The first derivative.
    """

    if scheme == FDSchemes.CENTRAL:
        if len(cell_sub) != 3:
            raise SFXM("Improper stencil size for central difference scheme")

        Fl = values[0]
        Fr = values[2]
        dx = cell_sub[2].x() - cell_sub[0].x()

        return (Fr - Fl) / dx
    else:
        raise SFXM("Unsupported scheme")


def d2x(values, cell_sub, scheme):
    """
    Calculate the second derivative of a given stencil.

    Parameters
    ----------
    values : values of a function at the grid points
        The function.
    cell_sub : list of Cell
        The stencil.

    Returns
    -------
    numpy.ndarray
        The second derivative.
    """

    # Only central scheme
    if scheme == FDSchemes.CENTRAL:
        if len(cell_sub) != 3:
            raise SFXM("Improper stencil size for central difference scheme")

        # West derivative
        Dl = values[0]
        Dr = values[1]
        dxw = cell_sub[1].x() - cell_sub[0].x()
        Dw = (Dr - Dl) / dxw

        # East derivative
        Dl = values[1]
        Dr = values[2]
        dxe = cell_sub[2].x() - cell_sub[1].x()
        De = (Dr - Dl) / dxe

        dx = 0.5 * (dxw + dxe)
        return (De - Dw) / dx
    else:
        raise SFXM("Unsupported scheme")
