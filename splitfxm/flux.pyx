import numpy as np
cimport numpy as cnp
from .error import SFXM
from .schemes import stencil_sizes, FVSchemes


def fluxes(F, cell_sub, scheme, dFdU=None, limiter=None):
    """
    Calculate the fluxes for a given stencil and numerical scheme with limiters.

    This function calculates the west and east fluxes for a given stencil
    of cells, using the specified finite volume scheme. Optionally, a slope 
    limiter can be applied to reduce non-physical oscillations in the fluxes.

    Parameters
    ----------
    F : function
        The flux function, which defines the relationship between the values
        of the conserved quantities and their fluxes.
    cell_sub : list of Cell
        The stencil of cells over which the fluxes are calculated.
    scheme : FVSchemes
        The finite volume scheme to use for flux calculations.
        Available options include:

        - LAX_FRIEDRICHS: Lax-Friedrichs scheme
        - UPWIND: Upwind differencing scheme
        - CENTRAL: Central differencing scheme

    dFdU : function, optional
        The Jacobian of the flux function, used for schemes that require
        knowledge of wave propagation direction (e.g., upwind schemes).

    Returns
    -------
    tuple of numpy.ndarray
        The west (Fw) and east (Fe) fluxes computed using the given scheme.

    Raises
    ------
    SFXM
        If the stencil size is not appropriate for the chosen scheme.

    Notes
    -----
    Each scheme has its own method of calculating fluxes:

    - LAX_FRIEDRICHS: Uses a central average with a diffusive term.
    - UPWIND: Chooses fluxes based on the direction of the flow.
    - CENTRAL: Takes a simple average of fluxes at the cell interfaces.
    """

    if len(cell_sub) != stencil_sizes.get(scheme):
        raise SFXM(
            f"Improper stencil size. Expected {stencil_sizes.get(scheme)}, got {len(cell_sub)}")

    if scheme == FVSchemes.LAX_FRIEDRICHS:
        return lax_friedrichs(F, cell_sub, dFdU)

    elif scheme == FVSchemes.UPWIND:
        return upwind(F, cell_sub, dFdU)

    elif scheme == FVSchemes.CENTRAL:
        return central(F, cell_sub)


cdef tuple lax_friedrichs(F, list cell_sub, dFdU):
    """Calculate fluxes using the Lax-Friedrichs scheme.

    Parameters
    ----------
    F : function
        The flux function.
    cell_sub : list of Cell
        The stencil of cells.
    dFdU : function
        The Jacobian of the flux function.

    Returns
    -------
    tuple of np.ndarray
        The west (Fw) and east (Fe) fluxes.
    """
    cdef cnp.ndarray ul, uc, ur
    cdef cnp.ndarray Fl, Fr
    cdef cnp.ndarray u_diff
    cdef double sigma

    # Get values from the cells
    ul = cell_sub[0].values()
    uc = cell_sub[1].values()
    ur = cell_sub[2].values()

    # Lax-Friedrichs requires spectral radius, so dFdU is needed
    if dFdU is None:
        raise SFXM("dFdU is required for determining spectral radius")

    Fl = F(ul)
    Fr = F(uc)
    u_diff = uc - ul
    sigma = np.max(np.abs(np.linalg.eigvals(dFdU(uc))))  # Spectral radius
    Fw = 0.5 * (Fl + Fr) - 0.5 * sigma * u_diff

    Fl = F(uc)
    Fr = F(ur)
    u_diff = ur - uc
    Fe = 0.5 * (Fl + Fr) - 0.5 * sigma * u_diff

    return Fw, Fe


cdef tuple upwind(F, list cell_sub, dFdU):
    """Calculate fluxes using the Upwind scheme.

    Parameters
    ----------
    F : function
        The flux function.
    cell_sub : list of Cell
        The stencil of cells.
    dFdU : function
        The Jacobian of the flux function.

    Returns
    -------
    tuple of np.ndarray
        The west (Fw) and east (Fe) fluxes.
    """
    cdef cnp.ndarray ul, uc, ur, A, eigvals, R, R_inv, wl, wc, wr
    cdef cnp.ndarray Fl, Fc, Fr, Fw_char, Fe_char, Fw, Fe
    cdef int i
    cdef double eig

    ul = cell_sub[0].values()
    uc = cell_sub[1].values()
    ur = cell_sub[2].values()

    if dFdU is None:
        raise SFXM(
            "dFdU is required for determining upwind direction.")

    # Compute the flux Jacobian matrix at the central state (uc)
    A = dFdU(uc)

    # Eigenvalue decomposition of the Jacobian matrix A
    eigvals, R = np.linalg.eig(A)
    R_inv = np.linalg.inv(R)

    # Transform the conservative variables (ul, uc, ur) into characteristic variables
    wl = R_inv @ ul  # Characteristic variables at the left state
    wc = R_inv @ uc  # Characteristic variables at the central state
    wr = R_inv @ ur  # Characteristic variables at the right state

    # Evaluate the functions at the characteristic variables
    Fl = F(wl)
    Fc = F(wc)
    Fr = F(wr)

    # Apply upwind scheme in the characteristic space
    # Flux at the west (left) side in characteristic space
    Fw_char = np.zeros_like(uc)
    # Flux at the east (right) side in characteristic space
    Fe_char = np.zeros_like(uc)

    for i, eig in enumerate(eigvals):
        if eig > 0:
            # Positive eigenvalue: wave moves to the right, use left (upwind) state for Fw
            Fw_char[i] = Fl[i]
            Fe_char[i] = Fc[i]
        else:
            # Negative eigenvalue: wave moves to the left, use right (upwind) state for Fe
            Fw_char[i] = Fc[i]
            Fe_char[i] = Fr[i]

    # Convert the characteristic fluxes back to the original space
    Fw = R @ Fw_char  # West (left) flux in original space
    Fe = R @ Fe_char  # East (right) flux in original space

    return Fw, Fe


cdef tuple central(F, list cell_sub):
    """Calculate fluxes using the Central scheme.

    Parameters
    ----------
    F : function
        The flux function.
    cell_sub : list of Cell
        The stencil of cells.

    Returns
    -------
    tuple of np.ndarray
        The west (Fw) and east (Fe) fluxes.
    """
    cdef cnp.ndarray ul, uc, ur, Fw, Fe

    ul = cell_sub[0].values()
    uc = cell_sub[1].values()
    ur = cell_sub[2].values()

    # Central differencing scheme
    Fw = 0.5 * (F(ul) + F(uc))
    Fe = 0.5 * (F(uc) + F(ur))

    return Fw, Fe


cpdef tuple diffusion_fluxes(D, list cell_sub, scheme):
    """
    Calculate the diffusion fluxes of a given stencil.

    Parameters
    ----------
    D : function
        The diffusion function.
    cell_sub : list of Cell
        The stencil.
    scheme: FVSchemes
        The finite-volume scheme to use. Defaults to central scheme for now.

    Returns
    -------
    tuple of numpy.ndarray
        The west and east diffusion fluxes.
    """
    cdef cnp.ndarray ul, uc, ur, Dl, Dc, Dr, Dw, De
    cdef double dxw, dxe

    if len(cell_sub) != stencil_sizes.get(scheme):
        raise SFXM(
            f"Improper stencil size. Expected {stencil_sizes.get(scheme)}, got {len(cell_sub)}")

    # Only central scheme for diffusion fluxes
    # West Flux
    ul = cell_sub[0].values()
    uc = cell_sub[1].values()
    Dl = D(ul)
    Dr = D(uc)
    dxw = 0.5 * (cell_sub[1].x() - cell_sub[0].x())
    Dw = (Dr - Dl) / dxw

    # East Flux
    uc = cell_sub[1].values()
    ur = cell_sub[2].values()
    Dl = D(uc)
    Dr = D(ur)
    dxe = 0.5 * (cell_sub[2].x() - cell_sub[1].x())
    De = (Dr - Dl) / dxe

    return Dw, De
