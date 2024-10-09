import cython
from cython.parallel import prange
import numpy as np
cimport numpy as cnp
from .error import SFXM
from .schemes import stencil_sizes, FVSchemes
from cpython.array cimport array, clone


# Helper to initialize memoryviews

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef allocate_double_array(int n):
    cdef array arr, template = array('d')
    arr = clone(template, n, zero=False)
    return arr


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] matvec(double[:, :] A, double[:] x):
    cdef int i, j, m, n
    cdef double sum

    m = A.shape[0]  # Number of rows in A
    n = A.shape[1]  # Number of columns in A
    cdef double[:] y = allocate_double_array(m)

    # Use prange for parallel computation
    for i in range(m):
        sum = 0.0  # Local sum variable
        for j in range(n):  # Assuming A is n x n
            sum += A[i, j] * x[j]  # Access elements using memoryview
        y[i] = sum  # Assign the computed sum to the result array

    return y


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
        # Lax-Friedrichs requires spectral radius, so dFdU is needed
        if dFdU is None:
            raise SFXM("dFdU is required for determining spectral radius")

        # Determine spectral radius
        uc = cell_sub[1].values()
        sigma = np.max(np.abs(np.linalg.eigvals(dFdU(uc))))

        # Calculate fluxes
        F_values = memoryview(np.array([F(cell.values()) for cell in cell_sub], dtype=np.double))
        u_values = memoryview(np.array([cell.values() for cell in cell_sub], dtype=np.double))
        Fw, Fe = lax_friedrichs(F_values, u_values, sigma)
        return np.asarray(Fw), np.asarray(Fe)

    elif scheme == FVSchemes.UPWIND:
        if dFdU is None:
            raise SFXM(
            "dFdU is required for determining upwind direction.")

        F_values = memoryview(np.array([F(cell.values()) for cell in cell_sub], dtype=np.double))
        u_values = memoryview(np.array([cell.values() for cell in cell_sub], dtype=np.double))
        eigvals, R = np.linalg.eig(dFdU(cell_sub[1].values()))
        eigvals, R = memoryview(eigvals), memoryview(R)
        R_inv = memoryview(np.linalg.inv(R))
        Fw, Fe = upwind(F_values, u_values, R, R_inv, eigvals)
        return np.asarray(Fw), np.asarray(Fe)

    elif scheme == FVSchemes.CENTRAL:
        F_values = memoryview(np.array([F(cell.values()) for cell in cell_sub], dtype=np.double))
        u_values = memoryview(np.array([cell.values() for cell in cell_sub], dtype=np.double))
        Fw, Fe = central(F_values, u_values)
        return np.asarray(Fw), np.asarray(Fe)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple lax_friedrichs(
    double[:, :]F, 
    double[:, :]u_values, 
    double sigma
):

    cdef double[:] ul, uc, ur
    cdef double[:] Fl, Fr
    cdef int i, n

    # Get shape
    n = F.shape[1]

    # Get values from the cells
    ul = u_values[0, :]
    uc = u_values[1, :]
    ur = u_values[2, :]

    Fl = F[0, :]
    Fr = F[1, :]

    cdef double[:] Fw = allocate_double_array(n)
    cdef double[:] Fe = allocate_double_array(n)
    cdef double[:] u_diff = allocate_double_array(n)

    for i in prange(n, nogil=True):
        u_diff[i] = uc[i] - ul[i]

    for i in prange(n, nogil=True):
        Fw[i] = 0.5 * (Fl[i] + Fr[i]) - 0.5 * sigma * u_diff[i]

    Fl = F[1, :]
    Fr = F[2, :]

    for i in prange(n, nogil=True):
        u_diff[i] = ur[i] - uc[i]

    for i in prange(n, nogil=True):
        Fe[i] = 0.5 * (Fl[i] + Fr[i]) - 0.5 * sigma * u_diff[i]

    return Fw, Fe

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple upwind(
    double[:, :]F, 
    double[:, :]u,
    double[:, :] R,
    double[:, :] R_inv,
    double[:] eigvals
):

    cdef double[:] ul, uc, ur
    cdef double[:] wl, wc, wr
    cdef double[:] Fl, Fc, Fr
    cdef double[:] Fw, Fe
    cdef int i, n

    # Get shape
    n = F.shape[1]

    ul = u[0, :]
    uc = u[1, :]
    ur = u[2, :]

    # Transform the conservative variables (ul, uc, ur) into characteristic variables
    wl = matvec(R_inv, ul)  # Characteristic variables at the left state
    wc = matvec(R_inv, uc)  # Characteristic variables at the central state
    wr = matvec(R_inv, ur)  # Characteristic variables at the right state

    # Evaluate the functions at the characteristic variables
    Fl = F[0, :]
    Fc = F[1, :]
    Fr = F[2, :]

    # Apply upwind scheme in the characteristic space
    # Flux at the west (left) side in characteristic space
    cdef double[:] Fw_char = allocate_double_array(n)
    # Flux at the east (right) side in characteristic space
    cdef double[:] Fe_char = allocate_double_array(n)

    for i in range(n):
        eig = eigvals[i]
        if eig > 0:  # Positive eigenvalue: wave moves to the right, use left (upwind) state for Fw
            Fw_char[i] = Fl[i]
            Fe_char[i] = Fc[i]
        else:  # Negative eigenvalue: wave moves to the left, use right (upwind) state for Fe
            Fw_char[i] = Fc[i]
            Fe_char[i] = Fr[i]

    # Convert the characteristic fluxes back to the original space
    Fw = matvec(R, Fw_char)  # West (left) flux in original space
    Fe = matvec(R, Fe_char)  # East (right) flux in original space

    return Fw, Fe

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple central(
    double[:, :] F, 
    double[:, :] u
):
    
    cdef double[:] ul, uc, ur
    cdef int i, n

    # Get shape
    n = F.shape[1]

    ul = u[0, :]
    uc = u[1, :]
    ur = u[2, :]

    Fl = F[0, :]
    Fc = F[1, :]
    Fr = F[2, :]

    # Allocate memory for the fluxes
    cdef double[:] Fw = allocate_double_array(n)
    cdef double[:] Fe = allocate_double_array(n)

    # Central differencing scheme
    for i in prange(n, nogil=True):
        Fw[i] = 0.5 * (Fl[i] + Fc[i])
        Fe[i] = 0.5 * (Fc[i] + Fr[i])

    return Fw, Fe


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple diff_central(
    double[:, :] D,
    double[:, :] u,
    double[:] cell_sub_x
):
    cdef double[:] ul, uc, ur
    cdef double[:] Dl, Dc, Dr
    cdef double dxw, dxe
    cdef int i, n

    # Get shape
    n = D.shape[1]

    # Allocate memory for the fluxes
    cdef double[:] Dw = allocate_double_array(n)
    cdef double[:] De = allocate_double_array(n)

    # Only central scheme for diffusion fluxes
    # West Flux
    ul = u[0, :]
    uc = u[1, :]
    Dl = D[0, :]
    Dr = D[1, :]
    dxw = 0.5 * (cell_sub_x[1] - cell_sub_x[0])
    for i in prange(n, nogil=True):
        Dw[i] = (Dr[i] - Dl[i]) / dxw

    # East Flux
    uc = u[1, :]
    ur = u[2, :]
    Dl = D[1, :]
    Dr = D[2, :]
    dxe = 0.5 * (cell_sub_x[2] - cell_sub_x[1])
    for i in prange(n, nogil=True):
        De[i] = (Dr[i] - Dl[i]) / dxe

    return Dw, De


def diffusion_fluxes(D, list cell_sub, scheme):
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
    if len(cell_sub) != stencil_sizes.get(scheme):
        raise SFXM(
            f"Improper stencil size. Expected {stencil_sizes.get(scheme)}, got {len(cell_sub)}")
    D_values = memoryview(np.array([D(cell.values()) for cell in cell_sub], dtype=np.double))
    u_values = memoryview(np.array([cell.values() for cell in cell_sub], dtype=np.double))
    cell_sub_x = memoryview(np.array([cell.x() for cell in cell_sub]))
    Dw, De = diff_central(D_values, u_values, cell_sub_x)
    return np.asarray(Dw), np.asarray(De)
