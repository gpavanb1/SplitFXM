import numpy as np
from .error import SFXM
from .schemes import stencil_sizes, FVSchemes
from .flux_limiters import psi


def smoothness_indicator(F, states):
    """
    Compute the smoothness indicator based on the states provided.

    Parameters:
    F : function
        The flux function.
    states : tuple
        A tuple of states to evaluate the smoothness.

    Returns:
    float
        The smoothness indicator value.
    """
    # Unpack states
    f0, f1, f2 = F(states[0]), F(states[1]), F(states[2])

    # Calculate the differences
    d1 = f1 - f0
    d2 = f2 - f1

    # Calculate the smoothness indicator (L2 norm of the differences)
    # Adding a small constant to avoid division by zero
    indicator = (d1**2 + d2**2) + 1e-6
    return indicator


def weno_weights(betas):
    """
    Compute WENO weights based on the smoothness indicators.

    Parameters:
    betas : list
        List of smoothness indicators for each stencil.

    Returns:
    np.ndarray
        Array of normalized weights.
    """
    epsilon = 1e-6  # Regularization term to avoid division by zero
    alpha = [1 / (beta + epsilon) for beta in betas]
    alpha_sum = sum(alpha)
    weights = [a / alpha_sum for a in alpha]

    return np.array(weights)


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
        - LAX_WENDROFF: Lax-Wendroff scheme
        - QUICK: Quadratic upwind interpolation
        - BQUICK: Bounded QUICK scheme
        - MUSCL: Monotonic Upwind Scheme for Conservation Laws
        - ENO: Essentially Non-Oscillatory scheme
        - WENO: Weighted Essentially Non-Oscillatory scheme

    dFdU : function, optional
        The Jacobian of the flux function, used for schemes that require
        knowledge of wave propagation direction (e.g., upwind schemes).
    limiter : FVLimiters, optional
        The slope limiter function to be applied. Limiters are used in
        higher-order schemes like MUSCL to prevent spurious oscillations.
        Available limiters include:

        - MINMOD: Reduces oscillations by limiting the slope to the minimum.
        - VAN_LEER: Smooth, differentiable limiter with good accuracy.
        - VAN_ALBADA: A smoother version of MINMOD, avoids excessive flattening.
        - SUPERBEE: A sharp limiter that maximizes the steepness of the slope.

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
    - LAX_WENDROFF: Uses a Taylor series expansion to improve accuracy.
    - QUICK: A 3rd-order upwind-biased scheme.
    - BQUICK: A bounded version of QUICK to prevent oscillations.
    - MUSCL: Uses a slope limiter to prevent non-physical oscillations.
    - ENO: Selects the smoothest stencil for flux calculation.
    - WENO: A higher-order extension of ENO using weighted averages.
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

    elif scheme == FVSchemes.LAX_WENDROFF:
        return lax_wendroff(F, cell_sub)

    elif scheme == FVSchemes.QUICK:
        return quick(F, cell_sub, dFdU)

    elif scheme == FVSchemes.BQUICK:
        return bquick(F, cell_sub, dFdU)

    # Ensure that limiter is specified for MUSCL
    elif scheme == FVSchemes.MUSCL:
        return muscl(F, cell_sub, dFdU, limiter)

    elif scheme == FVSchemes.ENO:
        return eno(F, cell_sub, dFdU)

    elif scheme == FVSchemes.WENO:
        return weno(F, cell_sub, dFdU)


def lax_friedrichs(F, cell_sub, dFdU):
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
    ul = cell_sub[0].values()
    uc = cell_sub[1].values()
    ur = cell_sub[2].values()

    # Lax-Friedrichs requires spectral radius, so dFdU is needed
    if dFdU is None:
        raise SFXM(
            f"dFdU is required for determining spectral radius")

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


def upwind(F, cell_sub, dFdU):
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


def central(F, cell_sub):
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
    ul = cell_sub[0].values()
    uc = cell_sub[1].values()
    ur = cell_sub[2].values()

    # Central differencing scheme
    Fw = 0.5 * (F(ul) + F(uc))
    Fe = 0.5 * (F(uc) + F(ur))

    return Fw, Fe


def lax_wendroff(F, cell_sub):
    """Calculate fluxes using the Lax-Wendroff scheme.

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
    ul = cell_sub[0].values()
    uc = cell_sub[1].values()
    ur = cell_sub[2].values()

    # Lax-Wendroff scheme
    Fl = F(ul)
    Fr = F(uc)
    Fw = Fl + 0.5 * (uc - ul) * (Fr - Fl)

    Fl = F(uc)
    Fr = F(ur)
    Fe = Fl + 0.5 * (ur - uc) * (Fr - Fl)

    return Fw, Fe


def quick(F, cell_sub, dFdU):
    """Calculate fluxes using the Quick scheme.

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
    uc = cell_sub[2].values()

    if dFdU is None:
        raise SFXM(
            "dFdU is required for determining upwind direction.")

    # Compute the flux Jacobian matrix at the central state (uc)
    A = dFdU(uc)

    # Eigenvalue decomposition of the Jacobian matrix A
    eigvals, R = np.linalg.eig(A)
    R_inv = np.linalg.inv(R)

    # Initialize characteristic fluxes
    Fw_char = np.zeros_like(uc)
    Fe_char = np.zeros_like(uc)

    # Compute the characteristic variables at the left and right states
    wl2 = R_inv @ cell_sub[0].values()
    wl = R_inv @ cell_sub[1].values()
    wc = R_inv @ cell_sub[2].values()
    wr = R_inv @ cell_sub[3].values()
    wr2 = R_inv @ cell_sub[4].values()

    # Evaluate the functions at the characteristic variables
    Fwl2, Fwl, Fwc, Fwr, Fwr2 = F(wl2), F(wl), F(wc), F(wr), F(wr2)

    # Loop over each characteristic field (based on the eigenvalues)
    for i, eig in enumerate(eigvals):
        # Positive eigenvalue, use left-biased stencil (upwind)
        if eig > 0:
            # Use 3rd-order interpolation with left-biased stencil
            Fw_char[i] = 3/8 * Fwl2[i] + 6/8 * Fwl[i] - 1/8 * Fwc[i]
            Fe_char[i] = -1/8 * Fwl[i] + 6/8 * Fwc[i] + 3/8 * Fwr[i]
        else:  # Negative eigenvalue, use right-biased stencil (downwind)
            Fw_char[i] = 3/8 * Fwc[i] + 6/8 * Fwr[i] - 1/8 * Fwr2[i]
            Fe_char[i] = -1/8 * Fwc[i] + 6/8 * Fwr[i] + 3/8 * Fwr2[i]

    # Convert fluxes back to physical space
    Fw = R @ Fw_char  # West (left) flux in physical space
    Fe = R @ Fe_char  # East (right) flux in physical space

    return Fw, Fe


def bquick(F, cell_sub, dFdU):
    """Calculate fluxes using the BQUICK scheme.

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
    uc = cell_sub[2].values()

    if dFdU is None:
        raise SFXM(
            "dFdU is required for determining upwind direction.")

    # Compute the flux Jacobian matrix at the central state (uc)
    A = dFdU(uc)

    # Eigenvalue decomposition of the Jacobian matrix A
    eigvals, R = np.linalg.eig(A)
    R_inv = np.linalg.inv(R)

    # Initialize characteristic fluxes
    Fw_char = np.zeros_like(uc)
    Fe_char = np.zeros_like(uc)

    # Compute the characteristic variables at the left and right states
    wl2 = R_inv @ cell_sub[0].values()
    wl = R_inv @ cell_sub[1].values()
    wc = R_inv @ cell_sub[2].values()
    wr = R_inv @ cell_sub[3].values()
    wr2 = R_inv @ cell_sub[4].values()

    # Evaluate the functions at the characteristic variables
    Fwl2, Fwl, Fwc, Fwr, Fwr2 = F(wl2), F(wl), F(wc), F(wr), F(wr2)

    # Loop over each characteristic field (based on the eigenvalues)
    for i, eig in enumerate(eigvals):
        # Positive eigenvalue, use left-biased stencil (upwind)
        if eig > 0:
            # Use 3rd-order interpolation with left-biased stencil
            Fw_char[i] = 3/8 * Fwl2[i] + 6/8 * Fwl[i] - 1/8 * Fwc[i]
            Fe_char[i] = -1/8 * Fwl[i] + 6/8 * Fwc[i] + 3/8 * Fwr[i]
        else:  # Negative eigenvalue, use right-biased stencil (downwind)
            Fw_char[i] = 3/8 * Fwc[i] + 6/8 * Fwr[i] - 1/8 * Fwr2[i]
            Fe_char[i] = -1/8 * Fwc[i] + 6/8 * Fwr[i] + 3/8 * Fwr2[i]

        # Apply bounds to avoid oscillations (based on min/max of neighboring states)
        Fw_char[i] = np.maximum(np.minimum(Fw_char[i], Fwc[i]), Fwl[i])
        Fe_char[i] = np.maximum(np.minimum(Fe_char[i], Fwr[i]), Fwc[i])

    # Convert fluxes back to physical space
    Fw = R @ Fw_char  # West (left) flux in physical space
    Fe = R @ Fe_char  # East (right) flux in physical space

    return Fw, Fe


def muscl(F, cell_sub, dFdU, limiter):
    """Calculate fluxes using the MUSCL scheme with slope limiters.

    Parameters
    ----------
    F : function
        The flux function.
    cell_sub : list of Cell
        The stencil of cells.
    dFdU : function
        The Jacobian of the flux function.
    limiter : FVLimiters
        The name of the limiter dictates the slope limiter function psi(r, limiter) to be used.

    Returns
    -------
    tuple of np.ndarray
        The west (Fw) and east (Fe) fluxes.
    """
    uc = cell_sub[2].values()  # Central cell values

    if dFdU is None:
        raise SFXM("dFdU is required for determining upwind direction.")

    if limiter is None:
        raise SFXM("Explicit limiter definition is required for this scheme")

    # Compute the flux Jacobian matrix at the central state (uc)
    A = dFdU(uc)

    # Eigenvalue decomposition of the Jacobian matrix A
    eigvals, R = np.linalg.eig(A)
    R_inv = np.linalg.inv(R)

    # Initialize face values
    w_west = np.zeros_like(uc)
    w_east = np.zeros_like(uc)

    # Compute the characteristic variables at the left, central, and right states
    wl2 = R_inv @ cell_sub[0].values()  # Left-left state
    wl = R_inv @ cell_sub[1].values()   # Left state
    wc = R_inv @ cell_sub[2].values()   # Central state
    wr = R_inv @ cell_sub[3].values()   # Right state
    wr2 = R_inv @ cell_sub[4].values()  # Right-right state

    # Loop over each characteristic field (based on the eigenvalues)
    for i, eig in enumerate(eigvals):
        if eig > 0:  # Positive eigenvalue (upwind to the left)
            # Use the left-most values on each face for flux determination
            # Use the left-biased slope ratios for flux determination
            delta_wl = wc[i] - wl[i]
            delta_wl2 = wl[i] - wl2[i]
            r_w = delta_wl2 / (delta_wl + 1e-6)  # Avoid division by zero
            slope_w = psi(r_w, limiter) * delta_wl
            # Extrapolated left state at the west interface
            w_west[i] = wl[i] + 0.5 * slope_w

            # Similarly on the east side
            delta_wc = wr[i] - wc[i]
            delta_wr = wr2[i] - wr[i]
            r_e = delta_wc / (delta_wr + 1e-6)  # Avoid division by zero
            slope_e = psi(r_e, limiter) * delta_wr
            # Extrapolated left state at the east interface
            w_east[i] = wc[i] + 0.5 * slope_e

        else:  # Negative eigenvalue (upwind to the right)
            # Use the right-most values on each face for flux determination
            # Use the right-biased slope ratios for flux determination
            delta_wl = wc[i] - wl[i]
            delta_wl2 = wl[i] - wl2[i]
            r_w = delta_wl / (delta_wl2 + 1e-6)  # Avoid division by zero
            slope_w = psi(r_w, limiter) * delta_wl2
            # Extrapolated right state at the west interface
            w_west[i] = wc[i] - 0.5 * slope_w

            # Similarly on the east side
            delta_wc = wr[i] - wc[i]
            delta_wr = wr2[i] - wr[i]
            r_e = delta_wr / (delta_wc + 1e-6)  # Avoid division by zero
            slope_e = psi(r_e, limiter) * delta_wc
            # Extrapolated right state at the east interface
            w_east[i] = wr[i] - 0.5 * slope_e

    # Compute fluxes at the extrapolated states (for characteristic variables)
    Fw_char = F(w_west)
    Fe_char = F(w_east)

    # Convert fluxes back to physical space
    Fw = R @ Fw_char  # West (left) flux in physical space
    Fe = R @ Fe_char  # East (right) flux in physical space

    return Fw, Fe


def eno(F, cell_sub, dFdU):
    """Calculate fluxes using the ENO scheme.

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
    uc = cell_sub[2].values()

    if dFdU is None:
        raise SFXM("dFdU is required for determining upwind direction.")

    # Compute the flux Jacobian matrix at the central state (uc)
    A = dFdU(uc)

    # Eigenvalue decomposition of the Jacobian matrix A
    eigvals, R = np.linalg.eig(A)
    R_inv = np.linalg.inv(R)

    # Initialize characteristic fluxes
    Fw_char = np.zeros_like(uc)
    Fe_char = np.zeros_like(uc)

    # Compute the characteristic variables at the stencils
    wl2 = R_inv @ cell_sub[0].values()
    wl = R_inv @ cell_sub[1].values()
    wc = R_inv @ cell_sub[2].values()
    wr = R_inv @ cell_sub[3].values()
    wr2 = R_inv @ cell_sub[4].values()

    # Evaluate the fluxes at the characteristic variables
    Fwl2, Fwl, Fwc, Fwr, Fwr2 = F(wl2), F(wl), F(wc), F(wr), F(wr2)

    # Compute smoothness indicators for each stencil
    smoothness = [
        (Fwl2 - 2 * Fwl + Fwc) ** 2,  # Smoothness for stencil 1
        (Fwl - 2 * Fwc + Fwr) ** 2,    # Smoothness for stencil 2
        (Fwc - 2 * Fwr + Fwr2) ** 2     # Smoothness for stencil 3
    ]

    # Determine the index of the minimum smoothness indicator
    min_smoothness_index = np.argmin(smoothness)

    # Loop over each characteristic field (based on the eigenvalues)
    for i, eig in enumerate(eigvals):
        if eig > 0:
            # Use left-biased stencil based on minimum smoothness index
            if min_smoothness_index == 0:
                Fw_char[i] = Fwl2[i]  # Stencil 1
            elif min_smoothness_index == 1:
                Fw_char[i] = Fwl[i]   # Stencil 2
            else:
                Fw_char[i] = Fwc[i]   # Stencil 3

            # Calculate the right flux for consistency
            Fe_char[i] = Fwc[i]

        elif eig < 0:
            # Use right-biased stencil based on minimum smoothness index
            if min_smoothness_index == 0:
                Fe_char[i] = Fwc[i]   # Stencil 1
            elif min_smoothness_index == 1:
                Fe_char[i] = Fwr[i]   # Stencil 2
            else:
                Fe_char[i] = Fwr2[i]  # Stencil 3

            # Calculate the left flux for consistency
            Fw_char[i] = Fwc[i]

    # Convert fluxes back to physical space
    Fw = R @ Fw_char  # West (left) flux in physical space
    Fe = R @ Fe_char  # East (right) flux in physical space

    return Fw, Fe


def weno(F, cell_sub, dFdU):
    """Calculate fluxes using the WENO scheme.

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
    uc = cell_sub[2].values()

    if dFdU is None:
        raise SFXM("dFdU is required for determining upwind direction.")

    # Compute the flux Jacobian matrix at the central state (uc)
    A = dFdU(uc)

    # Eigenvalue decomposition of the Jacobian matrix A
    eigvals, R = np.linalg.eig(A)
    R_inv = np.linalg.inv(R)

    # Initialize characteristic fluxes
    Fw_char = np.zeros_like(uc)
    Fe_char = np.zeros_like(uc)

    # Compute the characteristic variables at the stencils
    wl2 = R_inv @ cell_sub[0].values()
    wl = R_inv @ cell_sub[1].values()
    wc = R_inv @ cell_sub[2].values()
    wr = R_inv @ cell_sub[3].values()
    wr2 = R_inv @ cell_sub[4].values()

    # Evaluate the fluxes at the characteristic variables
    Fwl2, Fwl, Fwc, Fwr, Fwr2 = F(wl2), F(wl), F(wc), F(wr), F(wr2)

    # Compute smoothness indicators for each stencil
    smoothness = [
        (Fwl2 - 2 * Fwl + Fwc) ** 2,  # Smoothness for stencil 1
        (Fwl - 2 * Fwc + Fwr) ** 2,    # Smoothness for stencil 2
        (Fwc - 2 * Fwr + Fwr2) ** 2     # Smoothness for stencil 3
    ]

    # Calculate WENO weights
    weights = weno_weights(smoothness)

    # Loop over each characteristic field (based on the eigenvalues)
    for i, eig in enumerate(eigvals):
        # For positive eigenvalue: use left-biased stencil
        if eig > 0:
            Fw_char[i] = (weights[0][i] * Fwl2[i] +
                          weights[1][i] * Fwl[i] +
                          weights[2][i] * Fwc[i])
            # We might still want to calculate the right flux for consistency
            Fe_char[i] = (weights[0][i] * Fwc[i] +
                          weights[1][i] * Fwr[i] +
                          weights[2][i] * Fwr2[i])

        # For negative eigenvalue: use right-biased stencil
        else:
            Fe_char[i] = (weights[0][i] * Fwc[i] +
                          weights[1][i] * Fwr[i] +
                          weights[2][i] * Fwr2[i])
            # We might still want to calculate the left flux for consistency
            Fw_char[i] = (weights[0][i] * Fwl2[i] +
                          weights[1][i] * Fwl[i] +
                          weights[2][i] * Fwc[i])

    # Convert fluxes back to physical space
    Fw = R @ Fw_char  # West (left) flux in physical space
    Fe = R @ Fe_char  # East (right) flux in physical space

    return Fw, Fe


def diffusion_fluxes(D, cell_sub, scheme):
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
