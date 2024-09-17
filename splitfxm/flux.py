import numpy as np
from .error import SFXM
from .schemes import stencil_sizes, FVSchemes


def minmod(a, b):
    """
    Compute the minmod of two values, a and b.

    The minmod function is a slope limiter used in high-resolution schemes
    (like MUSCL) to prevent non-physical oscillations in numerical solutions
    of hyperbolic PDEs

    Parameters
    ----------
    a : float or numpy.ndarray
        First value or array of values.
    b : float or numpy.ndarray
        Second value or array of values.

    Returns
    -------
    float or numpy.ndarray
        The minmod value, which is either the smaller magnitude of a or b
        if they have the same sign, or zero otherwise.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # Compute the sign of a and b
    sign_a = np.sign(a)
    sign_b = np.sign(b)

    # Compute minmod element-wise
    minmod_result = np.where(sign_a == sign_b, sign_a *
                             np.minimum(np.abs(a), np.abs(b)), 0)

    return minmod_result


def fluxes(F, cell_sub, scheme):
    """
    Calculate the fluxes for a given stencil and numerical scheme.

    This function calculates the west and east fluxes for a given stencil
    of cells, using the specified finite volume scheme. The scheme determines
    how the fluxes are computed, such as using upwind differencing, central
    differencing, or more advanced schemes like QUICK, MUSCL, ENO, or WENO.

    Parameters
    ----------
    F : function
        The flux function, which defines the relationship between the values
        of the conserved quantities and their fluxes.
    cell_sub : list of Cell
        The stencil of cells over which the fluxes are calculated.
        Typically, this list contains 3 or more cells.
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

    ul = cell_sub[0].values()
    uc = cell_sub[1].values()
    ur = cell_sub[2].values()

    if scheme == FVSchemes.LAX_FRIEDRICHS:
        # Lax-Friedrichs scheme
        Fl = F(ul)
        Fr = F(uc)
        u_diff = uc - ul
        sigma = 0.1  # TODO: Evaluate spectral radius
        Fw = 0.5 * (Fl + Fr) - 0.5 * sigma * u_diff

        Fl = F(uc)
        Fr = F(ur)
        u_diff = ur - uc
        Fe = 0.5 * (Fl + Fr) - 0.5 * sigma * u_diff

    elif scheme == FVSchemes.UPWIND:
        # Upwind scheme
        if np.all(uc > 0):
            Fw = F(ul)
        else:
            Fw = F(uc)

        if np.all(ur > 0):
            Fe = F(uc)
        else:
            Fe = F(ur)

    elif scheme == FVSchemes.CENTRAL:
        # Central differencing scheme
        Fw = 0.5 * (F(ul) + F(uc))
        Fe = 0.5 * (F(uc) + F(ur))

    elif scheme == FVSchemes.LAX_WENDROFF:
        # Lax-Wendroff scheme
        Fl = F(ul)
        Fr = F(uc)
        Fw = Fl + 0.5 * (uc - ul) * (Fr - Fl)

        Fl = F(uc)
        Fr = F(ur)
        Fe = Fl + 0.5 * (ur - uc) * (Fr - Fl)

    elif scheme == FVSchemes.QUICK:
        # QUICK Scheme (3rd-order upwind)
        ul = cell_sub[0].values()
        um = cell_sub[1].values()
        ur = cell_sub[2].values()
        # Quadratic interpolation at the face
        Fw = (3/8) * F(ul) + (6/8) * F(um) - (1/8) * F(ur)

        ul = cell_sub[1].values()
        um = cell_sub[2].values()
        ur = cell_sub[3].values()
        Fe = (3/8) * F(ul) + (6/8) * F(um) - (1/8) * F(ur)

    elif scheme == FVSchemes.BQUICK:
        # BQUICK Scheme (Bounded QUICK)
        # This requires a limiter to avoid oscillations near discontinuities.
        ul = cell_sub[0].values()
        um = cell_sub[1].values()
        ur = cell_sub[2].values()

        Fw = np.clip((3/8) * F(ul) + (6/8) * F(um) -
                     (1/8) * F(ur), F(ul), F(ur))

        ul = cell_sub[1].values()
        um = cell_sub[2].values()
        ur = cell_sub[3].values()
        Fe = np.clip((3/8) * F(ul) + (6/8) * F(um) -
                     (1/8) * F(ur), F(ul), F(ur))

    elif scheme == FVSchemes.MUSCL:
        # MUSCL (Monotonic Upwind Scheme for Conservation Laws)
        ul = cell_sub[0].values()
        um = cell_sub[1].values()
        ur = cell_sub[2].values()

        slope = minmod(um - ul, ur - um)
        Fw = F(um - 0.5 * slope)
        Fe = F(um + 0.5 * slope)

    elif scheme == FVSchemes.ENO:
        # ENO (Essentially Non-Oscillatory Scheme)
        ul = cell_sub[0].values()
        um = cell_sub[1].values()
        ur = cell_sub[2].values()
        us = cell_sub[3].values()

        # ENO chooses the smoothest stencil, basic implementation for 2nd-order
        diff1 = um - ul
        diff2 = ur - um
        diff3 = us - ur

        # Check smoothest region based on divided differences
        if np.all(abs(diff2 - diff1) < abs(diff3 - diff2)):
            Fw = F(ul)
        else:
            Fw = F(um)

        Fe = F(ur)

    elif scheme == FVSchemes.WENO:
        # WENO (Weighted Essentially Non-Oscillatory Scheme)
        ul = cell_sub[0].values()
        um = cell_sub[1].values()
        ur = cell_sub[2].values()
        us = cell_sub[3].values()

        # WENO coefficients for a simple 5th-order WENO scheme
        beta0 = 13/12 * (ul - 2*um + ur)**2 + 1/4 * (ul - 4*um + 3*ur)**2
        beta1 = 13/12 * (um - 2*ur + us)**2 + 1/4 * (um - us)**2
        epsilon = 1e-6
        alpha0 = 1 / (epsilon + beta0)**2
        alpha1 = 1 / (epsilon + beta1)**2
        w0 = alpha0 / (alpha0 + alpha1)
        w1 = alpha1 / (alpha0 + alpha1)

        Fw = w0 * F(ul) + w1 * F(um)
        Fe = w0 * F(ur) + w1 * F(us)

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
