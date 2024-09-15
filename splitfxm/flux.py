from .error import SFXM
from .schemes import stencil_sizes, FVSchemes


def fluxes(F, cell_sub, scheme):
    """
    Calculate the fluxes of a given stencil.

    Parameters
    ----------
    F : function
        The flux function.
    cell_sub : list of Cell
        The stencil.
    scheme : Schemes
        The scheme to use.

    Returns
    -------
    tuple of numpy.ndarray
        The west and east fluxes.
    """

    if len(cell_sub) != stencil_sizes.get(scheme):
        raise SFXM(
            f"Improper stencil size. Expected {stencil_sizes.get(scheme)}, got {len(cell_sub)}")

    if scheme == FVSchemes.LF:
        # West Flux
        ul = cell_sub[0].values()
        ur = cell_sub[1].values()
        Fl = F(ul)
        Fr = F(ur)
        u_diff = ur - ul
        # TODO: Evaluate spectral radius
        sigma = 0.1
        Fw = 0.5 * (Fl + Fr) - 0.5 * sigma * u_diff

        # East Flux
        ul = cell_sub[1].values()
        ur = cell_sub[2].values()
        Fl = F(ul)
        Fr = F(ur)
        u_diff = ur - ul
        # TODO: Evaluate spectral radius
        sigma = 0.1
        Fe = 0.5 * (Fl + Fr) - 0.5 * sigma * u_diff

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

    Returns
    -------
    tuple of numpy.ndarray
        The west and east diffusion fluxes.
    """
    if len(cell_sub) != stencil_sizes.get(scheme):
        raise SFXM(
            f"Improper stencil size. Expected {stencil_sizes.get(scheme)}, got {len(cell_sub)}")

    # Only central scheme for diffusion fluxes
    if scheme == FVSchemes.LF:
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
