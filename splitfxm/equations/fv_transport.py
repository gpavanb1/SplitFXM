from splitfxm.flux import fluxes, diffusion_fluxes


class FVTransportEquation:
    """
    A class representing a transport equation.

    Parameters
    ----------
    F : function
        The function representing the flux for the equation.
    D : function
        The function representing the diffusion term for the equation.
    S : function
        The function representing the source term for the equation.
    """

    def __init__(self, F, D, S):
        """
        Initialize a `TransportEquation` object.
        """
        self.F = F
        self.D = D
        self.S = S

    def residuals(self, cell_sub, scheme):
        """
        Calculate the residuals for the transport equation at the given cell subset using the given scheme.

        Parameters
        ----------
        cell_sub : list of Cell
            The subset of cells to calculate the residuals for.
        scheme : Scheme
            The scheme to use to calculate the residuals.

        Returns
        -------
        numpy.ndarray
            The residuals for the transport equation at the given cell subset.
        """

        # Cell width
        # Calculate for center cell
        # Average of distance between adjacent cell centers
        ic = len(cell_sub) // 2
        dxw = cell_sub[ic].x() - cell_sub[ic - 1].x()
        dxe = cell_sub[ic + 1].x() - cell_sub[ic].x()
        dx = 0.5 * (dxw + dxe)

        # Calculate fluxes
        Fw, Fe = fluxes(self.F, cell_sub, scheme)
        DFw, DFe = diffusion_fluxes(self.D, cell_sub, scheme)
        rhs = self.S(cell_sub) - (1 / dx) * (Fe - Fw) + (1 / dx) * (DFe - DFw)
        return rhs
