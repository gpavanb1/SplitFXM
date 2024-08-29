from splitfxm.derivatives import Dx, D2x


class FDTransportEquation:
    """
    A class representing a transport equation.

    Parameters
    ----------
    F : function
        The function representing the advection term for the equation.
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

        # Calculate equation
        rhs = (
            self.S(cell_sub)
            - Dx(self.F, cell_sub, scheme)
            + D2x(self.D, cell_sub, scheme)
        )
        return rhs
