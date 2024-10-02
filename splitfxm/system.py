import numpy as np
from .domain import Domain
from .constants import btype


class System:
    """
    A class representing a system of equations.

    Parameters
    ----------
    model : Model
        The model for which to solve the system of equations.
    scheme : Schemes
        The discretization scheme to use for the system
    scheme_opts : dict
        A dictionary of options for the scheme.
    """

    def __init__(self, model, scheme, scheme_opts):
        """
        Initialize a System object.
        """
        self._model = model
        self._scheme = scheme
        self._scheme_opts = scheme_opts

    def residuals(self, d: Domain):
        """
        Calculate the residuals for the system of equations.

        Parameters
        ----------
        d : Domain
            The domain for which to calculate the residuals.

        Returns
        -------
        rhs_list : list of ndarray
            The list of residual arrays for each cell in the domain.
        """

        cells = d.cells()

        # Interior indices
        ilo = d.ilo()
        ihi = d.ihi()

        # Get list of boundary points in each direction
        nb_left = d.nb(btype.LEFT)
        nb_right = d.nb(btype.RIGHT)

        # Fetch equation for model
        eq = self._model.equation()

        rhs_list = []

        for i in range(ilo, ihi + 1):
            # Define the neighborhood and band around the current cell
            cell_sub = [cells[i + offset]
                        for offset in range(-nb_left, nb_right + 1)]
            # Send two-sided stencil
            # Let model decide computation
            rhs = eq.residuals(cell_sub, self._scheme, self._scheme_opts)
            rhs_list.append(rhs)

        # Stack the residuals to form a 2D array
        rhs_array = np.vstack(rhs_list)

        return rhs_array
