import numpy as np
from splitfxm.error import SFXM
from splitfxm.model import Model
from splitfxm.equations.fd_transport import FDTransportEquation
from splitfxm.equations.fv_transport import FVTransportEquation


class Euler1DConservative(Model):
    """
    A class representing the 1D Euler equations for compressible fluid flow.

    Parameters
    ----------
    gamma : float
        The adiabatic index (ratio of specific heats).
    method : str
        The numerical method to use, either 'FDM' or 'FVM'.
    """

    def __init__(self, gamma=1.4, method='FDM', eps=1e-10):
        """
        Initialize an `Euler1D` object.
        """
        self.gamma = gamma

        # Define the flux function for Euler equations
        # Obtained from the flux_jacobian.py code
        def F(U):
            U1, U2, U3 = U
            F1 = U2
            F2 = 0.5*(3-gamma)*U2**2/U1 + (gamma-1)*U3
            F3 = gamma*U2*U3/U1 - 0.5*(gamma-1)*(U2**3/U1**2)
            return np.array([F1, F2, F3])

        # Diffusion term (not used in standard Euler equations, but can be extended)
        def D(U):
            return np.zeros_like(U)  # No diffusion term in the Euler equations

        # Source term (assuming no external forces, source term is zero)
        def S(U):
            return np.zeros_like(U)

        # Derivative of the flux function with respect to the state vector U
        def dFdU(U):
            U1, U2, U3 = U

            # Second row
            D4 = -0.5*(3 - gamma)*(U2**2/U1**2)
            D5 = (3 - gamma)*U2/U1
            D6 = gamma - 1

            # Third row
            D7 = -gamma*U2*U3/U1**2 + (gamma - 1)*(U2**3/U1**3)
            D8 = gamma*U3/U1 - 1.5*(gamma - 1)*(U2**2/U1**2)
            D9 = gamma * U2/U1

            # Obtained from flux_jacobian.py
            dF_matrix = np.array([
                [0, 1, 0],
                [D4, D5, D6],
                [D7, D8, D9]
            ])
            return dF_matrix

        if method == 'FDM':
            self._equation = FDTransportEquation(F, D, S)
        elif method == 'FVM':
            self._equation = FVTransportEquation(F, D, S, dFdU)
        else:
            raise SFXM("Invalid numerical method specified")

    def conservative_to_primitive(self, U):
        """
        Convert conservative variables to primitive variables.

        Parameters
        ----------
        U : numpy.ndarray
            The conservative variables.

        Returns
        -------
        numpy.ndarray
            The primitive variables.
        """
        U1, U2, U3 = U
        rho = U1
        u = U2 / U1
        p = (self.gamma - 1) * (U3 - 0.5 * U2**2 / U1)
        return np.array([rho, u, p])
