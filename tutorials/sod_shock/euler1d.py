import numpy as np
from splitfxm.error import SFXM
from splitfxm.model import Model
from splitfxm.equations.fd_transport import FDTransportEquation
from splitfxm.equations.fv_transport import FVTransportEquation


class Euler1D(Model):
    """
    A class representing the 1D Euler equations for compressible fluid flow.

    Parameters
    ----------
    gamma : float
        The adiabatic index (ratio of specific heats).
    method : str
        The numerical method to use, either 'FDM' or 'FVM'.
    """

    def __init__(self, gamma=1.4, method='FDM'):
        """
        Initialize an `Euler1D` object.
        """
        self.gamma = gamma

        # Define the flux function for Euler equations
        def F(U):
            rho, u, p = U
            E = p / (self.gamma - 1) + 0.5 * rho * u**2  # Total energy
            return np.array([rho * u, rho * u**2 + p, u * (E + p)])

        # Diffusion term (not used in standard Euler equations, but can be extended)
        def D(U):
            return np.zeros_like(U)  # No diffusion term in the Euler equations

        # Source term (assuming no external forces, source term is zero)
        def S(U):
            return np.zeros_like(U)

        # Derivative of the flux function with respect to the state vector U
        def dFdU(U):
            rho, u, p = U
            # Obtained from flux_jacobian.py
            dF_matrix = np.array([
                [u, rho, 0],
                [u**2, 2*rho*u, 1],
                [0.5*(u**3.), 1.5*rho*(u**2.) + gamma*p /
                 (gamma - 1.), gamma*u/(gamma - 1.)]
            ])
            return dF_matrix

        if method == 'FDM':
            self._equation = FDTransportEquation(F, D, S)
        elif method == 'FVM':
            self._equation = FVTransportEquation(F, D, S, dFdU)
        else:
            raise SFXM("Invalid numerical method specified")
