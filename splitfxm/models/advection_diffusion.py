import numpy as np

from splitfxm.error import SFXM
from splitfxm.model import Model
from splitfxm.equations.fd_transport import FDTransportEquation
from splitfxm.equations.fv_transport import FVTransportEquation


class AdvectionDiffusion(Model):
    """
    A class representing an advection-diffusion model.

    Parameters
    ----------
    c : float
        The advection coefficient.
    nu : float
        The diffusion coefficient.
    """

    def __init__(self, c, nu, method='FDM'):
        """
        Initialize an `AdvectionDiffusion` object.
        """

        self.c = c
        self.nu = nu
        def F(u): return np.array([self.c * x for x in u])
        def D(u): return np.array([self.nu * x for x in u])
        def S(u): return np.array([0.0])
        def dFdU(u): return np.diag([self.c] * len(u))
        if method == 'FDM':
            self._equation = FDTransportEquation(F, D, S)
        elif method == 'FVM':
            self._equation = FVTransportEquation(F, D, S, dFdU)
        else:
            raise SFXM("Invalid numerical method specified")
