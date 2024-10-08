import numpy as np
import pytest

from splitfxm.error import SFXM
from splitfxm.model import Model
from splitfxm.equations.fd_transport import FDTransportEquation
from splitfxm.equations.fv_transport import FVTransportEquation


def F(u): return np.array([self.c * x for x in u])
def D(u): return np.array([self.nu * x for x in u])
def S(u): return np.array([0.0])
def dFdU(u): return np.diag([self.c] * len(u))


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

        if method == 'FDM':
            self._equation = FDTransportEquation(F, D, S)
        elif method == 'FVM':
            self._equation = FVTransportEquation(F, D, S, dFdU)
        else:
            raise SFXM("Invalid numerical method specified")


def test_model_initialization():
    """
    Test the initialization of the Model class.
    """
    model = Model()
    assert model._equation == None, "Model should have no equation by default"


def test_model_equations_method():
    """
    Test the equations method of the Model class.
    """
    model = AdvectionDiffusion(c=0.2, nu=0.001, method='FDM')
    equation = model.equation()
    assert isinstance(
        equation, FDTransportEquation), "equations() method should return a Advection-Diffusion equations"

    model = AdvectionDiffusion(c=0.2, nu=0.001, method='FVM')
    equation = model.equation()
    assert isinstance(
        equation, FVTransportEquation), "equations() method should return a Advection-Diffusion equations"
