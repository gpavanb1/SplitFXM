import pytest
import numpy as np
from splitfxm.cell import Cell
from splitfxm.error import SFXM
from splitfxm.flux import fluxes, diffusion_fluxes
from splitfxm.schemes import FVSchemes


def flux_function(values):
    """Simple flux function for testing."""
    return values * 2


def diffusion_function(values):
    """Simple diffusion function for testing."""
    return values * 0.5


@pytest.fixture
def cells():
    """Fixture to create a standard 3-cell stencil."""
    return [
        Cell(x=0.0, value=np.array([1.0, 2.0])),
        Cell(x=1.0, value=np.array([3.0, 4.0])),
        Cell(x=2.0, value=np.array([5.0, 6.0]))
    ]


def test_fluxes(cells):
    """Test west and east fluxes calculation."""
    Fw, Fe = fluxes(flux_function, cells, FVSchemes.LF)

    expected_Fw = 0.5 * (flux_function(cells[0].values()) + flux_function(
        cells[1].values())) - 0.5 * 0.1 * (cells[1].values() - cells[0].values())
    expected_Fe = 0.5 * (flux_function(cells[1].values()) + flux_function(
        cells[2].values())) - 0.5 * 0.1 * (cells[2].values() - cells[1].values())

    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_invalid_stencil():
    """Test flux calculation with an invalid stencil size."""
    with pytest.raises(SFXM):
        fluxes(flux_function, [
               Cell(x=0.0, value=np.array([1.0]))], FVSchemes.LF)


def test_fluxes_invalid_scheme(cells):
    """Test flux calculation with an unsupported scheme."""
    with pytest.raises(SFXM):
        fluxes(flux_function, cells, "UNSUPPORTED_SCHEME")


def test_diffusion_fluxes(cells):
    """Test west and east diffusion fluxes calculation."""
    Dw, De = diffusion_fluxes(diffusion_function, cells, FVSchemes.LF)

    dxw = 0.5 * (cells[1].x() - cells[0].x())
    dxe = 0.5 * (cells[2].x() - cells[1].x())

    expected_Dw = (diffusion_function(
        cells[1].values()) - diffusion_function(cells[0].values())) / dxw
    expected_De = (diffusion_function(
        cells[2].values()) - diffusion_function(cells[1].values())) / dxe

    assert np.allclose(Dw, expected_Dw)
    assert np.allclose(De, expected_De)


def test_diffusion_fluxes_invalid_stencil():
    """Test diffusion flux calculation with an invalid stencil size."""
    with pytest.raises(SFXM):
        diffusion_fluxes(diffusion_function, [Cell(
            x=0.0, value=np.array([1.0]))], FVSchemes.LF)


def test_diffusion_fluxes_invalid_scheme(cells):
    """Test diffusion flux calculation with an unsupported scheme."""
    with pytest.raises(SFXM):
        diffusion_fluxes(diffusion_function, cells, "UNSUPPORTED_SCHEME")
