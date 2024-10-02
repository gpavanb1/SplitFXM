import pytest
import numpy as np
from splitfxm.cell import Cell
from splitfxm.error import SFXM
from splitfxm.flux import fluxes, diffusion_fluxes
from splitfxm.schemes import FVSchemes


def flux_function(values):
    """Simple flux function for testing."""
    return values * 2


def dfdu(values):
    """Simple dFdU function for testing."""
    return np.diag([2] * len(values))


def flux_function_n(values):
    """Simple flux function for testing."""
    return values * -2.


def dfdu_n(values):
    """Simple dFdU function for testing."""
    return np.diag([-2.] * len(values))


def diffusion_function(values):
    """Simple diffusion function for testing."""
    return values * 0.5


@pytest.fixture
def cells3():
    """Fixture to create a standard 3-cell stencil."""
    return [
        Cell(x=0.0, value=np.array([1.0, 2.0])),
        Cell(x=1.0, value=np.array([3.0, 4.0])),
        Cell(x=2.0, value=np.array([2.0, 6.0])),
    ]


@pytest.fixture
def cells3_n():
    """Fixture to create a standard 3-cell stencil with non-positive entries"""
    return [
        Cell(x=0.0, value=np.array([1.0, 2.0])),
        Cell(x=1.0, value=np.array([-3.0, 4.0])),
        Cell(x=2.0, value=np.array([2.0, -6.0]))
    ]


def test_fluxes_lf(cells3):
    """Test west and east fluxes calculation for Lax-Friedrichs scheme."""
    Fw, Fe = fluxes(flux_function, cells3, FVSchemes.LAX_FRIEDRICHS, dfdu)

    expected_Fw = 0.5 * (flux_function(cells3[0].values()) + flux_function(
        cells3[1].values())) - 0.5 * 2 * (cells3[1].values() - cells3[0].values())
    expected_Fe = 0.5 * (flux_function(cells3[1].values()) + flux_function(
        cells3[2].values())) - 0.5 * 2 * (cells3[2].values() - cells3[1].values())

    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_lf_fail(cells3):
    """Test west and east fluxes calculation for Lax-Friedrichs scheme."""
    with pytest.raises(SFXM):
        Fw, Fe = fluxes(flux_function, cells3, FVSchemes.LAX_FRIEDRICHS)


def test_fluxes_upwind(cells3):
    """Test west and east fluxes calculation for Upwind scheme."""
    Fw, Fe = fluxes(flux_function, cells3, FVSchemes.UPWIND, dfdu)

    # Extract values
    ul = cells3[0].values()
    uc = cells3[1].values()
    ur = cells3[2].values()

    # Calculate expected fluxes
    expected_Fw = flux_function(ul)
    expected_Fe = flux_function(uc)

    # Check if the computed fluxes are close to expected values
    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_upwind_neg(cells3_n):
    """Test west and east fluxes calculation for Upwind scheme."""
    Fw, Fe = fluxes(flux_function_n, cells3_n, FVSchemes.UPWIND, dfdu_n)

    # Extract values
    ul = cells3_n[0].values()
    uc = cells3_n[1].values()
    ur = cells3_n[2].values()

    # Calculate expected fluxes
    expected_Fw = flux_function_n(uc)
    expected_Fe = flux_function_n(ur)

    # Check if the computed fluxes are close to expected values
    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_upwind_fail(cells3):
    """Test west and east fluxes calculation for Upwind scheme."""
    with pytest.raises(SFXM):
        Fw, Fe = fluxes(flux_function, cells3, FVSchemes.UPWIND)


def test_fluxes_central(cells3):
    """Test west and east fluxes calculation for Central differencing scheme."""
    Fw, Fe = fluxes(flux_function, cells3, FVSchemes.CENTRAL)

    expected_Fw = 0.5 * \
        (flux_function(cells3[0].values()) + flux_function(cells3[1].values()))
    expected_Fe = 0.5 * \
        (flux_function(cells3[1].values()) + flux_function(cells3[2].values()))

    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_invalid_stencil():
    """Test flux calculation with an invalid stencil size."""
    with pytest.raises(SFXM):
        fluxes(flux_function, [
               Cell(x=0.0, value=np.array([1.0]))], FVSchemes.LAX_FRIEDRICHS, dfdu)


def test_fluxes_invalid_scheme(cells3):
    """Test flux calculation with an unsupported scheme."""
    with pytest.raises(SFXM):
        fluxes(flux_function, cells3, "UNSUPPORTED_SCHEME", dfdu)


def test_diffusion_fluxes(cells3):
    """Test west and east diffusion fluxes calculation."""
    Dw, De = diffusion_fluxes(
        diffusion_function, cells3, FVSchemes.LAX_FRIEDRICHS)

    dxw = 0.5 * (cells3[1].x() - cells3[0].x())
    dxe = 0.5 * (cells3[2].x() - cells3[1].x())

    expected_Dw = (diffusion_function(
        cells3[1].values()) - diffusion_function(cells3[0].values())) / dxw
    expected_De = (diffusion_function(
        cells3[2].values()) - diffusion_function(cells3[1].values())) / dxe

    assert np.allclose(Dw, expected_Dw)
    assert np.allclose(De, expected_De)


def test_diffusion_fluxes_invalid_stencil():
    """Test diffusion flux calculation with an invalid stencil size."""
    with pytest.raises(SFXM):
        diffusion_fluxes(diffusion_function, [Cell(
            x=0.0, value=np.array([1.0]))], FVSchemes.LAX_FRIEDRICHS)


def test_diffusion_fluxes_invalid_scheme(cells3):
    """Test diffusion flux calculation with an unsupported scheme."""
    with pytest.raises(SFXM):
        diffusion_fluxes(diffusion_function, cells3, "UNSUPPORTED_SCHEME")
