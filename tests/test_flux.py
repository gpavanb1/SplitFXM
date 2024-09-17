import pytest
import numpy as np
from splitfxm.cell import Cell
from splitfxm.error import SFXM
from splitfxm.flux import fluxes, diffusion_fluxes, minmod
from splitfxm.schemes import FVSchemes


def flux_function(values):
    """Simple flux function for testing."""
    return values * 2


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


@pytest.fixture
def cells4():
    """Fixture to create a standard 4-cell stencil."""
    return [
        Cell(x=0.0, value=np.array([1.0, 2.0])),
        Cell(x=1.0, value=np.array([-3.0, 4.0])),
        Cell(x=2.0, value=np.array([5.0, 6.0])),
        Cell(x=3.0, value=np.array([-2.0, 8.0]))
    ]


@pytest.fixture
def cells4_s():
    """Fixture to create a smoother 4-cell stencil."""
    return [
        Cell(x=0.0, value=np.array([1.0, 2.0])),
        Cell(x=1.0, value=np.array([1.0, 2.0])),
        Cell(x=2.0, value=np.array([2.0, 3.0])),
        Cell(x=3.0, value=np.array([8.0, 9.0]))
    ]


def test_fluxes_lf(cells3):
    """Test west and east fluxes calculation for Lax-Friedrichs scheme."""
    Fw, Fe = fluxes(flux_function, cells3, FVSchemes.LAX_FRIEDRICHS)

    expected_Fw = 0.5 * (flux_function(cells3[0].values()) + flux_function(
        cells3[1].values())) - 0.5 * 0.1 * (cells3[1].values() - cells3[0].values())
    expected_Fe = 0.5 * (flux_function(cells3[1].values()) + flux_function(
        cells3[2].values())) - 0.5 * 0.1 * (cells3[2].values() - cells3[1].values())

    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_upwind(cells3):
    """Test west and east fluxes calculation for Upwind scheme."""
    Fw, Fe = fluxes(flux_function, cells3, FVSchemes.UPWIND)

    # Extract values
    ul = cells3[0].values()
    uc = cells3[1].values()
    ur = cells3[2].values()

    # Calculate expected fluxes
    if np.all(uc > 0):
        expected_Fw = flux_function(ul)
    else:
        expected_Fw = flux_function(uc)

    if np.all(ur > 0):
        expected_Fe = flux_function(uc)
    else:
        expected_Fe = flux_function(ur)

    # Check if the computed fluxes are close to expected values
    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_upwind_neg(cells3_n):
    """Test west and east fluxes calculation for Upwind scheme."""
    Fw, Fe = fluxes(flux_function, cells3_n, FVSchemes.UPWIND)

    # Extract values
    ul = cells3_n[0].values()
    uc = cells3_n[1].values()
    ur = cells3_n[2].values()

    # Calculate expected fluxes
    if np.all(uc > 0):
        expected_Fw = flux_function(ul)
    else:
        expected_Fw = flux_function(uc)

    if np.all(ur > 0):
        expected_Fe = flux_function(uc)
    else:
        expected_Fe = flux_function(ur)

    # Check if the computed fluxes are close to expected values
    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_central(cells3):
    """Test west and east fluxes calculation for Central differencing scheme."""
    Fw, Fe = fluxes(flux_function, cells3, FVSchemes.CENTRAL)

    expected_Fw = 0.5 * \
        (flux_function(cells3[0].values()) + flux_function(cells3[1].values()))
    expected_Fe = 0.5 * \
        (flux_function(cells3[1].values()) + flux_function(cells3[2].values()))

    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_lax_wendroff(cells3):
    """Test west and east fluxes calculation for Lax-Wendroff scheme."""
    Fw, Fe = fluxes(flux_function, cells3, FVSchemes.LAX_WENDROFF)

    expected_Fw = flux_function(cells3[0].values()) + 0.5 * (cells3[1].values() - cells3[0].values()) * (
        flux_function(cells3[1].values()) - flux_function(cells3[0].values()))
    expected_Fe = flux_function(cells3[1].values()) + 0.5 * (cells3[2].values() - cells3[1].values()) * (
        flux_function(cells3[2].values()) - flux_function(cells3[1].values()))

    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_quick(cells4):
    """Test west and east fluxes calculation for QUICK scheme."""
    Fw, Fe = fluxes(flux_function, cells4, FVSchemes.QUICK)

    expected_Fw = (3/8) * flux_function(cells4[0].values()) + (6/8) * flux_function(
        cells4[1].values()) - (1/8) * flux_function(cells4[2].values())
    expected_Fe = (3/8) * flux_function(cells4[1].values()) + (6/8) * flux_function(
        cells4[2].values()) - (1/8) * flux_function(cells4[3].values())

    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_bquick(cells4):
    """Test west and east fluxes calculation for BQUICK scheme."""
    Fw, Fe = fluxes(flux_function, cells4, FVSchemes.BQUICK)

    expected_Fw = np.clip((3/8) * flux_function(cells4[0].values()) + (6/8) * flux_function(cells4[1].values()) - (
        1/8) * flux_function(cells4[2].values()), flux_function(cells4[0].values()), flux_function(cells4[2].values()))
    expected_Fe = np.clip((3/8) * flux_function(cells4[1].values()) + (6/8) * flux_function(cells4[2].values()) - (
        1/8) * flux_function(cells4[3].values()), flux_function(cells4[1].values()), flux_function(cells4[3].values()))

    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_muscl(cells4):
    """Test west and east fluxes calculation for MUSCL scheme."""
    Fw, Fe = fluxes(flux_function, cells4, FVSchemes.MUSCL)

    slope_w = minmod(cells4[1].values() - cells4[0].values(),
                     cells4[2].values() - cells4[1].values())
    slope_e = minmod(cells4[2].values() - cells4[1].values(),
                     cells4[3].values() - cells4[2].values())

    expected_Fw = flux_function(cells4[1].values() - 0.5 * slope_w)
    expected_Fe = flux_function(cells4[1].values() + 0.5 * slope_e)

    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_eno(cells4):
    """Test west and east fluxes calculation for ENO scheme."""
    Fw, Fe = fluxes(flux_function, cells4, FVSchemes.ENO)

    diff1_w = cells4[1].values() - cells4[0].values()
    diff2_w = cells4[2].values() - cells4[1].values()
    diff3_w = cells4[3].values() - cells4[2].values()

    if np.all(np.abs(diff2_w - diff1_w) < np.abs(diff3_w - diff2_w)):
        expected_Fw = flux_function(cells4[0].values())
    else:
        expected_Fw = flux_function(cells4[1].values())

    expected_Fe = flux_function(cells4[2].values())

    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_eno_smooth(cells4_s):
    """Test west and east fluxes calculation for ENO scheme with a smooth stencil"""
    Fw, Fe = fluxes(flux_function, cells4_s, FVSchemes.ENO)

    diff1_w = cells4_s[1].values() - cells4_s[0].values()
    diff2_w = cells4_s[2].values() - cells4_s[1].values()
    diff3_w = cells4_s[3].values() - cells4_s[2].values()

    if np.all(np.abs(diff2_w - diff1_w) < np.abs(diff3_w - diff2_w)):
        expected_Fw = flux_function(cells4_s[0].values())
    else:
        expected_Fw = flux_function(cells4_s[1].values())

    expected_Fe = flux_function(cells4_s[2].values())

    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_weno(cells4):
    """Test west and east fluxes calculation for WENO scheme."""
    Fw, Fe = fluxes(flux_function, cells4, FVSchemes.WENO)

    beta0_w = (13/12) * (cells4[0].values() - 2*cells4[1].values() + cells4[2].values())**2 + (
        1/4) * (cells4[0].values() - 4*cells4[1].values() + 3*cells4[2].values())**2
    beta1_w = (13/12) * (cells4[1].values() - 2*cells4[2].values() +
                         cells4[3].values())**2 + (1/4) * (cells4[1].values() - cells4[3].values())**2

    epsilon = 1e-6
    alpha0_w = 1 / (epsilon + beta0_w)**2
    alpha1_w = 1 / (epsilon + beta1_w)**2
    w0_w = alpha0_w / (alpha0_w + alpha1_w)
    w1_w = alpha1_w / (alpha0_w + alpha1_w)

    expected_Fw = w0_w * \
        flux_function(cells4[0].values()) + w1_w * \
        flux_function(cells4[1].values())
    expected_Fe = w0_w * \
        flux_function(cells4[2].values()) + w1_w * \
        flux_function(cells4[3].values())

    assert np.allclose(Fw, expected_Fw)
    assert np.allclose(Fe, expected_Fe)


def test_fluxes_invalid_stencil():
    """Test flux calculation with an invalid stencil size."""
    with pytest.raises(SFXM):
        fluxes(flux_function, [
               Cell(x=0.0, value=np.array([1.0]))], FVSchemes.LAX_FRIEDRICHS)


def test_fluxes_invalid_scheme(cells3):
    """Test flux calculation with an unsupported scheme."""
    with pytest.raises(SFXM):
        fluxes(flux_function, cells3, "UNSUPPORTED_SCHEME")


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
