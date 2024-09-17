import pytest

from splitfxm.error import SFXM
from splitfxm.schemes import default_scheme, FVSchemes, FDSchemes


def test_default_scheme_fvm():
    """
    Test default_scheme with 'FVM' method.
    """
    assert default_scheme(
        "FVM") == FVSchemes.LAX_FRIEDRICHS, "default_scheme('FVM') should return FVSchemes.LAX_FRIEDRICHS"


def test_default_scheme_fdm():
    """
    Test default_scheme with 'FDM' method.
    """
    assert default_scheme(
        "FDM") == FDSchemes.CENTRAL, "default_scheme('FDM') should return FDSchemes.CENTRAL"


def test_default_scheme_invalid_method():
    """
    Test default_scheme with an invalid method.
    """
    with pytest.raises(SFXM, match="Invalid numerical method specified"):
        default_scheme("InvalidMethod")
