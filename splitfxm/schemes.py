from enum import Enum
from .error import SFXM

FDSchemes = Enum("FDSchemes", "CENTRAL RIGHT_BIAS")
FVSchemes = Enum(
    "FVSchemes", "LAX_FRIEDRICHS LAX_WENDROFF CENTRAL UPWIND MUSCL ENO WENO BQUICK QUICK")

FVLimiters = Enum("FVLimiters", "MINMOD VAN_ALBADA SUPERBEE VAN_LEER")

stencil_sizes = {
    FDSchemes.CENTRAL: 3,
    FDSchemes.RIGHT_BIAS: 4,
    FVSchemes.LAX_FRIEDRICHS: 3,
    FVSchemes.LAX_WENDROFF: 3,
    FVSchemes.CENTRAL: 3,
    FVSchemes.UPWIND: 3,
    FVSchemes.MUSCL: 5,
    FVSchemes.ENO: 5,
    FVSchemes.WENO: 5,
    FVSchemes.BQUICK: 5,
    FVSchemes.QUICK: 5
}


def default_scheme(method):
    if method == "FVM":
        return FVSchemes.LAX_FRIEDRICHS
    elif method == "FDM":
        return FDSchemes.CENTRAL
    else:
        raise SFXM("Invalid numerical method specified")
