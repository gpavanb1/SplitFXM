from enum import Enum
from .error import SFXM

FDSchemes = Enum("FDSchemes", "CENTRAL RIGHT_BIAS")
FVSchemes = Enum(
    "FVSchemes", "LAX_FRIEDRICHS LAX_WENDROFF CENTRAL UPWIND MUSCL ENO WENO BQUICK QUICK")

stencil_sizes = {
    FDSchemes.CENTRAL: 3,
    FDSchemes.RIGHT_BIAS: 4,
    FVSchemes.LAX_FRIEDRICHS: 3,
    FVSchemes.LAX_WENDROFF: 3,
    FVSchemes.CENTRAL: 3,
    FVSchemes.UPWIND: 3,
    FVSchemes.MUSCL: 4,
    FVSchemes.ENO: 4,
    FVSchemes.WENO: 4,
    FVSchemes.BQUICK: 4,
    FVSchemes.QUICK: 4
}


def default_scheme(method):
    if method == "FVM":
        return FVSchemes.LAX_FRIEDRICHS
    elif method == "FDM":
        return FDSchemes.CENTRAL
    else:
        raise SFXM("Invalid numerical method specified")
