from enum import Enum
from .error import SFXM

FDSchemes = Enum("FDSchemes", "CENTRAL RIGHT_BIAS")
FVSchemes = Enum(
    "FVSchemes", "LAX_FRIEDRICHS CENTRAL UPWIND MUSCL ENO WENO QUICK")

stencil_sizes = {
    FDSchemes.CENTRAL: 3,
    FDSchemes.RIGHT_BIAS: 4,
    FVSchemes.LAX_FRIEDRICHS: 3,
    FVSchemes.CENTRAL: 3,
    FVSchemes.UPWIND: 3
}


def default_scheme(method):
    if method == "FVM":
        return FVSchemes.LAX_FRIEDRICHS
    elif method == "FDM":
        return FDSchemes.CENTRAL
    else:
        raise SFXM("Invalid numerical method specified")
