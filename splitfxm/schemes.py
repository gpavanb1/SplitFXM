from enum import Enum
from .error import SFXM

FDSchemes = Enum("FDSchemes", "CENTRAL RIGHT_BIAS")
FVSchemes = Enum("FVSchemes", "LF")

stencil_sizes = {FDSchemes.CENTRAL: 3,
                 FDSchemes.RIGHT_BIAS: 4, FVSchemes.LF: 3}


def default_scheme(method):
    if method == "FVM":
        return FVSchemes.LF
    elif method == "FDM":
        return FDSchemes.CENTRAL
    else:
        raise SFXM("Invalid numerical method specified")
