from .derivatives import FDSchemes
from .flux import FVSchemes
from .error import SFXM


def default_scheme(method):
    if method == "FVM":
        return FVSchemes.LF
    elif method == "FDM":
        return FDSchemes.CENTRAL
    else:
        raise SFXM("Invalid numerical method specified")
