import numpy as np
from .schemes import FVLimiters


def minmod(r):
    """Minmod limiter."""
    return np.maximum(0, np.minimum(1, r))


def van_albada(r):
    """Van Albada limiter."""
    return (r + r**2) / (1 + r**2)


def superbee(r):
    """Superbee limiter."""
    return np.maximum(0, np.maximum(np.minimum(2 * r, 1), np.minimum(r, 2)))


def van_leer(r):
    """Van Leer limiter."""
    return (r + np.abs(r)) / (1 + np.abs(r))


def psi(r, limiter=FVLimiters.MINMOD):
    if limiter == FVLimiters.MINMOD:
        return minmod(r)
    elif limiter == FVLimiters.VAN_ALBADA:
        return van_albada(r)
    elif limiter == FVLimiters.SUPERBEE:
        return superbee(r)
    elif limiter == FVLimiters.VAN_LEER:
        return van_leer(r)
    else:
        raise SFXM("Invalid flux limiter specified")
