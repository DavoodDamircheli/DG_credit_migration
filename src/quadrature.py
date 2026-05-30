import numpy as np
from functools import lru_cache


@lru_cache(maxsize=32)
def gauss_legendre_ref(n_points):
    """Gauss-Legendre nodes and weights on [-1, 1] (cached per n_points)."""
    xi, w = np.polynomial.legendre.leggauss(n_points)
    xi.flags.writeable = False
    w.flags.writeable  = False
    return xi, w


def map_ref_to_phys(xi, x_left, x_right):
    """Map reference coordinates xi in [-1,1] to physical [x_left, x_right].

    Returns (x_phys, J) where J = h_K/2 is the Jacobian.
    """
    h_K = x_right - x_left
    x_phys = 0.5 * (x_left + x_right) + 0.5 * h_K * xi
    J = h_K / 2.0
    return x_phys, J
