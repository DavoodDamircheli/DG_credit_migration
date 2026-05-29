import numpy as np
import pytest
from src.basis import LagrangeBasis, gauss_lobatto_points
from src.quadrature import gauss_legendre_ref


def make_basis(p):
    n_quad = 2 * p + 4
    xi_q, _ = gauss_legendre_ref(n_quad)
    return LagrangeBasis(p, xi_q)


def test_partition_of_unity():
    rng = np.random.default_rng(0)
    xi_test = rng.uniform(-1.0, 1.0, 20)
    for p in [1, 2, 3]:
        basis = make_basis(p)
        row_sums = basis.phi(xi_test).sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-14,
                                   err_msg=f"Partition of unity failed for p={p}")


def test_interpolation_exactness():
    """p=2 basis interpolates f(xi)=xi^2 exactly."""
    p = 2
    basis = make_basis(p)
    f_nodes = basis.nodes ** 2                      # values at GLL nodes
    xi_test = np.linspace(-1.0, 1.0, 10)
    f_interp = basis.phi(xi_test) @ f_nodes         # Σ_j f(ξ_j) φ_j(ξ)
    np.testing.assert_allclose(f_interp, xi_test ** 2, atol=1e-13,
                               err_msg="Interpolation of f=xi^2 not exact for p=2")


def test_derivative_consistency():
    """Centered finite-difference check of dphi_dxi for p=1,2."""
    delta = 1e-6
    xi_test = np.linspace(-0.9, 0.9, 7)
    for p in [1, 2]:
        basis = make_basis(p)
        dphi_exact = basis.dphi_dxi(xi_test)
        dphi_fd = (basis.phi(xi_test + delta) - basis.phi(xi_test - delta)) / (2.0 * delta)
        np.testing.assert_allclose(dphi_exact, dphi_fd, atol=1e-5,
                                   err_msg=f"Derivative FD check failed for p={p}")
