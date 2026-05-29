import numpy as np
import pytest
from src.basis import LagrangeBasis, local_mass_matrix
from src.mesh import Mesh1D
from src.quadrature import gauss_legendre_ref, map_ref_to_phys


def test_symmetry():
    for p in [1, 2, 3]:
        M = local_mass_matrix(p, 2 * p + 4, h_K=1.0)
        assert np.linalg.norm(M - M.T) < 1e-14, f"p={p}: mass matrix not symmetric"


def test_positive_definite():
    for p in [1, 2, 3]:
        M = local_mass_matrix(p, 2 * p + 4, h_K=1.0)
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), f"p={p}: eigenvalues not all positive: {eigvals}"


def test_exactness_constants():
    """ones^T M ones = h_K (L2 norm of the constant 1 on element of length h_K)."""
    for p in [1, 2, 3]:
        h = 0.7
        M = local_mass_matrix(p, 2 * p + 4, h_K=h)
        ones = np.ones(p + 1)
        np.testing.assert_allclose(ones @ M @ ones, h, rtol=1e-13,
                                   err_msg=f"p={p}: constant integration wrong")


def test_l2_projection_convergence():
    """L2 projection of sin(pi*x) onto V_h^p converges at rate O(h^{p+1})."""
    f = lambda x: np.sin(np.pi * x)
    x_min, x_max = -1.0, 1.0
    N_list = [4, 8, 16, 32]

    for p in [1, 2]:
        n_quad = 2 * p + 4
        xi_q, w_q = gauss_legendre_ref(n_quad)
        basis = LagrangeBasis(p, xi_q)
        phi_vals = basis.phi(xi_q)          # (n_quad, p+1) — fixed for all elements

        errors = []
        for N in N_list:
            mesh = Mesh1D(x_min, x_max, N)
            err_sq = 0.0
            for K in range(N):
                x_L, x_R = mesh.element_interval(K)
                x_phys, J = map_ref_to_phys(xi_q, x_L, x_R)

                M = local_mass_matrix(p, n_quad, h_K=x_R - x_L)
                f_vals = f(x_phys)

                # RHS: b_i = J * Σ_q w_q f(x_q) φ_i(ξ_q)
                b = J * np.einsum('q,q,qi->i', w_q, f_vals, phi_vals)
                c = np.linalg.solve(M, b)

                # L2 error on element K
                diff = f_vals - phi_vals @ c
                err_sq += J * np.dot(w_q, diff ** 2)

            errors.append(np.sqrt(err_sq))

        errors = np.array(errors)
        h_vals = (x_max - x_min) / np.array(N_list, dtype=float)
        eoc = np.log(errors[:-1] / errors[1:]) / np.log(h_vals[:-1] / h_vals[1:])
        assert np.all(eoc > p + 0.5), (
            f"p={p}: EOC={np.round(eoc, 2)}, expected ≈{p + 1}"
        )
