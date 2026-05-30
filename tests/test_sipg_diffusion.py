import numpy as np
import pytest
from scipy.sparse.linalg import eigsh
from src.mesh import Mesh1D
from src.basis import LagrangeBasis
from src.quadrature import gauss_legendre_ref
from src.dg_operators import (assemble_mass_matrix, assemble_sipg_diffusion,
                               assemble_load_vector)
from src.time_stepper import BackwardEulerSolver


def make_mesh_basis(N, p):
    mesh = Mesh1D(0.0, 1.0, N)
    xi_q, _ = gauss_legendre_ref(2 * p + 4)
    basis = LagrangeBasis(p, xi_q)
    return mesh, basis


# ------------------------------------------------------------------ #
# 1. Symmetry
# ------------------------------------------------------------------ #
@pytest.mark.parametrize('p', [1, 2, 3])
def test_symmetry(p):
    mesh, basis = make_mesh_basis(8, p)
    K = assemble_sipg_diffusion(mesh, basis, a_func=lambda x: 1.0, eta_C=10, p=p)
    diff = K - K.T
    assert diff.data.size == 0 or np.max(np.abs(diff.data)) < 1e-12, (
        f"p={p}: K_diff not symmetric, max off-sym = {np.max(np.abs(diff.data)):.2e}"
    )


# ------------------------------------------------------------------ #
# 2. Coercivity — smallest eigenvalue positive
# ------------------------------------------------------------------ #
def test_coercivity():
    p = 1
    mesh, basis = make_mesh_basis(8, p)
    K = assemble_sipg_diffusion(mesh, basis, a_func=lambda x: 1.0, eta_C=10, p=p)
    # smallest algebraic eigenvalue via ARPACK (shift-invert mode)
    lam_min = eigsh(K.toarray(), k=1, which='SM', return_eigenvectors=False)[0]
    assert lam_min > 0, f"K_diff not positive definite, λ_min = {lam_min:.4e}"


# ------------------------------------------------------------------ #
# MMS helpers shared by test and examples
# ------------------------------------------------------------------ #
def mms_functions(x_min=0.0, x_max=1.0, a_val=1.0):
    L = x_max - x_min

    def u_ex(x, t):
        return np.exp(-t) * np.sin(np.pi * (x - x_min) / L)

    def du_ex_dx(x, t):
        return np.exp(-t) * (np.pi / L) * np.cos(np.pi * (x - x_min) / L)

    def f_func(x, t):
        return u_ex(x, t) * (-1.0 + a_val * (np.pi / L) ** 2)

    def g_D_func(x, t):
        return u_ex(x, t)

    def a_func(x):
        return a_val

    return u_ex, du_ex_dx, f_func, g_D_func, a_func


def compute_errors(mesh, basis, U, u_ex, du_ex_dx, T, p):
    """L2 and broken-H1 (DG energy) errors at time T."""
    n_quad = 2 * p + 4
    xi_q, w_q = gauss_legendre_ref(n_quad)
    phi_q = basis.phi(xi_q)
    dphi_q = basis.dphi_dxi(xi_q)
    n_loc = p + 1

    e_L2_sq = 0.0
    e_DG_sq = 0.0
    for K in range(mesh.N):
        x_L, x_R = mesh.element_interval(K)
        h_K = x_R - x_L
        J = h_K / 2.0
        x_phys = 0.5 * (x_L + x_R) + 0.5 * h_K * xi_q

        dofs = np.arange(K * n_loc, (K + 1) * n_loc)
        U_K = U[dofs]

        u_h = phi_q @ U_K
        du_h_dx = (2.0 / h_K) * (dphi_q @ U_K)

        u_exact = np.fromiter((u_ex(x, T) for x in x_phys), dtype=float)
        du_exact = np.fromiter((du_ex_dx(x, T) for x in x_phys), dtype=float)

        e_L2_sq += J * np.dot(w_q, (u_exact - u_h) ** 2)
        e_DG_sq += J * np.dot(w_q, (du_exact - du_h_dx) ** 2)

    return np.sqrt(e_L2_sq), np.sqrt(e_DG_sq)


# ------------------------------------------------------------------ #
# 3. Heat equation spatial convergence (p=1, L2 EOC ≈ 2)
# ------------------------------------------------------------------ #
def test_heat_convergence_p1():
    p = 1
    T = 0.5
    dt = 1e-4
    N_list = [8, 16, 32]
    eta_C = 10
    u_ex, du_ex_dx, f_func, g_D_func, a_func = mms_functions()

    errors = []
    for N in N_list:
        mesh, basis = make_mesh_basis(N, p)
        config = {'penalty_C': eta_C}
        solver = BackwardEulerSolver(mesh, basis, config)
        history = solver.solve(
            u0_func=lambda x: u_ex(x, 0.0),
            f_func=f_func,
            a_func=a_func,
            g_D_func=g_D_func,
            T=T,
            dt=dt,
        )
        _, U_T = history[-1]
        e_L2, _ = compute_errors(mesh, basis, U_T, u_ex, du_ex_dx, T, p)
        errors.append(e_L2)

    errors = np.array(errors)
    h_vals = 1.0 / np.array(N_list, dtype=float)
    eoc = np.log(errors[:-1] / errors[1:]) / np.log(h_vals[:-1] / h_vals[1:])
    assert np.all(eoc > 1.8) and np.all(eoc < 2.2), (
        f"p=1 L2 EOC out of [1.8, 2.2]: {np.round(eoc, 3)}"
    )
