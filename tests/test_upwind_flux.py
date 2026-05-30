"""Tests for the upwind drift operator (Phase 3).

All tests use domain [-1, 1].  Tests 1 and 2 verify sign correctness for
constant positive/negative drift.  Tests 3 and 4 verify the γ-correction is
necessary for correct convergence with variable β.
"""
import numpy as np
import pytest
from scipy.sparse.linalg import splu

from src.mesh import Mesh1D
from src.basis import LagrangeBasis
from src.quadrature import gauss_legendre_ref
from src.dg_operators import (
    assemble_mass_matrix,
    assemble_upwind_drift,
    assemble_drift_inflow_bc,
    assemble_load_vector,
)


X_MIN, X_MAX = -1.0, 1.0


def make_mesh_basis(N, p):
    mesh = Mesh1D(X_MIN, X_MAX, N)
    xi_q, _ = gauss_legendre_ref(2 * p + 4)
    basis = LagrangeBasis(p, xi_q)
    return mesh, basis


def l2_error(mesh, basis, U, u_ex, T):
    """L² error of DG solution U against u_ex(x, T)."""
    p = basis.p
    n_loc = p + 1
    n_quad = 2 * p + 4
    xi_q, w_q = gauss_legendre_ref(n_quad)
    phi_q = basis.phi(xi_q)

    err_sq = 0.0
    for K in range(mesh.N):
        x_L, x_R = mesh.element_interval(K)
        h_K = x_R - x_L
        J = h_K / 2.0
        x_phys = 0.5 * (x_L + x_R) + 0.5 * h_K * xi_q
        U_K = U[K * n_loc:(K + 1) * n_loc]
        u_h = phi_q @ U_K
        u_e = u_ex(x_phys, T)
        err_sq += J * np.dot(w_q, (u_e - u_h) ** 2)
    return np.sqrt(err_sq)


def project_l2(mesh, basis, u_func, t=0.0):
    """L²-project u_func(x, t) into V_h^p."""
    p = basis.p
    n_loc = p + 1
    n_quad = 2 * p + 4
    xi_q, w_q = gauss_legendre_ref(n_quad)
    phi_q = basis.phi(xi_q)

    U = np.zeros(mesh.N * n_loc)
    for K in range(mesh.N):
        x_L, x_R = mesh.element_interval(K)
        h_K = x_R - x_L
        J = h_K / 2.0
        x_phys = 0.5 * (x_L + x_R) + 0.5 * h_K * xi_q
        u_q = u_func(x_phys, t)
        M_loc = J * np.einsum('q,qi,qj->ij', w_q, phi_q, phi_q)
        b_loc = J * np.einsum('q,q,qi->i', w_q, u_q, phi_q)
        dofs = slice(K * n_loc, (K + 1) * n_loc)
        U[dofs] = np.linalg.solve(M_loc, b_loc)
    return U


def run_advection(N, p, beta_val, T, dt, u_ex, g_D_func, gamma_val=0.0):
    """Backward Euler for u_t + β u_x = 0 with constant β and γ.

    Returns the L² error at time T.
    """
    mesh, basis = make_mesh_basis(N, p)

    beta_func  = lambda x: np.full_like(np.asarray(x, dtype=float), beta_val)
    gamma_func = lambda x: np.full_like(np.asarray(x, dtype=float), gamma_val)

    M      = assemble_mass_matrix(mesh, basis)
    K_drift = assemble_upwind_drift(mesh, basis, beta_func, gamma_func, t=0.0)
    A       = (M + dt * K_drift).tocsc()
    A_lu    = splu(A)

    U = project_l2(mesh, basis, u_ex, t=0.0)

    n_steps = int(round(T / dt))
    for n in range(n_steps):
        t_new = (n + 1) * dt
        F_bc = assemble_drift_inflow_bc(mesh, basis, beta_func, g_D_func, t_new)
        rhs = M @ U + dt * F_bc
        U = A_lu.solve(rhs)

    return l2_error(mesh, basis, U, u_ex, T)


def run_variable_drift(N, p, T, dt, beta_func, gamma_func, u_ex, f_func, g_D_func):
    """Backward Euler for u_t + β(x) u_x = f with variable β and γ=∂_x β.

    K_drift is re-assembled each step because beta may be time-dependent;
    here it is time-independent, so we assemble once.
    """
    mesh, basis = make_mesh_basis(N, p)
    n_loc = p + 1
    n_quad = 2 * p + 4
    xi_q, w_q = gauss_legendre_ref(n_quad)
    phi_q = basis.phi(xi_q)

    M       = assemble_mass_matrix(mesh, basis)
    K_drift = assemble_upwind_drift(mesh, basis, beta_func, gamma_func, t=0.0)
    A       = (M + dt * K_drift).tocsc()
    A_lu    = splu(A)

    # Volume load helper (vectorised)
    x_L_arr = mesh.x[:-1]; x_R_arr = mesh.x[1:]
    h_K_arr = x_R_arr - x_L_arr; J_K_arr = h_K_arr / 2.0
    x_phys_all = (0.5 * (x_L_arr[:, None] + x_R_arr[:, None])
                  + 0.5 * h_K_arr[:, None] * xi_q[None, :])

    def assemble_source(t):
        f_all = np.asarray(f_func(x_phys_all, t), dtype=float)
        return (J_K_arr[:, None]
                * np.einsum('q,Kq,qi->Ki', w_q, f_all, phi_q)).ravel()

    U = project_l2(mesh, basis, u_ex, t=0.0)

    n_steps = int(round(T / dt))
    for n in range(n_steps):
        t_new = (n + 1) * dt
        F_vol = assemble_source(t_new)
        F_bc  = assemble_drift_inflow_bc(mesh, basis, beta_func, g_D_func, t_new)
        rhs = M @ U + dt * (F_vol + F_bc)
        U = A_lu.solve(rhs)

    return l2_error(mesh, basis, U, u_ex, T)


# ------------------------------------------------------------------ #
# Test 1: Pure advection, β = +1 (solution travels right)
# ------------------------------------------------------------------ #
def _eval_dg_at(mesh, basis, U, x_target):
    """Evaluate the DG solution U at a single physical point x_target."""
    p = basis.p
    n_loc = p + 1
    # find element containing x_target
    K = int(np.searchsorted(mesh.x[1:], x_target, side='right'))
    K = min(K, mesh.N - 1)
    x_L, x_R = mesh.element_interval(K)
    h_K = x_R - x_L
    xi = 2.0 * (x_target - x_L) / h_K - 1.0
    phi = basis.phi(np.array([xi]))[0]
    return float(phi @ U[K * n_loc:(K + 1) * n_loc])


def test_advection_positive_beta():
    beta = 1.0
    T = 0.5
    dt = 1e-3
    N = 32
    p = 1

    def u_ex(x, t):
        return np.sin(np.pi * (x - beta * t))

    g_D_func = lambda x, t: u_ex(np.atleast_1d(np.asarray(x, float)), t)[0]

    mesh, basis = make_mesh_basis(N, p)
    beta_func  = lambda x: np.full_like(np.asarray(x, float), beta)
    gamma_func = lambda x: np.zeros_like(np.asarray(x, float))
    M = assemble_mass_matrix(mesh, basis)
    K_drift = assemble_upwind_drift(mesh, basis, beta_func, gamma_func, t=0.0)
    A = (M + dt * K_drift).tocsc()
    A_lu = splu(A)
    U = project_l2(mesh, basis, u_ex, t=0.0)
    n_steps = int(round(T / dt))
    for n in range(n_steps):
        t_new = (n + 1) * dt
        F_bc = assemble_drift_inflow_bc(mesh, basis, beta_func, g_D_func, t_new)
        U = A_lu.solve(M @ U + dt * F_bc)

    err = l2_error(mesh, basis, U, u_ex, T)

    # u_ex(0.75, 0.5) = sin(π*0.25) ≈ +0.71  →  positive (lobe shifted right)
    # u_ex(0.75, 0.0) = sin(3π/4)  ≈ +0.71  →  still positive initially
    # u_ex(-0.25, 0.5) = sin(-3π/4) < 0  ←  lobe moved away from x=-0.25
    assert _eval_dg_at(mesh, basis, U, 0.75) > 0.3, \
        "β=+1: solution should be positive near x=0.75 at T=0.5 (rightward shift)"
    assert _eval_dg_at(mesh, basis, U, -0.25) < 0.0, \
        "β=+1: solution should be negative near x=-0.25 at T=0.5"
    assert err < 0.05, f"L² error {err:.4e} ≥ 0.05 for β=+1, N=32, p=1"


# ------------------------------------------------------------------ #
# Test 2: Pure advection, β = −1 (solution travels left)
# ------------------------------------------------------------------ #
def test_advection_negative_beta():
    beta = -1.0
    T = 0.5
    dt = 1e-3
    N = 32
    p = 1

    def u_ex(x, t):
        return np.sin(np.pi * (x - beta * t))  # sin(π(x + t))

    g_D_func = lambda x, t: u_ex(np.atleast_1d(np.asarray(x, float)), t)[0]

    mesh, basis = make_mesh_basis(N, p)
    beta_func  = lambda x: np.full_like(np.asarray(x, float), beta)
    gamma_func = lambda x: np.zeros_like(np.asarray(x, float))
    M = assemble_mass_matrix(mesh, basis)
    K_drift = assemble_upwind_drift(mesh, basis, beta_func, gamma_func, t=0.0)
    A = (M + dt * K_drift).tocsc()
    A_lu = splu(A)
    U = project_l2(mesh, basis, u_ex, t=0.0)
    n_steps = int(round(T / dt))
    for n in range(n_steps):
        t_new = (n + 1) * dt
        F_bc = assemble_drift_inflow_bc(mesh, basis, beta_func, g_D_func, t_new)
        U = A_lu.solve(M @ U + dt * F_bc)

    err = l2_error(mesh, basis, U, u_ex, T)

    # u_ex(0.75, 0) = sin(3π/4) > 0  (initially positive)
    # u_ex(0.75, 0.5) = sin(π*1.25) = sin(5π/4) < 0  (lobe shifted left, now negative)
    assert _eval_dg_at(mesh, basis, U, 0.75) < 0.0, \
        "β=-1: solution should be negative near x=0.75 at T=0.5 (leftward shift)"
    assert err < 0.05, f"L² error {err:.4e} ≥ 0.05 for β=−1, N=32, p=1"


# ------------------------------------------------------------------ #
# Test 3: Variable drift with γ correction — EOC ≈ p+1 = 2
# ------------------------------------------------------------------ #
@pytest.mark.parametrize('p', [1])
def test_variable_drift_with_gamma(p):
    # β(x) = sin(πx),  γ(x) = ∂_x β = π cos(πx)
    # u_ex(x,t) = exp(-t) cos(πx)
    # PDE: u_t + β u_x = f,  f = ∂_t u_ex + β ∂_x u_ex
    #    = -exp(-t)cos(πx) + sin(πx)(-π exp(-t) sin(πx))
    #    = -exp(-t)[cos(πx) + π sin²(πx)]

    def beta_func(x):
        return np.sin(np.pi * np.asarray(x, dtype=float))

    def gamma_func(x):
        return np.pi * np.cos(np.pi * np.asarray(x, dtype=float))

    def u_ex(x, t):
        return np.exp(-t) * np.cos(np.pi * x)

    def f_func(x, t):
        return -np.exp(-t) * (np.cos(np.pi * x) + np.pi * np.sin(np.pi * x) ** 2)

    def g_D_func(x, t):
        xa = np.atleast_1d(np.asarray(x, dtype=float))
        return float(np.exp(-t) * np.cos(np.pi * xa[0])) if np.ndim(x) == 0 else u_ex(x, t)

    T  = 0.2
    dt = 1e-4
    N_list = [8, 16, 32]

    errors = []
    for N in N_list:
        err = run_variable_drift(N, p, T, dt, beta_func, gamma_func,
                                  u_ex, f_func, g_D_func)
        errors.append(err)

    errors = np.array(errors)
    h_vals = (X_MAX - X_MIN) / np.array(N_list, dtype=float)
    eoc = np.log(errors[:-1] / errors[1:]) / np.log(h_vals[:-1] / h_vals[1:])

    assert np.all(eoc > p + 0.5), (
        f"p={p}: EOC {np.round(eoc, 3)} not ≈ p+1={p+1} (need >  {p+0.5})"
    )


# ------------------------------------------------------------------ #
# Test 4: Omit γ — confirm divergence (EOC < 0.5 for p=1)
# ------------------------------------------------------------------ #
def test_variable_drift_without_gamma_fails():
    def beta_func(x):
        return np.sin(np.pi * np.asarray(x, dtype=float))

    def gamma_zero(x):
        return np.zeros_like(np.asarray(x, dtype=float))  # γ deliberately omitted

    def u_ex(x, t):
        return np.exp(-t) * np.cos(np.pi * x)

    def f_func(x, t):
        return -np.exp(-t) * (np.cos(np.pi * x) + np.pi * np.sin(np.pi * x) ** 2)

    def g_D_func(x, t):
        xa = np.atleast_1d(np.asarray(x, dtype=float))
        return float(np.exp(-t) * np.cos(np.pi * xa[0])) if np.ndim(x) == 0 else u_ex(x, t)

    p  = 1
    T  = 0.2
    dt = 1e-4
    N_list = [8, 16, 32]

    errors = []
    for N in N_list:
        err = run_variable_drift(N, p, T, dt, beta_func, gamma_zero,
                                  u_ex, f_func, g_D_func)
        errors.append(err)

    errors = np.array(errors)
    h_vals = (X_MAX - X_MIN) / np.array(N_list, dtype=float)
    eoc = np.log(errors[:-1] / errors[1:]) / np.log(h_vals[:-1] / h_vals[1:])

    assert np.any(eoc < 0.5), (
        f"Without γ, EOC should be < 0.5; got {np.round(eoc, 3)}"
    )
