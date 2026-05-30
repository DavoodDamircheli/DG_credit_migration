"""Spatial convergence study for the SIPG heat equation (MMS), plus a combined
diffusion+drift+reaction MMS study introduced in Phase 3.

Heat equation MMS:
    u_ex(x, t) = exp(-t) * sin(π(x - x_min) / L),   L = x_max - x_min
    a = 1,  f(x, t) = u_ex * (-1 + π²/L²)

Diffusion+drift+reaction MMS (Phase 3):
    PDE: u_t - a u_xx + β u_x + r u = f
    a=0.5, β=1.0, r=0.05
    u_ex(x, t) = exp(-t) * sin(π(x - x_min) / L)
    f = u_t + β u_x - a u_xx + r u  (computed analytically)

Run with: python examples/run_mms_spatial.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.sparse.linalg import splu
from src.mesh import Mesh1D
from src.basis import LagrangeBasis
from src.quadrature import gauss_legendre_ref
from src.time_stepper import BackwardEulerSolver
from src.dg_operators import (
    assemble_mass_matrix, assemble_stiffness, assemble_drift_inflow_bc,
    assemble_sipg_diffusion, assemble_load_vector,
)

# ------------------------------------------------------------------
# Problem setup
# ------------------------------------------------------------------
X_MIN, X_MAX = 0.0, 1.0
L      = X_MAX - X_MIN
A_VAL  = 1.0
T_FINAL = 0.5
ETA_C   = 10

# dt and T per polynomial degree.
# dt chosen so time error (O(dt·T)) << coarsest-mesh spatial error O(h^{p+1}).
# p=3 uses a smaller T to cap the number of steps at a manageable level
# (spatial EOC is independent of T for smooth solutions).
DT_BY_P = {1: 1e-4, 2: 1e-5, 3: 1e-6}
T_BY_P  = {1: 0.5,  2: 0.5,  3: 0.05}

# ---- vectorised MMS functions (accept numpy arrays) ----
def u_ex(x, t):
    return np.exp(-t) * np.sin(np.pi * (x - X_MIN) / L)

def du_ex_dx(x, t):
    return np.exp(-t) * (np.pi / L) * np.cos(np.pi * (x - X_MIN) / L)

def f_func(x, t):
    return u_ex(x, t) * (-1.0 + A_VAL * (np.pi / L) ** 2)

def g_D_func(x, t):
    return u_ex(x, t)

def a_func(x):
    return np.ones_like(np.asarray(x, dtype=float))


# ------------------------------------------------------------------
# Error computation
# ------------------------------------------------------------------
def compute_errors(mesh, basis, U, T, p):
    n_quad = 2 * p + 4
    xi_q, w_q = gauss_legendre_ref(n_quad)
    phi_q  = basis.phi(xi_q)
    dphi_q = basis.dphi_dxi(xi_q)
    n_loc  = p + 1

    e_L2_sq = e_DG_sq = 0.0
    for K in range(mesh.N):
        x_L, x_R = mesh.element_interval(K)
        h_K = x_R - x_L
        J   = h_K / 2.0
        x_phys = 0.5 * (x_L + x_R) + 0.5 * h_K * xi_q

        U_K     = U[K * n_loc : (K + 1) * n_loc]
        u_h     = phi_q @ U_K
        du_h_dx = (2.0 / h_K) * (dphi_q @ U_K)

        u_e  = u_ex(x_phys, T)
        du_e = du_ex_dx(x_phys, T)

        e_L2_sq += J * np.dot(w_q, (u_e - u_h) ** 2)
        e_DG_sq += J * np.dot(w_q, (du_e - du_h_dx) ** 2)

    return np.sqrt(e_L2_sq), np.sqrt(e_DG_sq)


# ------------------------------------------------------------------
# Spatial convergence: fixed dt per p, refine N
# ------------------------------------------------------------------
# N range per p: cap where temporal error is still negligible
N_BY_P = {1: [8, 16, 32, 64, 128],
           2: [8, 16, 32, 64, 128],
           3: [8, 16, 32]}
P_LIST = [1, 2, 3]

print("=" * 72)
print("Spatial convergence study (dt and T chosen per p; see header)")
print("=" * 72)

for p in P_LIST:
    dt     = DT_BY_P[p]
    T_p    = T_BY_P[p]
    N_LIST = N_BY_P[p]
    print(f"\np = {p}  (dt = {dt:.0e}, T = {T_p}):")
    print(f"  {'N':>5}  {'h':>8}  {'E_L2':>12}  {'EOC_L2':>8}  {'E_DG':>12}  {'EOC_DG':>8}")
    prev_eL2 = prev_eDG = prev_h = None

    for N in N_LIST:
        mesh   = Mesh1D(X_MIN, X_MAX, N)
        xi_q, _ = gauss_legendre_ref(2 * p + 4)
        basis  = LagrangeBasis(p, xi_q)
        config = {'penalty_C': ETA_C}

        solver  = BackwardEulerSolver(mesh, basis, config)
        history = solver.solve(
            u0_func  = lambda x: u_ex(x, 0.0),
            f_func   = f_func,
            a_func   = a_func,
            g_D_func = g_D_func,
            T        = T_p,
            dt       = dt,
        )
        _, U_T = history[-1]
        eL2, eDG = compute_errors(mesh, basis, U_T, T_p, p)

        h = (X_MAX - X_MIN) / N
        if prev_eL2 is not None:
            eoc_L2 = np.log(prev_eL2 / eL2) / np.log(prev_h / h)
            eoc_DG = np.log(prev_eDG / eDG) / np.log(prev_h / h)
            print(f"  {N:>5}  {h:>8.4f}  {eL2:>12.4e}  {eoc_L2:>8.3f}  {eDG:>12.4e}  {eoc_DG:>8.3f}")
        else:
            print(f"  {N:>5}  {h:>8.4f}  {eL2:>12.4e}  {'—':>8}  {eDG:>12.4e}  {'—':>8}")
        prev_eL2, prev_eDG, prev_h = eL2, eDG, h


# ------------------------------------------------------------------
# Temporal convergence: fixed fine mesh, refine dt  (p=1)
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("Temporal convergence study (N = 128, p = 1)")
print("=" * 72)

p      = 1
N_fine = 128
DT_LIST = [0.1, 0.05, 0.025, 0.0125]

mesh   = Mesh1D(X_MIN, X_MAX, N_fine)
xi_q, _ = gauss_legendre_ref(2 * p + 4)
basis  = LagrangeBasis(p, xi_q)
config = {'penalty_C': ETA_C}

print(f"\n  {'dt':>8}  {'E_L2':>12}  {'EOC':>8}")
prev_eL2 = prev_dt = None

for dt in DT_LIST:
    solver  = BackwardEulerSolver(mesh, basis, config)
    history = solver.solve(
        u0_func  = lambda x: u_ex(x, 0.0),
        f_func   = f_func,
        a_func   = a_func,
        g_D_func = g_D_func,
        T        = T_FINAL,
        dt       = dt,
    )
    _, U_T = history[-1]
    eL2, _ = compute_errors(mesh, basis, U_T, T_FINAL, p)

    if prev_eL2 is not None:
        eoc = np.log(prev_eL2 / eL2) / np.log(prev_dt / dt)
        print(f"  {dt:>8.4f}  {eL2:>12.4e}  {eoc:>8.3f}")
    else:
        print(f"  {dt:>8.4f}  {eL2:>12.4e}  {'—':>8}")
    prev_eL2, prev_dt = eL2, dt


# ======================================================================
# Phase 3 — Combined diffusion + drift + reaction MMS
# PDE: u_t - a u_xx + β u_x + r u = f
# u_ex(x,t) = exp(-t) sin(π(x - X_MIN)/L),  L = X_MAX - X_MIN
# a=0.5, β=1.0, r=0.05
#
# Exact source term (derived analytically):
#   u_t = -u_ex
#   -a u_xx = a (π/L)² u_ex
#   β u_x = β (π/L) exp(-t) cos(π(x-X_MIN)/L)
#   r u = r u_ex
#   f = u_ex(-1 + a(π/L)² + r) + β(π/L) exp(-t) cos(π(x-X_MIN)/L)
# ======================================================================
print("\n" + "=" * 72)
print("Phase 3: Diffusion + Drift + Reaction MMS  (a=0.5, β=1.0, r=0.05)")
print("=" * 72)

A_DDR  = 0.5
BETA   = 1.0
R_DDR  = 0.05
ETA_C3 = 10

def u_ex_ddr(x, t):
    return np.exp(-t) * np.sin(np.pi * (x - X_MIN) / L)

def du_ex_ddr_dx(x, t):
    return np.exp(-t) * (np.pi / L) * np.cos(np.pi * (x - X_MIN) / L)

def f_func_ddr(x, t):
    coef_sin = -1.0 + A_DDR * (np.pi / L) ** 2 + R_DDR
    coef_cos = BETA * (np.pi / L)
    return (coef_sin * np.exp(-t) * np.sin(np.pi * (x - X_MIN) / L)
            + coef_cos * np.exp(-t) * np.cos(np.pi * (x - X_MIN) / L))

def g_D_ddr(x, t):
    return u_ex_ddr(np.atleast_1d(np.asarray(x, float)), t)[0]

def a_func_ddr(x):
    return np.full_like(np.asarray(x, float), A_DDR)

def beta_func_ddr(x):
    return np.full_like(np.asarray(x, float), BETA)

def gamma_func_ddr(x):  # ∂_x β = 0 for constant β
    return np.zeros_like(np.asarray(x, float))


def compute_l2_error_ddr(mesh, basis, U, T, p):
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
        u_e = u_ex_ddr(x_phys, T)
        err_sq += J * np.dot(w_q, (u_e - u_h) ** 2)
    return np.sqrt(err_sq)


def solve_ddr(N, p, dt, T, eta_C):
    """Backward Euler for the combined PDE using assemble_stiffness."""
    mesh  = Mesh1D(X_MIN, X_MAX, N)
    xi_q, _ = gauss_legendre_ref(2 * p + 4)
    basis = LagrangeBasis(p, xi_q)
    n_loc = p + 1
    n_quad = 2 * p + 4
    _, w_q = gauss_legendre_ref(n_quad)
    phi_q = basis.phi(xi_q)

    M = assemble_mass_matrix(mesh, basis)
    K = assemble_stiffness(mesh, basis, a_func_ddr, beta_func_ddr,
                           gamma_func_ddr, R_DDR, eta_C, p, t=0.0)
    A = (M + dt * K).tocsc()
    A_lu = splu(A)

    # Physical quadrature grid (all elements)
    x_L_arr = mesh.x[:-1]; x_R_arr = mesh.x[1:]
    h_K_arr = x_R_arr - x_L_arr; J_K_arr = h_K_arr / 2.0
    x_phys_all = (0.5 * (x_L_arr[:, None] + x_R_arr[:, None])
                  + 0.5 * h_K_arr[:, None] * xi_q[None, :])

    # L²-project initial condition
    U = np.zeros(mesh.N * n_loc)
    for Ke in range(mesh.N):
        x_L, x_R = mesh.element_interval(Ke)
        h_K = x_R - x_L
        J = h_K / 2.0
        x_p = 0.5 * (x_L + x_R) + 0.5 * h_K * xi_q
        u0_q = u_ex_ddr(x_p, 0.0)
        M_loc = J * np.einsum('q,qi,qj->ij', w_q, phi_q, phi_q)
        b_loc = J * np.einsum('q,q,qi->i', w_q, u0_q, phi_q)
        U[Ke * n_loc:(Ke + 1) * n_loc] = np.linalg.solve(M_loc, b_loc)

    eta = eta_C * (p + 1) ** 2
    phi_m1 = basis.phi(np.array([-1.0]))[0]
    phi_p1 = basis.phi(np.array([ 1.0]))[0]

    # Precompute SIPG BC coefficients (Nitsche)
    x_bc_L = float(mesh.faces[0]);   h_bc_L = float(mesh.h[0])
    x_bc_R = float(mesh.faces[N]);   h_bc_R = float(mesh.h[N - 1])
    a_bc_L = float(a_func_ddr(np.array([x_bc_L]))[0])
    a_bc_R = float(a_func_ddr(np.array([x_bc_R]))[0])
    pen_L  = eta * a_bc_L / h_bc_L;  pen_R = eta * a_bc_R / h_bc_R
    dphi_m1 = basis.dphi_dx(np.array([-1.0]), h_bc_L)[0]
    dphi_p1 = basis.dphi_dx(np.array([ 1.0]), h_bc_R)[0]
    coeff_L = -a_bc_L * dphi_m1 - pen_L * phi_m1
    coeff_R =  a_bc_R * dphi_p1 - pen_R * phi_p1

    n_steps = int(round(T / dt))
    for n in range(n_steps):
        t_new = (n + 1) * dt

        # Volume source
        f_all = f_func_ddr(x_phys_all, t_new)
        F = (J_K_arr[:, None]
             * np.einsum('q,Kq,qi->Ki', w_q, f_all, phi_q)).ravel()

        # Nitsche diffusion BC
        g_L = g_D_ddr(x_bc_L, t_new)
        g_R = g_D_ddr(x_bc_R, t_new)
        F[:n_loc]  += coeff_L * g_L
        F[-n_loc:] += coeff_R * g_R

        # Upwind drift inflow BC (β=1>0 → inflow at left, outflow at right)
        F_drift = assemble_drift_inflow_bc(mesh, basis, beta_func_ddr,
                                           g_D_ddr, t_new)
        F += F_drift

        U = A_lu.solve(M @ U + dt * F)

    return mesh, basis, U


# Spatial convergence for p = 1, 2, 3
N_BY_P3  = {1: [8, 16, 32, 64], 2: [8, 16, 32], 3: [8, 16, 32]}
DT_BY_P3 = {1: 1e-4, 2: 1e-5, 3: 1e-6}
T_BY_P3  = {1: 0.5,  2: 0.5,  3: 0.05}

print("\nSpatial convergence (dt fine per p, T fixed)")
for p in [1, 2, 3]:
    dt   = DT_BY_P3[p]
    T_p  = T_BY_P3[p]
    Ns   = N_BY_P3[p]
    print(f"\n  p={p}  (dt={dt:.0e}, T={T_p}):")
    print(f"  {'N':>5}  {'h':>8}  {'E_L2':>12}  {'EOC_L2':>8}")
    prev_eL2 = prev_h = None
    for N in Ns:
        mesh, basis, U_T = solve_ddr(N, p, dt, T_p, ETA_C3)
        eL2 = compute_l2_error_ddr(mesh, basis, U_T, T_p, p)
        h = (X_MAX - X_MIN) / N
        if prev_eL2 is not None:
            eoc = np.log(prev_eL2 / eL2) / np.log(prev_h / h)
            print(f"  {N:>5}  {h:>8.4f}  {eL2:>12.4e}  {eoc:>8.3f}")
        else:
            print(f"  {N:>5}  {h:>8.4f}  {eL2:>12.4e}  {'—':>8}")
        prev_eL2, prev_h = eL2, h

# Temporal convergence: fixed fine mesh (N=128, p=1), refine dt
print("\nTemporal convergence (N=128, p=1, combined PDE)")
p_t = 1
mesh_t, basis_t, _ = solve_ddr(128, p_t, 0.001, 0.5, ETA_C3)  # dummy to warm up
DT_T = [0.1, 0.05, 0.025, 0.0125]
print(f"\n  {'dt':>8}  {'E_L2':>12}  {'EOC':>8}")
prev_eL2 = prev_dt = None
for dt_t in DT_T:
    mesh_t, basis_t, U_T = solve_ddr(128, p_t, dt_t, T_FINAL, ETA_C3)
    eL2 = compute_l2_error_ddr(mesh_t, basis_t, U_T, T_FINAL, p_t)
    if prev_eL2 is not None:
        eoc = np.log(prev_eL2 / eL2) / np.log(prev_dt / dt_t)
        print(f"  {dt_t:>8.4f}  {eL2:>12.4e}  {eoc:>8.3f}")
    else:
        print(f"  {dt_t:>8.4f}  {eL2:>12.4e}  {'—':>8}")
    prev_eL2, prev_dt = eL2, dt_t
