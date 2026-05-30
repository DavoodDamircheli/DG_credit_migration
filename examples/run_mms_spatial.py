"""Spatial convergence study for the SIPG heat equation (MMS).

Manufactured solution:
    u_ex(x, t) = exp(-t) * sin(π(x - x_min) / L),   L = x_max - x_min
    a = 1 (constant diffusivity)
    f(x, t) = u_ex * (-1 + π²/L²)
    g_D = u_ex at both boundaries (both are zero for this choice)

All functions accept numpy arrays so the assembly loop is fully vectorised.

Run with: python examples/run_mms_spatial.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.mesh import Mesh1D
from src.basis import LagrangeBasis
from src.quadrature import gauss_legendre_ref
from src.time_stepper import BackwardEulerSolver

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
