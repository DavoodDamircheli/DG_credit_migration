"""Picard vs frozen-coefficient comparison.

Verifies that PicardSolver matches BackwardEulerSolver (frozen) at small dt
and measures Picard iteration counts for the MMS and financial problems.

Outputs
-------
results/tables/picard_vs_frozen.csv   — MMS comparison table
stdout                                — financial-model iteration report
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv, pathlib
import numpy as np

from src.mesh import Mesh1D
from src.basis import LagrangeBasis
from src.quadrature import gauss_legendre_ref
from src.time_stepper import (BackwardEulerSolver, PicardSolver, NewtonSolver,
                               _eval_dg_solution)
from src.coefficients import a_eps as _a_eps


pathlib.Path('results/tables').mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# MMS problem  (same parameters as Phase 5)
# ══════════════════════════════════════════════════════════════════════════════
X_MIN, X_MAX = -2.0, 2.0
L       = X_MAX - X_MIN
T_MMS   = 0.5
R       = 0.05
SIGMA_H = 0.4
SIGMA_L = 0.1
EPS     = 0.05
ETA_C   = 10
AMP_MMS = 0.1          # amplitude well below Psi=0.5

Psi_func   = lambda x, t: np.full_like(np.asarray(x, float), 0.5)
Psi_x_func = lambda x, t: np.zeros_like(np.asarray(x, float))


def u_ex(x, t):
    return AMP_MMS * np.exp(-t) * np.sin(np.pi * (x - X_MIN) / L)

def u_ex_x(x, t):
    return AMP_MMS * np.exp(-t) * (np.pi / L) * np.cos(np.pi * (x - X_MIN) / L)

def u_ex_xx(x, t):
    return -AMP_MMS * np.exp(-t) * (np.pi / L)**2 * np.sin(np.pi * (x - X_MIN) / L)

def f_mms(x, t):
    xa = np.asarray(x, float)
    z  = u_ex(xa, t)
    a  = _a_eps(z, xa, t, Psi_func, SIGMA_H, SIGMA_L, EPS)
    return -z - a * u_ex_xx(xa, t) - (R - a) * u_ex_x(xa, t) + R * z

def g_D_mms(x, t):
    return 0.0   # u_ex = 0 at both boundaries

def u0_mms(x):
    return u_ex(x, 0.0)


def compute_l2_error(mesh, basis, U, t_val):
    p = basis.p; n_loc = p + 1; N = mesh.N
    nq = 2 * p + 4
    xi, wq = gauss_legendre_ref(nq)
    phi_q = basis.phi(xi)
    sq = 0.0
    for K in range(N):
        xL, xR = mesh.element_interval(K)
        hK = xR - xL; J = hK / 2.0
        xp = 0.5 * (xL + xR) + 0.5 * hK * xi
        uh = phi_q @ U[K * n_loc:(K + 1) * n_loc]
        sq += J * np.dot(wq, (u_ex(xp, t_val) - uh)**2)
    return float(np.sqrt(sq))


# ══════════════════════════════════════════════════════════════════════════════
# Comparison: dt sweep  (N=256, p=1)
# ══════════════════════════════════════════════════════════════════════════════
N_CMP  = 256
P_CMP  = 1
DT_LIST = [0.1, 0.05, 0.025, 0.0125]

print("=" * 88)
print("Frozen / Picard / Newton — MMS problem  (N=256, p=1, T=0.5)")
print("=" * 88)
print(f"{'dt':>8}  {'E_L2 frozen':>12}  {'E_L2 Picard':>12}  {'E_L2 Newton':>12}  "
      f"{'P iters':>8}  {'N iters':>8}")

rows = []

for dt in DT_LIST:
    mesh  = Mesh1D(X_MIN, X_MAX, N_CMP)
    xi_q, _ = gauss_legendre_ref(2 * P_CMP + 4)
    basis = LagrangeBasis(P_CMP, xi_q)
    cfg   = {'penalty_C': ETA_C}

    # ── Frozen solve ──────────────────────────────────────────────────
    solver_f = BackwardEulerSolver(mesh, basis, cfg)
    _, U_frz = solver_f.solve(
        u0_func=u0_mms, f_func=f_mms, a_func=None, g_D_func=g_D_mms,
        T=T_MMS, dt=dt, r=R,
        Psi_func=Psi_func, Psi_x_func=Psi_x_func,
        sigma_H=SIGMA_H, sigma_L=SIGMA_L, eps=EPS,
    )[-1]
    e_frz = compute_l2_error(mesh, basis, U_frz, T_MMS)

    # ── Picard solve ──────────────────────────────────────────────────
    solver_p = PicardSolver(mesh, basis, cfg)
    pstats   = []
    _, U_pic = solver_p.solve(
        u0_func=u0_mms, f_func=f_mms, g_D_func=g_D_mms,
        T=T_MMS, dt=dt, r=R,
        Psi_func=Psi_func, Psi_x_func=Psi_x_func,
        sigma_H=SIGMA_H, sigma_L=SIGMA_L, eps=EPS,
        picard_stats=pstats,
    )[-1]
    e_pic      = compute_l2_error(mesh, basis, U_pic, T_MMS)
    avg_p_iter = float(np.mean([s['n_iters'] for s in pstats]))

    # ── Newton solve ──────────────────────────────────────────────────
    solver_n = NewtonSolver(mesh, basis, cfg)
    nstats   = []
    _, U_nwt = solver_n.solve(
        u0_func=u0_mms, f_func=f_mms, g_D_func=g_D_mms,
        T=T_MMS, dt=dt, r=R,
        Psi_func=Psi_func, Psi_x_func=Psi_x_func,
        sigma_H=SIGMA_H, sigma_L=SIGMA_L, eps=EPS,
        newton_stats=nstats,
    )[-1]
    e_nwt      = compute_l2_error(mesh, basis, U_nwt, T_MMS)
    avg_n_iter = float(np.mean([s['n_iters'] for s in nstats]))

    print(f"{dt:>8.4f}  {e_frz:>12.4e}  {e_pic:>12.4e}  {e_nwt:>12.4e}  "
          f"{avg_p_iter:>8.2f}  {avg_n_iter:>8.2f}")

    rows.append({
        'dt':               dt,
        'E_L2_frozen':      e_frz,
        'E_L2_picard':      e_pic,
        'E_L2_newton':      e_nwt,
        'avg_picard_iters': avg_p_iter,
        'avg_newton_iters': avg_n_iter,
        'picard_all_conv':  all(s['converged'] for s in pstats),
        'newton_all_conv':  all(s['converged'] for s in nstats),
    })

fields = ['dt', 'E_L2_frozen', 'E_L2_picard', 'E_L2_newton',
          'avg_picard_iters', 'avg_newton_iters',
          'picard_all_conv', 'newton_all_conv']
with open('results/tables/picard_vs_frozen.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=fields)
    wr.writeheader()
    wr.writerows(rows)
print(f"\n  → results/tables/picard_vs_frozen.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Financial-model Picard iteration count check
# (run T=0.1 so it is fast; dt=0.005 = default)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 74)
print("Financial model Picard iteration check  (N=80, p=2, dt=0.005, T=0.1)")
print("=" * 74)

def u0_fin(x):
    x = np.asarray(x, dtype=float)
    return np.minimum(1.0, np.exp(x))

def f_fin(x, t):
    return np.zeros_like(np.asarray(x, dtype=float))

def g_D_fin(x, t):
    return float(u0_fin(np.asarray([x], dtype=float))[0])

Psi_fin   = lambda x, t: np.full_like(np.asarray(x, float), 0.5)
Psi_x_fin = lambda x, t: np.zeros_like(np.asarray(x, float))

mesh_f  = Mesh1D(-2.0, 2.0, 80)
xi_f, _ = gauss_legendre_ref(2 * 2 + 4)
basis_f = LagrangeBasis(2, xi_f)

print()
print("  --- Picard ---")
solver_pic_fin = PicardSolver(mesh_f, basis_f, {'penalty_C': 10})
fstats_pic     = []
solver_pic_fin.solve(
    u0_func=u0_fin, f_func=f_fin, g_D_func=g_D_fin,
    T=0.1, dt=0.005, r=0.05,
    Psi_func=Psi_fin, Psi_x_func=Psi_x_fin,
    sigma_H=0.4, sigma_L=0.1, eps=0.05,
    picard_stats=fstats_pic,
)
avg_pic_fin = float(np.mean([s['n_iters'] for s in fstats_pic]))
max_pic_fin = int(max([s['n_iters'] for s in fstats_pic]))
print(f"  Avg iters   : {avg_pic_fin:.2f}  Max: {max_pic_fin}  "
      f"Conv: {all(s['converged'] for s in fstats_pic)}")

print()
print("  --- Newton (analytical diffusion Jacobian, norm-based step safeguard) ---")
solver_nwt_fin = NewtonSolver(mesh_f, basis_f, {'penalty_C': 10})
fstats_nwt     = []
solver_nwt_fin.solve(
    u0_func=u0_fin, f_func=f_fin, g_D_func=g_D_fin,
    T=0.1, dt=0.005, r=0.05,
    Psi_func=Psi_fin, Psi_x_func=Psi_x_fin,
    sigma_H=0.4, sigma_L=0.1, eps=0.05,
    newton_stats=fstats_nwt,
)
avg_nwt_fin = float(np.mean([s['n_iters'] for s in fstats_nwt]))
max_nwt_fin = int(max([s['n_iters'] for s in fstats_nwt]))
all_nwt_fin = all(s['converged'] for s in fstats_nwt)
print(f"  Avg iters   : {avg_nwt_fin:.2f}  Max: {max_nwt_fin}  Conv: {all_nwt_fin}")

if max_nwt_fin <= 5:
    print()
    print("  Newton converges in ≤ 5 iterations — Done when criterion met.")
else:
    print()
    print(f"  Newton still needs {max_nwt_fin} iters max on this problem.")
    print("  The strong nonlinearity near the free boundary (H'_eps large)")
    print("  causes the problem — drift and reaction Jacobian corrections")
    print("  would further improve convergence but are omitted here.")
