"""Epsilon sensitivity sweep for the financial model.

For each ε ∈ {0.1, 0.05, 0.01, 0.005} automatically enforces h ≤ 0.5 ε
by setting N = ceil(L / (0.5 ε)).

Outputs
-------
results/tables/eps_sweep.csv
results/figures/eps_sensitivity_boundary.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv, math, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.mesh import Mesh1D
from src.basis import LagrangeBasis
from src.quadrature import gauss_legendre_ref
from src.time_stepper import BackwardEulerSolver, _eval_dg_solution
from src.diagnostics import DiagnosticsRecorder
from src.boundary_tracker import BoundaryTracker
from src.coefficients import a_eps as _a_eps


pathlib.Path('results/tables').mkdir(parents=True, exist_ok=True)
pathlib.Path('results/figures').mkdir(parents=True, exist_ok=True)


# ── Fixed parameters ──────────────────────────────────────────────────────────
X_MIN, X_MAX = -2.0, 2.0
L       = X_MAX - X_MIN   # 4.0
T       = 1.0
R       = 0.05
SIGMA_H = 0.4
SIGMA_L = 0.1
ETA_C   = 10
P       = 2
DT      = 0.01   # 100 steps; manageable for large N

EPS_LIST = [0.1, 0.05, 0.01, 0.005]


def u0(x):
    x = np.asarray(x, dtype=float)
    return np.minimum(1.0, np.exp(x))


def Psi_func(x, t):
    return np.full_like(np.asarray(x, dtype=float), 0.5)


def Psi_x_func(x, t):
    return np.zeros_like(np.asarray(x, dtype=float))


def f_func(x, t):
    return np.zeros_like(np.asarray(x, dtype=float))


def g_D_func(x, t):
    return float(u0(np.asarray([x], dtype=float))[0])


def _make_coeff_fn(eps):
    def cfn(U, m, b, t):
        nq     = 2 * b.p + 4
        xi_t, _ = gauss_legendre_ref(nq)
        xL = m.x[:-1]; xR = m.x[1:]; hK = xR - xL
        xall = (0.5 * (xL[:, None] + xR[:, None])
                + 0.5 * hK[:, None] * xi_t[None, :]).ravel()
        z = _eval_dg_solution(m, b, U, xall)
        return _a_eps(z, xall, t, Psi_func, SIGMA_H, SIGMA_L, eps)
    return cfn


# ── Sweep ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"Epsilon sweep  (p={P}, T={T}, dt={DT})")
print("=" * 70)
print(f"{'eps':>8}  {'N':>6}  {'h':>8}  {'h/eps':>7}  {'s_T':>10}  "
      f"{'max_resid':>12}  {'max_coeff_chg':>14}  {'kappa_T':>10}")

rows = []
s_final_list = []

for eps in EPS_LIST:
    # Enforce h ≤ 0.5 * eps
    N = int(math.ceil(L / (0.5 * eps)))
    h = L / N
    h_over_eps = h / eps

    mesh  = Mesh1D(X_MIN, X_MAX, N)
    xi_q, _ = gauss_legendre_ref(2 * P + 4)
    basis = LagrangeBasis(P, xi_q)

    diag    = DiagnosticsRecorder(coeff_func=_make_coeff_fn(eps))
    tracker = BoundaryTracker(mesh, basis, Psi_func, n_plot_factor=5)

    boundary  = []
    s_holder  = [np.nan]   # mutable so the closure can update it

    def on_step(step, t, U,
                _tracker=tracker, _boundary=boundary, _sh=s_holder):
        bt = _tracker.track(U, t, s_prev=_sh[0])
        if not np.isnan(bt['s']):
            _sh[0] = bt['s']
        _boundary.append(bt)

    solver = BackwardEulerSolver(mesh, basis, {'penalty_C': ETA_C})
    solver.solve(
        u0_func=u0, f_func=f_func, a_func=None, g_D_func=g_D_func,
        T=T, dt=DT, r=R,
        Psi_func=Psi_func, Psi_x_func=Psi_x_func,
        sigma_H=SIGMA_H, sigma_L=SIGMA_L, eps=eps,
        diagnostics=diag,
        on_step=on_step,
    )

    # Extract summary metrics
    s_T = boundary[-1]['s'] if boundary else float('nan')
    kappa_T = boundary[-1]['kappa'] if boundary else float('nan')

    resid_vals = [r['residual_norm'] for r in diag.records
                  if not np.isnan(r['residual_norm'])]
    max_resid  = max(resid_vals) if resid_vals else float('nan')

    coeff_vals = [r['coeff_change'] for r in diag.records
                  if not np.isnan(r['coeff_change'])]
    max_coeff  = max(coeff_vals) if coeff_vals else float('nan')

    print(f"{eps:>8.4f}  {N:>6d}  {h:>8.5f}  {h_over_eps:>7.4f}  "
          f"{s_T:>10.6f}  {max_resid:>12.4e}  {max_coeff:>14.4e}  {kappa_T:>10.4e}")

    rows.append({
        'eps': eps, 'N': N, 'h': h, 'h/eps': h_over_eps,
        's_T': s_T, 'max_residual': max_resid,
        'max_coeff_change': max_coeff, 'kappa_T': kappa_T,
    })
    s_final_list.append(s_T)


# ── Save table ────────────────────────────────────────────────────────────────
fields = ['eps', 'N', 'h', 'h/eps', 's_T', 'max_residual', 'max_coeff_change', 'kappa_T']
with open('results/tables/eps_sweep.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=fields)
    wr.writeheader()
    wr.writerows(rows)
print("\n  → results/tables/eps_sweep.csv")


# ── Figure: s_{h,ε}(T) vs ε ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(EPS_LIST, s_final_list, 'o-', color='tab:blue', lw=1.5, ms=6)
ax.set_xlabel(r'$\varepsilon$')
ax.set_ylabel(r'$s_{h,\varepsilon}(T)$')
ax.set_title(r'Boundary $s_{h,\varepsilon}(T)$ vs $\varepsilon$')
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig('results/figures/eps_sensitivity_boundary.png', dpi=150)
plt.close(fig)
print("  → results/figures/eps_sensitivity_boundary.png")
print("\nDone.")
