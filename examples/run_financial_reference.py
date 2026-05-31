"""Financial reference run: base solve (N=80, p=2) + high-res reference (N=320, p=3).

Outputs
-------
results/logs/financial_diagnostics.csv
results/logs/boundary_trajectory.csv
results/figures/financial_solution_snapshots.png
results/figures/sigma_eps_snapshots.png
results/figures/H_eps_snapshots.png
results/figures/boundary_trajectory.png
results/figures/kappa_trajectory.png
results/figures/stability_diagnostics.png
results/tables/financial_reference_comparison.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.mesh import Mesh1D
from src.basis import LagrangeBasis
from src.quadrature import gauss_legendre_ref
from src.time_stepper import BackwardEulerSolver, _eval_dg_solution
from src.diagnostics import DiagnosticsRecorder
from src.boundary_tracker import BoundaryTracker, _make_continuous
from src.coefficients import a_eps as _a_eps, sigma_eps as _sigma_eps, H_eps as _H_eps


# ── Ensure output directories exist ───────────────────────────────────────────
for _d in ('results/logs', 'results/figures', 'results/tables'):
    pathlib.Path(_d).mkdir(parents=True, exist_ok=True)


# ── Physical parameters ────────────────────────────────────────────────────────
X_MIN, X_MAX = -2.0, 2.0
T       = 1.0
R       = 0.05
SIGMA_H = 0.4
SIGMA_L = 0.1
EPS     = 0.05
PSI_VAL = 0.5
ETA_C   = 10

# Base discretisation
N_BASE  = 80
P_BASE  = 2
DT_BASE = 0.005

# High-resolution reference
N_REF  = 320
P_REF  = 3
DT_REF = 0.001

N_STEPS_BASE = int(round(T / DT_BASE))   # 200
N_STEPS_REF  = int(round(T / DT_REF))    # 1000


# ── Problem functions ─────────────────────────────────────────────────────────

def u0(x):
    x = np.asarray(x, dtype=float)
    return np.minimum(1.0, np.exp(x))


def Psi_func(x, t):
    return np.full_like(np.asarray(x, dtype=float), PSI_VAL)


def Psi_x_func(x, t):
    return np.zeros_like(np.asarray(x, dtype=float))


def f_func(x, t):
    return np.zeros_like(np.asarray(x, dtype=float))


def g_D_func(x, t):
    return float(u0(np.asarray([x], dtype=float))[0])


# ── Coefficient function factory for DiagnosticsRecorder ─────────────────────

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


# ════════════════════════════════════════════════════════════════════════════════
# BASE SOLVE  (N=80, p=2, dt=0.005)
# ════════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"Base solve  N={N_BASE}, p={P_BASE}, dt={DT_BASE}, T={T}")
print("=" * 70)

mesh_base  = Mesh1D(X_MIN, X_MAX, N_BASE)
xi_b, _    = gauss_legendre_ref(2 * P_BASE + 4)
basis_base = LagrangeBasis(P_BASE, xi_b)

tracker_base = BoundaryTracker(mesh_base, basis_base, Psi_func,
                                n_plot_factor=10)
diag_base    = DiagnosticsRecorder(coeff_func=_make_coeff_fn(EPS))

# Snapshot steps (1-indexed: step 50 = t=0.25, ..., step 200 = t=1.0)
SNAP_STEPS = {round(t_s / DT_BASE) for t_s in (0.25, 0.5, 0.75, 1.0)}

snapshots_base   = {}    # t → U
boundary_base    = []    # list of BoundaryTracker records
s_prev_base      = np.nan


def on_step_base(step, t, U):
    global s_prev_base
    if step in SNAP_STEPS:
        snapshots_base[t] = U.copy()
    bt = tracker_base.track(U, t, s_prev=s_prev_base)
    if not np.isnan(bt['s']):
        s_prev_base = bt['s']
    boundary_base.append(bt)


solver_base = BackwardEulerSolver(mesh_base, basis_base, {'penalty_C': ETA_C})
history_base = solver_base.solve(
    u0_func=u0, f_func=f_func, a_func=None, g_D_func=g_D_func,
    T=T, dt=DT_BASE, r=R,
    Psi_func=Psi_func, Psi_x_func=Psi_x_func,
    sigma_H=SIGMA_H, sigma_L=SIGMA_L, eps=EPS,
    diagnostics=diag_base,
    on_step=on_step_base,
)
U0_base    = history_base[0][1]
U_fin_base = history_base[-1][1]
snapshots_base[0.0] = U0_base

print(f"  Base solve done.  {len(boundary_base)} boundary records.")


# ════════════════════════════════════════════════════════════════════════════════
# REFERENCE SOLVE  (N=320, p=3, dt=0.001)
# ════════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"Reference solve  N={N_REF}, p={P_REF}, dt={DT_REF}, T={T}")
print("  (reassemble_every=5 to manage runtime)")
print("=" * 70)

mesh_ref  = Mesh1D(X_MIN, X_MAX, N_REF)
xi_r, _   = gauss_legendre_ref(2 * P_REF + 4)
basis_ref = LagrangeBasis(P_REF, xi_r)

tracker_ref = BoundaryTracker(mesh_ref, basis_ref, Psi_func,
                               n_plot_factor=5)

# Only track at times matching the base grid (every 5th reference step)
_REF_INTERVAL = round(DT_BASE / DT_REF)   # = 5
boundary_ref  = []
s_prev_ref    = np.nan


def on_step_ref(step, t, U):
    global s_prev_ref
    if step % _REF_INTERVAL == 0:
        bt = tracker_ref.track(U, t, s_prev=s_prev_ref)
        if not np.isnan(bt['s']):
            s_prev_ref = bt['s']
        boundary_ref.append(bt)


solver_ref = BackwardEulerSolver(mesh_ref, basis_ref, {'penalty_C': ETA_C})
history_ref = solver_ref.solve(
    u0_func=u0, f_func=f_func, a_func=None, g_D_func=g_D_func,
    T=T, dt=DT_REF, r=R,
    Psi_func=Psi_func, Psi_x_func=Psi_x_func,
    sigma_H=SIGMA_H, sigma_L=SIGMA_L, eps=EPS,
    reassemble_every=5,
    on_step=on_step_ref,
)
U_fin_ref = history_ref[-1][1]
print(f"  Reference solve done.  {len(boundary_ref)} boundary records.")


# ════════════════════════════════════════════════════════════════════════════════
# REFERENCE COMPARISON METRICS
# ════════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Computing reference comparison metrics...")
print("=" * 70)

n_quad_b = 2 * P_BASE + 4
xi_q_b, w_q_b = gauss_legendre_ref(n_quad_b)
phi_q_b = basis_base.phi(xi_q_b)
n_loc_b = P_BASE + 1

# E_L2_ref = ‖u_ref(·,T) - U_h^N(·,T)‖_{L²}  on base mesh quadrature
E_L2_sq = 0.0
for K in range(N_BASE):
    xL, xR = mesh_base.element_interval(K)
    hK = xR - xL; J = hK / 2.0
    x_pts = 0.5 * (xL + xR) + 0.5 * hK * xi_q_b
    u_ref_vals  = _eval_dg_solution(mesh_ref, basis_ref, U_fin_ref,  x_pts)
    u_base_vals = phi_q_b @ U_fin_base[K * n_loc_b:(K + 1) * n_loc_b]
    E_L2_sq += J * np.dot(w_q_b, (u_ref_vals - u_base_vals) ** 2)
E_L2_ref = float(np.sqrt(E_L2_sq))

# E_Linf_ref = ‖u_ref(·,T) - R_h U_h^N(·,T)‖_{L∞}  on a fine plotting grid
x_fine_cmp  = np.linspace(X_MIN, X_MAX, 5000)
U_cont_base = _make_continuous(mesh_base, basis_base, U_fin_base)
Rh_base     = _eval_dg_solution(mesh_base, basis_base, U_cont_base, x_fine_cmp)
u_ref_fine  = _eval_dg_solution(mesh_ref,  basis_ref,  U_fin_ref,   x_fine_cmp)
E_Linf_ref  = float(np.max(np.abs(u_ref_fine - Rh_base)))

# E_s_ref = max_n |s_ref(t_n) - s_base(t_n)|
ref_s_dict  = {round(bt['t'], 6): bt['s'] for bt in boundary_ref}
base_s_dict = {round(bt['t'], 6): bt['s'] for bt in boundary_base}
s_diffs = []
for t_n, s_b in base_s_dict.items():
    s_r = ref_s_dict.get(t_n, np.nan)
    if not (np.isnan(s_b) or np.isnan(s_r)):
        s_diffs.append(abs(s_r - s_b))
E_s_ref = float(max(s_diffs)) if s_diffs else float('nan')

print(f"  E_L2_ref   = {E_L2_ref:.4e}")
print(f"  E_Linf_ref = {E_Linf_ref:.4e}")
print(f"  E_s_ref    = {E_s_ref:.4e}")

with open('results/tables/financial_reference_comparison.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=['metric', 'value'])
    wr.writeheader()
    wr.writerows([
        {'metric': 'E_L2_ref',   'value': E_L2_ref},
        {'metric': 'E_Linf_ref', 'value': E_Linf_ref},
        {'metric': 'E_s_ref',    'value': E_s_ref},
    ])
print("  → results/tables/financial_reference_comparison.csv")


# ════════════════════════════════════════════════════════════════════════════════
# SAVE LOG FILES
# ════════════════════════════════════════════════════════════════════════════════
diag_base.save_csv('results/logs/financial_diagnostics.csv')
print("  → results/logs/financial_diagnostics.csv")

bt_fields = ['t', 's', 'kappa', 'n_roots']
with open('results/logs/boundary_trajectory.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=bt_fields)
    wr.writeheader()
    for bt in boundary_base:
        wr.writerow({k: bt[k] for k in bt_fields})
print("  → results/logs/boundary_trajectory.csv")

# Verify no consecutive NaN in boundary trajectory
s_vals = [bt['s'] for bt in boundary_base]
for i in range(len(s_vals) - 1):
    if np.isnan(s_vals[i]) and np.isnan(s_vals[i + 1]):
        print(f"  WARNING: consecutive NaN in boundary at steps {i+1}, {i+2}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Generating figures...")
print("=" * 70)

SNAP_TIMES_SORTED = sorted(snapshots_base.keys())
x_plt = np.linspace(X_MIN, X_MAX, 500)
_clrs = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']


# ── Figure 1: Solution snapshots ──────────────────────────────────────────────
fig1, axes = plt.subplots(1, len(SNAP_TIMES_SORTED), figsize=(3.5 * len(SNAP_TIMES_SORTED), 4),
                           sharey=True)
for ax, (t_s, clr) in zip(axes, zip(SNAP_TIMES_SORTED, _clrs)):
    U_s = snapshots_base[t_s]
    u_h = _eval_dg_solution(mesh_base, basis_base, U_s, x_plt)
    ax.plot(x_plt, u_h, color=clr, lw=1.5, label=r'$U_h$')
    ax.axhline(PSI_VAL, color='k', ls='--', lw=1.0, label=r'$\Psi{=}0.5$')
    ax.set_title(f't = {t_s:.2f}'); ax.set_xlabel('x')
    ax.legend(fontsize=8)
axes[0].set_ylabel('u')
fig1.suptitle(f'Financial model: solution snapshots (N={N_BASE}, p={P_BASE}, ε={EPS})')
fig1.tight_layout()
fig1.savefig('results/figures/financial_solution_snapshots.png', dpi=150)
plt.close(fig1)
print("  → results/figures/financial_solution_snapshots.png")


# ── Figure 2: σ_ε snapshots ───────────────────────────────────────────────────
fig2, axes = plt.subplots(1, len(SNAP_TIMES_SORTED), figsize=(3.5 * len(SNAP_TIMES_SORTED), 4),
                           sharey=True)
for ax, (t_s, clr) in zip(axes, zip(SNAP_TIMES_SORTED, _clrs)):
    U_s  = snapshots_base[t_s]
    u_h  = _eval_dg_solution(mesh_base, basis_base, U_s, x_plt)
    sig  = _sigma_eps(u_h, x_plt, t_s, Psi_func, SIGMA_H, SIGMA_L, EPS)
    ax.plot(x_plt, sig, color=clr, lw=1.5)
    ax.set_title(f't = {t_s:.2f}'); ax.set_xlabel('x')
axes[0].set_ylabel(r'$\sigma_\varepsilon$')
fig2.suptitle(r'Financial model: $\sigma_\varepsilon(U_h, x, t)$ snapshots')
fig2.tight_layout()
fig2.savefig('results/figures/sigma_eps_snapshots.png', dpi=150)
plt.close(fig2)
print("  → results/figures/sigma_eps_snapshots.png")


# ── Figure 3: H_ε snapshots ───────────────────────────────────────────────────
fig3, axes = plt.subplots(1, len(SNAP_TIMES_SORTED), figsize=(3.5 * len(SNAP_TIMES_SORTED), 4),
                           sharey=True)
for ax, (t_s, clr) in zip(axes, zip(SNAP_TIMES_SORTED, _clrs)):
    U_s = snapshots_base[t_s]
    u_h = _eval_dg_solution(mesh_base, basis_base, U_s, x_plt)
    Psi_vals = np.full_like(x_plt, PSI_VAL)
    h_e = _H_eps(u_h - Psi_vals, EPS)
    ax.plot(x_plt, h_e, color=clr, lw=1.5)
    ax.set_title(f't = {t_s:.2f}'); ax.set_xlabel('x')
axes[0].set_ylabel(r'$H_\varepsilon(U_h - \Psi)$')
fig3.suptitle(r'Financial model: $H_\varepsilon(U_h - \Psi)$ snapshots')
fig3.tight_layout()
fig3.savefig('results/figures/H_eps_snapshots.png', dpi=150)
plt.close(fig3)
print("  → results/figures/H_eps_snapshots.png")


# ── Figure 4: Boundary trajectory ─────────────────────────────────────────────
t_bt = [bt['t'] for bt in boundary_base]
s_bt = [bt['s'] for bt in boundary_base]
fig4, ax4 = plt.subplots(figsize=(8, 4))
ax4.plot(t_bt, s_bt, 'b-', lw=1.5)
ax4.set_xlabel('t'); ax4.set_ylabel(r'$s_{h,\varepsilon}(t)$')
ax4.set_title(f'Boundary trajectory  (N={N_BASE}, p={P_BASE}, ε={EPS})')
ax4.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig('results/figures/boundary_trajectory.png', dpi=150)
plt.close(fig4)
print("  → results/figures/boundary_trajectory.png")


# ── Figure 5: Transversality κ trajectory ─────────────────────────────────────
k_bt = [bt['kappa'] for bt in boundary_base]
fig5, ax5 = plt.subplots(figsize=(8, 4))
ax5.semilogy(t_bt, k_bt, 'r-', lw=1.5)
ax5.set_xlabel('t'); ax5.set_ylabel(r'$\kappa_h^n$')
ax5.set_title(r'Transversality $\kappa_h^n$ vs $t$')
ax5.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig('results/figures/kappa_trajectory.png', dpi=150)
plt.close(fig5)
print("  → results/figures/kappa_trajectory.png")


# ── Figure 6: Stability diagnostics ───────────────────────────────────────────
t_diag  = [r['t']      for r in diag_base.records]
l2_diag = [r['L2_norm'] for r in diag_base.records]
umin    = [r['u_min']  for r in diag_base.records]
umax    = [r['u_max']  for r in diag_base.records]

fig6, (ax6a, ax6b) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax6a.plot(t_diag, l2_diag, 'b-', lw=1.5)
ax6a.set_ylabel(r'$\|U_h^n\|_{L^2}$')
ax6a.set_title('Stability diagnostics')
ax6a.grid(True, alpha=0.3)
ax6b.plot(t_diag, umin, 'b-', lw=1.2, label='min')
ax6b.plot(t_diag, umax, 'r-', lw=1.2, label='max')
ax6b.set_xlabel('t')
ax6b.set_ylabel(r'$\min / \max\; U_h^n$')
ax6b.legend(); ax6b.grid(True, alpha=0.3)
fig6.tight_layout()
fig6.savefig('results/figures/stability_diagnostics.png', dpi=150)
plt.close(fig6)
print("  → results/figures/stability_diagnostics.png")

print()
print("Done.  All output files written.")
