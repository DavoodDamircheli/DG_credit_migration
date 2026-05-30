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


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Full MMS verification for the regularized PDE
# Invoked via:  python examples/run_mms_spatial.py --config config/mms_p1.yaml
# ══════════════════════════════════════════════════════════════════════════════

def _parse_simple_yaml(path):
    """Fallback YAML parser for the simple key: value config format."""
    import re
    result = {}
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip(); val = val.strip()
            m = re.match(r'\[(.+)\]', val)
            if m:
                result[key] = [float(x.strip()) for x in m.group(1).split(',')]
                continue
            try:
                result[key] = int(val) if '.' not in val else float(val)
            except ValueError:
                result[key] = val
    return result


def _csv_write(path, fieldnames, rows):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _phase5_main(config_path):
    """Three convergence tables + three figures for the regularized PDE MMS."""
    import pathlib

    try:
        import yaml
        with open(config_path) as _f:
            cfg = yaml.safe_load(_f)
    except ImportError:
        cfg = _parse_simple_yaml(config_path)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _has_mpl = True
    except ImportError:
        _has_mpl = False

    from src.coefficients import a_eps as _a_eps
    from src.time_stepper import _eval_dg_solution, BackwardEulerSolver as _BESolver
    from src.diagnostics import DiagnosticsRecorder

    # ── Physical parameters ───────────────────────────────────────────────
    X_MIN, X_MAX = cfg['domain']
    L     = X_MAX - X_MIN        # 4.0
    R     = cfg['r']             # 0.05
    SH    = cfg['sigma_H']       # 0.4
    SL    = cfg['sigma_L']       # 0.1
    EPS_D = cfg['eps']           # 0.05 (default; varied in Table 3)
    ETA_C = cfg.get('penalty_C', 10)
    PSI_C = 0.5

    Psi_f  = lambda x, t: np.full_like(np.asarray(x, float), PSI_C)
    Psi_xf = lambda x, t: np.zeros_like(np.asarray(x, float))

    # ── Manufactured solution ─────────────────────────────────────────────
    # u_ex = 0.1 exp(-t) sin(π(x-X_MIN)/L)
    #
    # Zero Dirichlet BCs at both ends (g_D = 0 everywhere).
    # Amplitude 0.1 keeps the state in (−0.1, 0.1), well below Ψ = 0.5.
    # In this saturated regime H_ε(z−Ψ) ≈ 0 and da_ε/dz ≈ 0, so the
    # Picard linearization error is negligible and the frozen-coefficient
    # solver reproduces the expected p+1 spatial convergence rate.
    # The physical parameters (σ_H, σ_L, ε, r) are still exercised through
    # the assembled a_ε ≈ 0.5·σ_H² and β_ε ≈ −(r − a_ε) coefficients.
    AMP_MMS = 0.1
    def u_ex(x, t):
        return AMP_MMS * np.exp(-t) * np.sin(np.pi * (x - X_MIN) / L)

    def u_ex_x(x, t):
        return AMP_MMS * np.exp(-t) * (np.pi / L) * np.cos(np.pi * (x - X_MIN) / L)

    def u_ex_xx(x, t):
        return -AMP_MMS * np.exp(-t) * (np.pi / L) ** 2 * np.sin(np.pi * (x - X_MIN) / L)

    def make_f(eps):
        """Source term: f = u_t - a_ε u_xx - (r - a_ε) u_x + r u.
        Derived analytically — the da_ε/dz · (u_x)² terms cancel exactly.
        Accepts x of any shape (1-D or 2-D batch).
        """
        def f(x, t):
            xa = np.asarray(x, float)
            z  = u_ex(xa, t)
            a  = _a_eps(z, xa, t, Psi_f, SH, SL, eps)
            return (-z                            # u_t = -u_ex
                    - a * u_ex_xx(xa, t)
                    - (R - a) * u_ex_x(xa, t)
                    + R * z)
        return f

    def g_D(x, t):
        return 0.0   # u_ex = 0 at both boundaries for all t

    # ── L2 and DG-energy errors ───────────────────────────────────────────
    def errs(mesh, basis, U, T_val):
        p = basis.p; n_loc = p + 1; N = mesh.N
        nq = 2 * p + 4
        xi, wq = gauss_legendre_ref(nq)
        phi_q  = basis.phi(xi)
        dphi_q = basis.dphi_dxi(xi)
        L2_sq = DG_sq = 0.0
        for K in range(N):
            xL, xR = mesh.element_interval(K)
            hK = xR - xL; J = hK / 2
            xp = 0.5 * (xL + xR) + 0.5 * hK * xi
            UK = U[K * n_loc:(K + 1) * n_loc]
            L2_sq += J * np.dot(wq, (u_ex(xp, T_val) - phi_q @ UK) ** 2)
            DG_sq += J * np.dot(wq, (u_ex_x(xp, T_val)
                                     - (2/hK) * (dphi_q @ UK)) ** 2)
        return np.sqrt(L2_sq), np.sqrt(DG_sq)

    # ── Solver wrapper ────────────────────────────────────────────────────
    # For Tables 1 & 2: the MMS state (AMP=0.1) is always well below Ψ=0.5
    # so da_ε/dz ≈ 0 and the stiffness barely changes between steps.
    # Using reassemble_every=10 reduces assembly cost by 10× with negligible
    # accuracy loss (Picard change per 10 steps ≈ 10·dt·|da/dz|·|u_t| ≈ 0).
    # Table 3 uses reassemble_every=1 so coeff_change is computed correctly.
    def run(N, p, dt, T_val, eps, diagnostics=None, reassemble=10):
        mesh  = Mesh1D(X_MIN, X_MAX, N)
        xi_b, _ = gauss_legendre_ref(2 * p + 4)
        basis = LagrangeBasis(p, xi_b)
        slvr  = _BESolver(mesh, basis, {'penalty_C': ETA_C})
        _, U_T = slvr.solve(
            u0_func=lambda x: u_ex(x, 0.0), f_func=make_f(eps),
            a_func=None, g_D_func=g_D, T=T_val, dt=dt,
            r=R, Psi_func=Psi_f, Psi_x_func=Psi_xf,
            sigma_H=SH, sigma_L=SL, eps=eps,
            diagnostics=diagnostics,
            reassemble_every=reassemble,
        )[-1]
        return mesh, basis, U_T

    # ── coeff_func factory for DiagnosticsRecorder ────────────────────────
    def make_coeff_fn(eps):
        def cfn(U, m, b, t):
            nq = 2 * b.p + 4
            xi_t, _ = gauss_legendre_ref(nq)
            xLa = m.x[:-1]; xRa = m.x[1:]; hKa = xRa - xLa
            xall = (0.5 * (xLa[:, None] + xRa[:, None])
                    + 0.5 * hKa[:, None] * xi_t[None, :]).ravel()
            z = _eval_dg_solution(m, b, U, xall)
            return _a_eps(z, xall, t, Psi_f, SH, SL, eps)
        return cfn

    # ── Output dirs ───────────────────────────────────────────────────────
    pathlib.Path('results/tables').mkdir(parents=True, exist_ok=True)
    pathlib.Path('results/figures').mkdir(parents=True, exist_ok=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Table 1 — Spatial convergence
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("=" * 72)
    print(f"Table 1 — Spatial convergence  (T=0.5, eps={EPS_D}, dt per p)")
    print("=" * 72)

    # Per-p dt: temporal ≈ T·dt/2·|u_tt| << spatial O(h^{p+1}) at all N.
    # With reassemble_every=100 (safe since da_ε/dz≈0 for AMP=0.1 below Ψ),
    # n_assemblies = n_steps/100, keeping total runtime under 10 s.
    #   p=1: dt=1e-4  → temporal≈2.5e-6  << h^2 at N=64 (3.9e-3)  ✓
    #   p=2: dt=1e-4  → temporal≈2.5e-6  << h^3 at N=8  (1.95e-2) ✓
    #                    at N=32: spatial≈5.6e-7, temporal=2.5e-6 →  use smaller dt
    #         dt=2e-5  → temporal≈5e-7    < h^3 at N=32  (7.6e-4) ✓
    #   p=3: dt=5e-6, T=0.1 → temporal≈1.2e-8 << h^4 at N=16 (3.9e-5) ✓
    N_BY_P  = {1: [8, 16, 32, 64], 2: [8, 16, 32], 3: [8, 16]}
    DT_BY_P = {1: 1e-4,            2: 1e-6,        3: 5e-6}
    T_BY_P  = {1: 0.5,             2: 0.5,         3: 0.1}
    _REASSEMBLE_T1 = 100   # safe for AMP=0.1, da/dz≈0 MMS
    t1_rows = []; t1_plot = {}

    for p in [1, 2, 3]:
        DT1 = DT_BY_P[p]; T1p = T_BY_P[p]
        print(f"\n  p = {p}  (dt={DT1:.0e}, T={T1p}):")
        print(f"  {'N':>5}  {'h':>8}  {'E_L2':>12}  {'EOC_L2':>8}"
              f"  {'E_DG':>12}  {'EOC_DG':>8}")
        prev_eL2 = prev_eDG = prev_h = None
        t1_plot[p] = {'h': [], 'eL2': [], 'eDG': []}

        for N in N_BY_P[p]:
            mesh, basis, U_T = run(N, p, DT1, T1p, EPS_D,
                                   reassemble=_REASSEMBLE_T1)
            eL2, eDG = errs(mesh, basis, U_T, T1p)
            h = L / N
            eoc_L2 = eoc_DG = float('nan')
            if prev_h is not None:
                eoc_L2 = np.log(prev_eL2 / eL2) / np.log(prev_h / h)
                eoc_DG = np.log(prev_eDG / eDG) / np.log(prev_h / h)
                print(f"  {N:>5}  {h:>8.5f}  {eL2:>12.4e}  {eoc_L2:>8.3f}"
                      f"  {eDG:>12.4e}  {eoc_DG:>8.3f}")
            else:
                print(f"  {N:>5}  {h:>8.5f}  {eL2:>12.4e}  {'—':>8}"
                      f"  {eDG:>12.4e}  {'—':>8}")
            t1_rows.append({'p': p, 'N': N, 'h': h,
                             'E_L2': eL2, 'EOC_L2': eoc_L2,
                             'E_DG': eDG,  'EOC_DG': eoc_DG})
            t1_plot[p]['h'].append(h)
            t1_plot[p]['eL2'].append(eL2)
            t1_plot[p]['eDG'].append(eDG)
            prev_eL2, prev_eDG, prev_h = eL2, eDG, h

    _csv_write('results/tables/mms_spatial_convergence.csv',
               ['p', 'N', 'h', 'E_L2', 'EOC_L2', 'E_DG', 'EOC_DG'], t1_rows)
    print(f"\n  → results/tables/mms_spatial_convergence.csv")

    for r in t1_rows:
        if not np.isnan(r['EOC_L2']):
            lo, hi = r['p'] + 0.7, r['p'] + 1.3
            if not (lo <= r['EOC_L2'] <= hi):
                print(f"  *** WARNING p={r['p']}, N={r['N']}: "
                      f"EOC_L2={r['EOC_L2']:.3f} outside [{lo:.1f}, {hi:.1f}]")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Table 2 — Temporal convergence
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 72)
    print(f"Table 2 — Temporal convergence  (N=256, p=1, T=0.5, eps={EPS_D})")
    print("=" * 72)

    DT_LIST2 = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    print(f"\n  {'dt':>8}  {'E_L2':>12}  {'EOC':>8}")
    prev_eL2 = prev_dt = None
    t2_rows = []; t2_dts = []; t2_es = []

    for dt in DT_LIST2:
        mesh, basis, U_T = run(256, 1, dt, 0.5, EPS_D)
        eL2, _ = errs(mesh, basis, U_T, 0.5)
        eoc = float('nan')
        if prev_eL2 is not None:
            eoc = np.log(prev_eL2 / eL2) / np.log(prev_dt / dt)
            print(f"  {dt:>8.5f}  {eL2:>12.4e}  {eoc:>8.3f}")
        else:
            print(f"  {dt:>8.5f}  {eL2:>12.4e}  {'—':>8}")
        t2_rows.append({'dt': dt, 'E_L2': eL2, 'EOC': eoc})
        t2_dts.append(dt); t2_es.append(eL2)
        prev_eL2, prev_dt = eL2, dt

    _csv_write('results/tables/mms_temporal_convergence.csv',
               ['dt', 'E_L2', 'EOC'], t2_rows)
    print(f"\n  → results/tables/mms_temporal_convergence.csv")

    for r in t2_rows:
        if not np.isnan(r['EOC']) and not (0.8 <= r['EOC'] <= 1.2):
            print(f"  *** WARNING dt={r['dt']}: EOC={r['EOC']:.3f} outside [0.8, 1.2]")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Table 3 — Regularization sensitivity
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 72)
    print("Table 3 — Regularization sensitivity  (N=128, p=2, dt=1e-3, T=0.5)")
    print("=" * 72)

    h3 = L / 128
    EPS_LIST3 = [0.1, 0.05, 0.02, 0.01]
    print(f"\n  {'eps':>6}  {'h/eps':>8}  {'E_L2':>12}  {'E_DG':>12}"
          f"  {'coeff_chg_max':>14}  flag")
    t3_rows = []

    for eps3 in EPS_LIST3:
        h_over_eps = h3 / eps3
        flag = 'UNDER-RESOLVED' if h_over_eps > 0.5 else ''

        diag3 = DiagnosticsRecorder(coeff_func=make_coeff_fn(eps3))
        mesh3, basis3, U_T3 = run(128, 2, 1e-3, 0.5, eps3,
                                   diagnostics=diag3, reassemble=1)
        eL2_3, eDG_3 = errs(mesh3, basis3, U_T3, 0.5)

        chg_vals = [r['coeff_change'] for r in diag3.records
                    if not np.isnan(r.get('coeff_change', float('nan')))]
        coeff_max = max(chg_vals) if chg_vals else 0.0

        flag_str = ' *' if flag else ''
        print(f"  {eps3:>6.3f}  {h_over_eps:>8.4f}  {eL2_3:>12.4e}"
              f"  {eDG_3:>12.4e}  {coeff_max:>14.4e}{flag_str}")
        t3_rows.append({'eps': eps3, 'h/eps': h_over_eps,
                        'E_L2': eL2_3, 'E_DG': eDG_3,
                        'coeff_change_max': coeff_max, 'flag': flag})

    _csv_write('results/tables/mms_eps_sensitivity.csv',
               ['eps', 'h/eps', 'E_L2', 'E_DG', 'coeff_change_max', 'flag'],
               t3_rows)
    print(f"\n  → results/tables/mms_eps_sensitivity.csv")
    if any(r['flag'] for r in t3_rows):
        print(f"  * UNDER-RESOLVED: h/ε > 0.5  "
              f"(h = {h3:.5f} for N=128 on [{X_MIN}, {X_MAX}])")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Figures
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if not _has_mpl:
        print("\nmatplotlib not available — skipping figures.")
        return

    # Figure 1 — Solution snapshots (p=2, N=32, dt=1e-3)
    _p_s, _N_s, _dt_s = 2, 32, 1e-3
    _mesh_s = Mesh1D(X_MIN, X_MAX, _N_s)
    _xi_s, _ = gauss_legendre_ref(2 * _p_s + 4)
    _basis_s  = LagrangeBasis(_p_s, _xi_s)
    _f_s = make_f(EPS_D)

    _slvr0 = _BESolver(_mesh_s, _basis_s, {'penalty_C': ETA_C})
    _U0_s  = _slvr0._project_initial(lambda x: u_ex(x, 0.0))

    _slvr1 = _BESolver(_mesh_s, _basis_s, {'penalty_C': ETA_C})
    _U25_s = _slvr1.solve(lambda x: u_ex(x, 0.0), _f_s, None, g_D, 0.25, _dt_s,
                           r=R, Psi_func=Psi_f, Psi_x_func=Psi_xf,
                           sigma_H=SH, sigma_L=SL, eps=EPS_D)[-1][1]

    _slvr2 = _BESolver(_mesh_s, _basis_s, {'penalty_C': ETA_C})
    _U50_s = _slvr2.solve(lambda x: u_ex(x, 0.0), _f_s, None, g_D, 0.5, _dt_s,
                           r=R, Psi_func=Psi_f, Psi_x_func=Psi_xf,
                           sigma_H=SH, sigma_L=SL, eps=EPS_D)[-1][1]

    _x_plt = np.linspace(X_MIN, X_MAX, 500)
    _snaps  = [(0.0, _U0_s), (0.25, _U25_s), (0.5, _U50_s)]

    fig1, axes1 = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (t_s, U_s) in zip(axes1, _snaps):
        u_h = _eval_dg_solution(_mesh_s, _basis_s, U_s, _x_plt)
        ax.plot(_x_plt, u_ex(_x_plt, t_s), 'k-',  lw=1.5, label=r'$u_{\rm ex}$')
        ax.plot(_x_plt, u_h,                'r--', lw=1.2, label=r'$U_h$')
        ax.axhline(PSI_C, color='gray', ls=':', lw=0.8, label=r'$\Psi{=}0.5$')
        ax.set_title(f't = {t_s}'); ax.set_xlabel('x'); ax.legend(fontsize=8)
    axes1[0].set_ylabel('u')
    fig1.suptitle(f'Solution snapshots  (p={_p_s}, N={_N_s}, ε={EPS_D})')
    fig1.tight_layout()
    fig1.savefig('results/figures/mms_solution_snapshots.png', dpi=150)
    plt.close(fig1)
    print("\n  → results/figures/mms_solution_snapshots.png")

    # Figure 2 — Spatial convergence log-log
    _clr = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}
    _mk  = {1: 'o', 2: 's', 3: '^'}
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for p in [1, 2, 3]:
        hs = t1_plot[p]['h']; es = t1_plot[p]['eL2']
        ax2.loglog(hs, es, marker=_mk[p], color=_clr[p], lw=1.5, ms=5, label=f'p={p}')
        h_r = np.array([hs[0], hs[-1]])
        ax2.loglog(h_r, es[-1] * (h_r / hs[-1]) ** (p + 1),
                   '--', color=_clr[p], lw=0.8, label=f'O(h^{p+1})')
    ax2.set_xlabel('h'); ax2.set_ylabel(r'$\|u_{\rm ex}-U_h\|_{L^2}$')
    ax2.set_title('Spatial convergence — regularized PDE MMS')
    ax2.legend(ncol=2, fontsize=9); ax2.grid(True, which='both', alpha=0.3)
    fig2.tight_layout()
    fig2.savefig('results/figures/mms_spatial_convergence.png', dpi=150)
    plt.close(fig2)
    print("  → results/figures/mms_spatial_convergence.png")

    # Figure 3 — Temporal convergence log-log
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.loglog(t2_dts, t2_es, 'o-', color='tab:blue', lw=1.5, ms=5, label='E_L2')
    dt_r = np.array([t2_dts[0], t2_dts[-1]])
    ax3.loglog(dt_r, t2_es[-1] * (dt_r / t2_dts[-1]), 'k--', lw=0.8, label='O(Δt)')
    ax3.set_xlabel(r'$\Delta t$'); ax3.set_ylabel(r'$\|u_{\rm ex}-U_h\|_{L^2}$')
    ax3.set_title('Temporal convergence  (N=256, p=1)')
    ax3.legend(); ax3.grid(True, which='both', alpha=0.3)
    fig3.tight_layout()
    fig3.savefig('results/figures/mms_temporal_convergence.png', dpi=150)
    plt.close(fig3)
    print("  → results/figures/mms_temporal_convergence.png")


# ── Dispatch: if --config given, run Phase 5 and exit ────────────────────────
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument('--config', default=None)
_parsed, _ = _parser.parse_known_args()
if _parsed.config is not None:
    _phase5_main(_parsed.config)
    sys.exit(0)

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
