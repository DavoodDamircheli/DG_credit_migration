"""Tracks the migration boundary s_{h,ε}(t) from the DG solution."""
import numpy as np
from scipy.optimize import brentq


_ZERO_TOL  = 1e-12   # treat |F[i]| < this as an exact root at the grid node
_KAPPA_WARN = 0.01   # default transversality warning threshold


def _make_continuous(mesh, basis, U):
    """Nodal averaging at shared DG faces → continuous piecewise polynomial.

    For GLL bases the face nodes are the first (ξ=-1) and last (ξ=+1) local
    DOFs of each element.  Replacing each pair of shared face values with
    their arithmetic mean produces a globally continuous piecewise Lagrange
    polynomial R_h U_h, without altering the interior DOFs.
    """
    p     = basis.p
    n_loc = p + 1

    phi_p1 = basis.phi(np.array([1.0]))[0]
    phi_m1 = basis.phi(np.array([-1.0]))[0]

    U_cont = U.copy()
    for e_idx in range(len(mesh.interior_face_indices)):
        K_L = mesh.face_left_element[e_idx]
        K_R = mesh.face_right_element[e_idx]
        trace_L = float(phi_p1 @ U[K_L * n_loc:(K_L + 1) * n_loc])
        trace_R = float(phi_m1 @ U[K_R * n_loc:(K_R + 1) * n_loc])
        avg = 0.5 * (trace_L + trace_R)
        U_cont[K_L * n_loc + n_loc - 1] = avg  # right end of K_L (ξ = +1)
        U_cont[K_R * n_loc]             = avg  # left  end of K_R (ξ = -1)
    return U_cont


def find_roots_in_interval(F_func, x_min, x_max, n_plot,
                            kappa_warn=_KAPPA_WARN, s_prev=None):
    """Find all roots of F_func in [x_min, x_max] on a fine uniform grid.

    Algorithm
    ---------
    1. Evaluate F on n_plot+1 equally-spaced points.
    2. Collect near-zero grid values (|F[i]| < _ZERO_TOL) as roots directly.
    3. For each strict sign-change bracket, refine with scipy brentq.
    4. Sort + deduplicate roots closer than 1.5 grid spacings.
    5. If multiple roots are found, warn and select according to s_prev.
    6. Approximate transversality κ = |∂_x F(s)| via central finite difference.

    Parameters
    ----------
    F_func     : callable, accepts a 1-D numpy array, returns a 1-D array
    x_min, x_max : domain endpoints
    n_plot     : number of fine-grid sub-intervals
    kappa_warn : threshold below which a transversality warning is printed
    s_prev     : previous time step's root (for nearest-to-prev selection);
                 used only when n_roots > 1

    Returns
    -------
    dict with keys: s, kappa, n_roots, all_roots
    """
    x_fine = np.linspace(x_min, x_max, n_plot + 1)
    F_fine = np.asarray(F_func(x_fine), dtype=float).ravel()
    dx     = (x_max - x_min) / n_plot

    def F_scalar(x):
        return float(np.asarray(F_func(np.array([x], dtype=float))).ravel()[0])

    raw = []

    # Near-zero grid nodes
    for i in range(len(x_fine)):
        if abs(F_fine[i]) < _ZERO_TOL:
            raw.append(float(x_fine[i]))

    # Strict sign-change brackets → Brent's method
    for i in range(len(F_fine) - 1):
        if F_fine[i] * F_fine[i + 1] < 0:
            root = brentq(F_scalar, float(x_fine[i]), float(x_fine[i + 1]),
                          xtol=1e-10, rtol=1e-10)
            raw.append(float(root))

    # Sort and deduplicate
    raw.sort()
    dedup = 1.5 * dx
    merged = []
    for r in raw:
        if not merged or r - merged[-1] > dedup:
            merged.append(r)
    all_roots = merged
    n_roots   = len(all_roots)

    if n_roots == 0:
        return {'s': float('nan'), 'kappa': float('nan'),
                'n_roots': 0, 'all_roots': []}

    # Select the financial boundary root
    if n_roots > 1:
        if s_prev is not None and not np.isnan(float(s_prev)):
            rule = 'nearest to s_prev'
            idx  = int(np.argmin(np.abs(np.array(all_roots) - float(s_prev))))
        else:
            rule = 'rightmost'
            idx  = n_roots - 1
        print(f"WARNING: {n_roots} roots found: "
              f"{[f'{r:.6f}' for r in all_roots]}. "
              f"Selection rule: {rule}.")
        s = all_roots[idx]
    else:
        s = all_roots[0]

    # Transversality: κ = |∂_x F(s)| via central finite difference
    margin  = min(s - x_min, x_max - s)
    eps_fd  = max(min(0.5 * dx, 0.4 * margin) if margin > 0 else 0.5 * dx,
                  1e-8)
    x_plus  = min(s + eps_fd, x_max)
    x_minus = max(s - eps_fd, x_min)
    kappa   = abs((F_scalar(x_plus) - F_scalar(x_minus)) / (x_plus - x_minus))

    if kappa < kappa_warn:
        print(f"WARNING: small transversality κ={kappa:.4e} at s={s:.6f}")

    return {'s': s, 'kappa': kappa, 'n_roots': n_roots, 'all_roots': all_roots}


class BoundaryTracker:
    """Tracks s_{h,ε}(t) = {x : R_h U_h^n(x) = Ψ(x, t)}.

    Continuous reconstruction uses nodal averaging (_make_continuous).
    Root finding delegates to find_roots_in_interval.

    Parameters
    ----------
    mesh, basis   : Mesh1D and LagrangeBasis
    Psi_func      : callable Ψ(x, t) → 1-D array
    n_plot_factor : fine grid = n_plot_factor × N elements
    kappa_warn    : transversality warning threshold
    """

    def __init__(self, mesh, basis, Psi_func, n_plot_factor=10,
                 kappa_warn=_KAPPA_WARN):
        self.mesh         = mesh
        self.basis        = basis
        self.Psi_func     = Psi_func
        self.n_plot       = n_plot_factor * mesh.N
        self.kappa_warn   = kappa_warn

    def track(self, U, t, s_prev=None):
        """Track the boundary from the DG solution U at time t.

        Returns a dict with keys: t, s, kappa, n_roots, all_roots.
        """
        from .time_stepper import _eval_dg_solution

        mesh     = self.mesh
        basis    = self.basis
        Psi_func = self.Psi_func
        U_cont   = _make_continuous(mesh, basis, U)

        def F_h(x_arr):
            x_arr   = np.asarray(x_arr, dtype=float).ravel()
            Rh_vals = _eval_dg_solution(mesh, basis, U_cont, x_arr)
            Psi_val = np.asarray(Psi_func(x_arr, t), dtype=float).ravel()
            return Rh_vals - Psi_val

        result = find_roots_in_interval(
            F_h, float(mesh.x[0]), float(mesh.x[-1]),
            self.n_plot,
            kappa_warn=self.kappa_warn,
            s_prev=s_prev,
        )
        result['t'] = float(t)
        return result
