"""Diagnostics recorder: per-step norms, errors, and coefficient change."""
import csv
import numpy as np
from .quadrature import gauss_legendre_ref


class DiagnosticsRecorder:
    """Records solver diagnostics at each time step and saves to CSV.

    Call record_step() after each solve step.  Optional kwargs unlock
    residual_norm and coeff_change when the caller supplies the needed data.
    """

    def __init__(self, coeff_func=None):
        """
        coeff_func : optional callable (U, mesh, basis, t) → 1-D array
            Returns a_ε values at all quadrature points given the DG solution U.
            When provided, coeff_change is the L^∞ change in a_ε between
            consecutive steps.
        """
        self.records = []
        self._U_prev_stored = None
        self._coeff_func = coeff_func

    def record_step(self, n, t, U, mesh, basis, u_ex=None,
                    U_prev=None, dt=None, M=None, K=None, F=None,
                    a_func=None):
        """Compute and store diagnostics for time step n at time t.

        Parameters
        ----------
        n        : step index
        t        : physical time
        U        : current DOF vector U^n
        mesh     : Mesh1D object
        basis    : LagrangeBasis object
        u_ex     : callable u_ex(x, t) → array, used for L2_error (optional)
        U_prev   : DOF vector U^{n-1} (optional; stored internally as fallback)
        dt       : time step size (needed for residual_norm)
        M        : mass matrix (needed for residual_norm and fast L2_norm)
        K        : stiffness matrix K(U^{n-1}, t_n) (needed for residual_norm)
        F        : load vector F^n (needed for residual_norm)
        a_func   : callable a_func(x) → a values (needed for coeff_change);
                   should be the frozen-state a_w from the current step
        """
        p     = basis.p
        n_loc = p + 1
        N_elem = mesh.N
        n_quad = 2 * p + 4
        xi_q, w_q = gauss_legendre_ref(n_quad)
        phi_q  = basis.phi(xi_q)           # (n_quad, n_loc)
        dphi_q = basis.dphi_dxi(xi_q)      # (n_quad, n_loc)

        record = {'n': n, 't': t}

        # ── L2_norm ────────────────────────────────────────────────────
        if M is not None:
            record['L2_norm'] = float(np.sqrt(np.dot(U, M @ U)))
        else:
            L2_sq = 0.0
            for Ke in range(N_elem):
                x_L, x_R = mesh.element_interval(Ke)
                h_K = x_R - x_L
                J   = h_K / 2.0
                u_h = phi_q @ U[Ke * n_loc:(Ke + 1) * n_loc]
                L2_sq += J * np.dot(w_q, u_h ** 2)
            record['L2_norm'] = float(np.sqrt(L2_sq))

        # ── DG_energy: Σ_K ‖∂_x u_h‖²_K  +  Σ_e [u_h]²_e / h_e ─────
        phi_m1 = basis.phi(np.array([-1.0]))[0]
        phi_p1 = basis.phi(np.array([ 1.0]))[0]
        energy = 0.0
        for Ke in range(N_elem):
            x_L, x_R = mesh.element_interval(Ke)
            h_K = x_R - x_L
            J   = h_K / 2.0
            du_h = (2.0 / h_K) * (dphi_q @ U[Ke * n_loc:(Ke + 1) * n_loc])
            energy += J * np.dot(w_q, du_h ** 2)
        for e_idx in range(len(mesh.interior_face_indices)):
            K_L = mesh.face_left_element[e_idx]
            K_R = mesh.face_right_element[e_idx]
            h_e = min(mesh.h[K_L], mesh.h[K_R])
            u_L = phi_p1 @ U[K_L * n_loc:(K_L + 1) * n_loc]
            u_R = phi_m1 @ U[K_R * n_loc:(K_R + 1) * n_loc]
            energy += (u_L - u_R) ** 2 / h_e
        record['DG_energy'] = float(np.sqrt(energy))

        # ── u_min / u_max ──────────────────────────────────────────────
        record['u_min'] = float(U.min())
        record['u_max'] = float(U.max())

        # ── residual_norm: ‖(M + dt K) U - M U_prev - dt F‖ ──────────
        U_p = U_prev if U_prev is not None else self._U_prev_stored
        if (M is not None and K is not None and F is not None
                and U_p is not None and dt is not None):
            res = (M + dt * K) @ U - M @ U_p - dt * F
            record['residual_norm'] = float(np.linalg.norm(res))
        else:
            record['residual_norm'] = float('nan')

        # ── coeff_change: ‖a_ε(U^n,·) - a_ε(U^{n-1},·)‖_{L^∞} ──────
        if self._coeff_func is not None and U_p is not None:
            a_curr = np.asarray(self._coeff_func(U,   mesh, basis, t),  float)
            a_prev = np.asarray(self._coeff_func(U_p, mesh, basis, t),  float)
            record['coeff_change'] = float(np.max(np.abs(a_curr - a_prev)))
        else:
            record['coeff_change'] = float('nan')

        # ── L2_error ───────────────────────────────────────────────────
        if u_ex is not None:
            L2_err_sq = 0.0
            for Ke in range(N_elem):
                x_L, x_R = mesh.element_interval(Ke)
                h_K = x_R - x_L
                J   = h_K / 2.0
                x_pts = 0.5 * (x_L + x_R) + 0.5 * h_K * xi_q
                u_h = phi_q @ U[Ke * n_loc:(Ke + 1) * n_loc]
                u_e = np.asarray(u_ex(x_pts, t), dtype=float)
                L2_err_sq += J * np.dot(w_q, (u_e - u_h) ** 2)
            record['L2_error'] = float(np.sqrt(L2_err_sq))
        else:
            record['L2_error'] = float('nan')

        self.records.append(record)
        self._U_prev_stored = U.copy()

    def save_csv(self, filename):
        """Write all recorded diagnostics to a CSV file."""
        if not self.records:
            return
        fieldnames = list(self.records[0].keys())
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
