"""Frozen-coefficient backward Euler time stepper for DG diffusion."""
import numpy as np
from scipy.sparse.linalg import splu
from .quadrature import gauss_legendre_ref
from .dg_operators import (assemble_mass_matrix, assemble_sipg_diffusion,
                            assemble_stiffness, assemble_drift_inflow_bc)


def _eval_dg_solution(mesh, basis, U, x):
    """Evaluate DG solution U at physical points x — fully vectorized.

    Uses searchsorted to locate each point's element, maps to reference
    coordinates, and evaluates the Lagrange interpolant.  No Python loop.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = mesh.N
    n_loc = basis.p + 1
    K_arr = np.clip(np.searchsorted(mesh.x[1:], x), 0, N - 1)
    x_L = mesh.x[K_arr]
    h_K = mesh.x[K_arr + 1] - x_L
    xi = np.clip(2.0 * (x - x_L) / h_K - 1.0, -1.0, 1.0)
    phi = basis.phi(xi)                            # (n_pts, n_loc)
    return np.sum(phi * U.reshape(N, n_loc)[K_arr], axis=1)


def _eval_dg_gradient(mesh, basis, U, x):
    """Evaluate ∂_x U at physical points x — fully vectorized."""
    x = np.asarray(x, dtype=float).ravel()
    N = mesh.N
    n_loc = basis.p + 1
    K_arr = np.clip(np.searchsorted(mesh.x[1:], x), 0, N - 1)
    x_L = mesh.x[K_arr]
    h_K = mesh.x[K_arr + 1] - x_L
    xi = np.clip(2.0 * (x - x_L) / h_K - 1.0, -1.0, 1.0)
    dphi = basis.dphi_dxi(xi)                      # (n_pts, n_loc)
    return (2.0 / h_K) * np.sum(dphi * U.reshape(N, n_loc)[K_arr], axis=1)


class BackwardEulerSolver:
    def __init__(self, mesh, basis, config):
        self.mesh = mesh
        self.basis = basis
        self.eta_C = config.get('penalty_C', 10)
        self.p = basis.p

    def _project_initial(self, u0_func):
        """L2-project u0_func into V_h^p (M is block-diagonal → element solves).

        u0_func must accept a 1-D numpy array and return a 1-D array.
        """
        mesh = self.mesh
        basis = self.basis
        p = self.p
        n_loc = p + 1
        N = mesh.N
        n_quad = 2 * p + 4
        xi_q, w_q = gauss_legendre_ref(n_quad)
        phi_q = basis.phi(xi_q)

        # Local mass matrix (same for all elements on a uniform mesh)
        h0 = mesh.h[0]
        J0 = h0 / 2.0
        M_loc_ref = J0 * np.einsum('q,qi,qj->ij', w_q, phi_q, phi_q)

        # Batch all elements
        x_L = mesh.x[:-1]; x_R = mesh.x[1:]; h_K = x_R - x_L
        x_phys = 0.5*(x_L[:,None]+x_R[:,None]) + 0.5*h_K[:,None]*xi_q[None,:]  # (N,nq)
        u0_all = np.asarray(u0_func(x_phys), dtype=float)  # (N, nq)

        U0 = np.zeros(N * n_loc)
        for K in range(N):
            J = h_K[K] / 2.0
            M_loc = J / J0 * M_loc_ref   # scale if non-uniform (generalises)
            b_loc = J * np.einsum('q,q,qi->i', w_q, u0_all[K], phi_q)
            dofs = slice(K * n_loc, (K + 1) * n_loc)
            U0[dofs] = np.linalg.solve(M_loc, b_loc)

        return U0

    def solve(self, u0_func, f_func, a_func, g_D_func, T, dt,
              beta_func=None, gamma_func=None, r=0.0,
              Psi_func=None, Psi_x_func=None,
              sigma_H=None, sigma_L=None, eps=None,
              diagnostics=None, reassemble_every=1):
        """Backward Euler time stepper, dispatching between two paths.

        Old path (Psi_func is None): pure-diffusion heat equation with
        time-independent a_func(x).  K is assembled and factored once.

        Frozen-coefficient path (Psi_func provided): full regularized PDE.
        At each step the stiffness is re-assembled from U^{n-1} using the
        regularized coefficients a_ε, β_ε, γ_w.

        reassemble_every : int (default 1)
            Reassemble the stiffness matrix only every this many steps.
            Use values > 1 when the PDE coefficients change slowly (e.g.
            the state stays far from the free boundary) to reduce cost.
            reassemble_every=1 gives full Picard accuracy at every step.

        Returns [(0.0, U^0), (T, U^T)].
        """
        if Psi_func is None:
            return self._solve_linear(u0_func, f_func, a_func, g_D_func,
                                      T, dt, diagnostics=diagnostics)
        return self._solve_frozen(u0_func, f_func, g_D_func, T, dt,
                                  r, Psi_func,
                                  Psi_x_func if Psi_x_func is not None
                                  else (lambda x, t: np.zeros_like(np.asarray(x, float))),
                                  sigma_H, sigma_L, eps,
                                  diagnostics=diagnostics,
                                  reassemble_every=reassemble_every)

    # ------------------------------------------------------------------ #
    # Heat-equation (linear, frozen a) path                               #
    # ------------------------------------------------------------------ #

    def _solve_linear(self, u0_func, f_func, a_func, g_D_func, T, dt,
                      diagnostics=None):
        """Frozen-coefficient backward Euler for the heat equation.

        Solves (M + dt K_diff) U^n = M U^{n-1} + dt F^n at each step.
        a_func is assumed constant in time — K_diff is assembled once.

        f_func(x, t) and a_func(x) must accept 2-D / 1-D numpy arrays.
        """
        mesh  = self.mesh
        basis = self.basis
        eta_C = self.eta_C
        p     = self.p
        n_loc = p + 1
        N     = mesh.N

        M      = assemble_mass_matrix(mesh, basis)
        K_diff = assemble_sipg_diffusion(mesh, basis, a_func, eta_C, p)
        A      = (M + dt * K_diff).tocsc()
        A_lu   = splu(A)       # factor A once; reuse for every step

        # ── Precompute static quadrature data ──────────────────────────
        n_quad     = 2 * p + 4
        xi_q, w_q  = gauss_legendre_ref(n_quad)
        phi_q      = basis.phi(xi_q)           # (n_quad, p+1)

        x_L = mesh.x[:-1]
        x_R = mesh.x[1:]
        h_K = x_R - x_L
        J_K = h_K / 2.0

        # Physical quadrature points for all elements: (N, n_quad)
        x_phys = (0.5 * (x_L[:, None] + x_R[:, None])
                  + 0.5 * h_K[:, None] * xi_q[None, :])

        # ── Precompute Nitsche BC static data ──────────────────────────
        eta = eta_C * (p + 1) ** 2

        x_bc_L   = float(mesh.faces[0])
        x_bc_R   = float(mesh.faces[N])
        h_bc_L   = float(mesh.h[0])
        h_bc_R   = float(mesh.h[N - 1])

        a_bc_L   = float(np.asarray(a_func(np.array([x_bc_L]))).ravel()[0])
        a_bc_R   = float(np.asarray(a_func(np.array([x_bc_R]))).ravel()[0])
        pen_L    = eta * a_bc_L / h_bc_L
        pen_R    = eta * a_bc_R / h_bc_R

        phi_m1   = basis.phi(np.array([-1.0]))[0]    # (p+1,)
        phi_p1   = basis.phi(np.array([ 1.0]))[0]
        dphi_m1  = basis.dphi_dx(np.array([-1.0]), h_bc_L)[0]
        dphi_p1  = basis.dphi_dx(np.array([ 1.0]), h_bc_R)[0]

        # Precomputed coefficient vectors for BC load (right-hand side)
        coeff_bc_L = -a_bc_L * dphi_m1 - pen_L * phi_m1   # (p+1,)
        coeff_bc_R =  a_bc_R * dphi_p1 - pen_R * phi_p1

        # ── Time loop ──────────────────────────────────────────────────
        U  = self._project_initial(u0_func)
        U0 = U.copy()

        n_steps = int(round(T / dt))
        for n in range(n_steps):
            t_new = (n + 1) * dt

            # Volume source (vectorised over all elements)
            f_all = np.asarray(f_func(x_phys, t_new), dtype=float)
            F = (J_K[:, None] * np.einsum('q,Kq,qi->Ki', w_q, f_all, phi_q)).ravel()

            # Nitsche BC load (only non-zero when g_D ≠ 0 at boundary)
            g_L = float(g_D_func(x_bc_L, t_new))
            g_R = float(g_D_func(x_bc_R, t_new))
            if g_L:
                F[:n_loc]  += coeff_bc_L * g_L
            if g_R:
                F[-n_loc:] += coeff_bc_R * g_R

            rhs = M @ U + dt * F
            U   = A_lu.solve(rhs)

            if diagnostics is not None:
                diagnostics.record_step(n + 1, t_new, U, mesh, basis,
                                        U_prev=U0 if n == 0 else None,
                                        M=M, dt=dt)

        return [(0.0, U0), (T, U.copy())]

    # ------------------------------------------------------------------ #
    # Frozen-coefficient nonlinear path                                   #
    # ------------------------------------------------------------------ #

    def _solve_frozen(self, u0_func, f_func, g_D_func, T, dt,
                      r, Psi_func, Psi_x_func, sigma_H, sigma_L, eps,
                      diagnostics=None, reassemble_every=1):
        """Frozen-state backward Euler for the regularized nonlinear PDE.

        At each step the three coefficient callables a_w, beta_w, gamma_w
        are built from the frozen state U^{n-1} and passed to
        assemble_stiffness.  The LU factorization is recomputed every
        reassemble_every steps (default: 1 = every step).
        """
        from .coefficients import (a_eps as _a_eps, beta_eps as _beta_eps,
                                   gamma_eps_elementwise)

        mesh  = self.mesh
        basis = self.basis
        eta_C = self.eta_C
        p     = self.p
        n_loc = p + 1
        N     = mesh.N

        M = assemble_mass_matrix(mesh, basis)

        # ── Static quadrature data ─────────────────────────────────────
        n_quad    = 2 * p + 4
        xi_q, w_q = gauss_legendre_ref(n_quad)
        phi_q     = basis.phi(xi_q)           # (n_quad, p+1)

        x_L_arr = mesh.x[:-1]
        x_R_arr = mesh.x[1:]
        h_K_arr = x_R_arr - x_L_arr
        J_K_arr = h_K_arr / 2.0

        # Physical quadrature points for all elements: (N, n_quad)
        x_phys = (0.5 * (x_L_arr[:, None] + x_R_arr[:, None])
                  + 0.5 * h_K_arr[:, None] * xi_q[None, :])

        # ── Static Nitsche BC geometry (recompute scalars each step) ───
        eta     = eta_C * (p + 1) ** 2
        x_bc_L  = float(mesh.faces[0])
        x_bc_R  = float(mesh.faces[N])
        h_bc_L  = float(mesh.h[0])
        h_bc_R  = float(mesh.h[N - 1])

        phi_m1  = basis.phi(np.array([-1.0]))[0]    # (p+1,) — constant
        phi_p1  = basis.phi(np.array([ 1.0]))[0]
        dphi_m1 = basis.dphi_dx(np.array([-1.0]), h_bc_L)[0]
        dphi_p1 = basis.dphi_dx(np.array([ 1.0]), h_bc_R)[0]

        # ── Initial condition ──────────────────────────────────────────
        U  = self._project_initial(u0_func)
        U0 = U.copy()

        n_steps = int(round(T / dt))

        # Cached stiffness (rebuilt every reassemble_every steps)
        K_stiff = None; A_lu = None
        a_w_c = beta_w_c = gamma_w_c = None   # cached callables
        coeff_bc_L = coeff_bc_R = None

        for step in range(n_steps):
            t_n    = (step + 1) * dt
            U_prev = U.copy()

            if K_stiff is None or (step % reassemble_every == 0):
                # Build frozen callables that capture U_prev and t_n by value.
                def a_w(x_in, _U=U_prev, _t=t_n):
                    x = np.asarray(x_in, dtype=float).ravel()
                    w = _eval_dg_solution(mesh, basis, _U, x)
                    return _a_eps(w, x, _t, Psi_func, sigma_H, sigma_L, eps)

                def beta_w(x_in, _U=U_prev, _t=t_n):
                    x  = np.asarray(x_in, dtype=float).ravel()
                    w  = _eval_dg_solution(mesh, basis, _U, x)
                    wx = _eval_dg_gradient(mesh, basis, _U, x)
                    Px = np.asarray(Psi_x_func(x, _t), dtype=float)
                    return _beta_eps(w, wx, Px, x, _t, r, Psi_func,
                                     sigma_H, sigma_L, eps)

                def gamma_w(x_in, _U=U_prev, _t=t_n):
                    x  = np.asarray(x_in, dtype=float).ravel()
                    K  = int(np.clip(np.searchsorted(mesh.x[1:], x[0]), 0, N - 1))
                    w  = _eval_dg_solution(mesh, basis, _U, x)
                    wx = _eval_dg_gradient(mesh, basis, _U, x)
                    return gamma_eps_elementwise(w, wx, x, _t, mesh, basis, K, r,
                                                 Psi_func, Psi_x_func,
                                                 sigma_H, sigma_L, eps)

                a_w_c = a_w; beta_w_c = beta_w; gamma_w_c = gamma_w

                K_stiff = assemble_stiffness(mesh, basis, a_w_c, beta_w_c, gamma_w_c,
                                              r, eta_C, p, t_n)
                A_lu = splu((M + dt * K_stiff).tocsc())

                # Nitsche BC coefficients from the freshly frozen a_w_c
                a_bc_L = float(a_w_c(np.array([x_bc_L]))[0])
                a_bc_R = float(a_w_c(np.array([x_bc_R]))[0])
                pen_L  = eta * a_bc_L / h_bc_L
                pen_R  = eta * a_bc_R / h_bc_R
                coeff_bc_L = -(a_bc_L * dphi_m1 + pen_L * phi_m1)
                coeff_bc_R =  a_bc_R * dphi_p1 - pen_R * phi_p1

            # ── Load vector ────────────────────────────────────────────
            f_all = np.asarray(f_func(x_phys, t_n), dtype=float)
            F = (J_K_arr[:, None]
                 * np.einsum('q,Kq,qi->Ki', w_q, f_all, phi_q)).ravel()

            g_L = float(g_D_func(x_bc_L, t_n))
            g_R = float(g_D_func(x_bc_R, t_n))
            if g_L:
                F[:n_loc]  += coeff_bc_L * g_L
            if g_R:
                F[-n_loc:] += coeff_bc_R * g_R

            # Upwind drift inflow BC (beta_w_c evaluated at cached frozen state)
            F += assemble_drift_inflow_bc(mesh, basis, beta_w_c, g_D_func, t_n)

            U = A_lu.solve(M @ U_prev + dt * F)

            if diagnostics is not None:
                diagnostics.record_step(step + 1, t_n, U, mesh, basis,
                                        U_prev=U_prev, dt=dt, M=M,
                                        K=K_stiff, F=F)

        return [(0.0, U0), (T, U.copy())]
