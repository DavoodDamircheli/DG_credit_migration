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
              diagnostics=None, reassemble_every=1, on_step=None):
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

        on_step : callable(step, t, U) or None
            Called after each step with the 1-indexed step number, physical
            time t, and the current DOF vector U.

        Returns [(0.0, U^0), (T, U^T)].
        """
        if Psi_func is None:
            return self._solve_linear(u0_func, f_func, a_func, g_D_func,
                                      T, dt, diagnostics=diagnostics,
                                      on_step=on_step)
        return self._solve_frozen(u0_func, f_func, g_D_func, T, dt,
                                  r, Psi_func,
                                  Psi_x_func if Psi_x_func is not None
                                  else (lambda x, t: np.zeros_like(np.asarray(x, float))),
                                  sigma_H, sigma_L, eps,
                                  diagnostics=diagnostics,
                                  reassemble_every=reassemble_every,
                                  on_step=on_step)

    # ------------------------------------------------------------------ #
    # Heat-equation (linear, frozen a) path                               #
    # ------------------------------------------------------------------ #

    def _solve_linear(self, u0_func, f_func, a_func, g_D_func, T, dt,
                      diagnostics=None, on_step=None):
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

            if on_step is not None:
                on_step(n + 1, t_new, U)

        return [(0.0, U0), (T, U.copy())]

    # ------------------------------------------------------------------ #
    # Frozen-coefficient nonlinear path                                   #
    # ------------------------------------------------------------------ #

    def _solve_frozen(self, u0_func, f_func, g_D_func, T, dt,
                      r, Psi_func, Psi_x_func, sigma_H, sigma_L, eps,
                      diagnostics=None, reassemble_every=1, on_step=None):
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

            if on_step is not None:
                on_step(step + 1, t_n, U)

        return [(0.0, U0), (T, U.copy())]


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers shared by BackwardEulerSolver and PicardSolver
# ══════════════════════════════════════════════════════════════════════════════

def _build_frozen_callables(mesh, basis, U_frozen, t_frozen,
                             r, Psi_func, Psi_x_func,
                             sigma_H, sigma_L, eps):
    """Build coefficient callables (a_w, beta_w, gamma_w) from a frozen state.

    Parameters
    ----------
    mesh, basis   : Mesh1D and LagrangeBasis
    U_frozen      : DOF vector used as the frozen state for coefficient eval
    t_frozen      : physical time at which the state is evaluated
    r, Psi_func, Psi_x_func, sigma_H, sigma_L, eps : PDE parameters

    Returns
    -------
    a_w, beta_w, gamma_w : callables x_arr → 1-D array
    """
    from .coefficients import (a_eps as _a_eps, beta_eps as _beta_eps,
                                gamma_eps_elementwise)
    N = mesh.N

    def a_w(x_in):
        x = np.asarray(x_in, dtype=float).ravel()
        w = _eval_dg_solution(mesh, basis, U_frozen, x)
        return _a_eps(w, x, t_frozen, Psi_func, sigma_H, sigma_L, eps)

    def beta_w(x_in):
        x  = np.asarray(x_in, dtype=float).ravel()
        w  = _eval_dg_solution(mesh, basis, U_frozen, x)
        wx = _eval_dg_gradient(mesh, basis, U_frozen, x)
        Px = np.asarray(Psi_x_func(x, t_frozen), dtype=float)
        return _beta_eps(w, wx, Px, x, t_frozen, r, Psi_func,
                         sigma_H, sigma_L, eps)

    def gamma_w(x_in):
        x  = np.asarray(x_in, dtype=float).ravel()
        K  = int(np.clip(np.searchsorted(mesh.x[1:], x[0]), 0, N - 1))
        w  = _eval_dg_solution(mesh, basis, U_frozen, x)
        wx = _eval_dg_gradient(mesh, basis, U_frozen, x)
        return gamma_eps_elementwise(w, wx, x, t_frozen, mesh, basis, K, r,
                                     Psi_func, Psi_x_func,
                                     sigma_H, sigma_L, eps)

    return a_w, beta_w, gamma_w


def _project_l2(mesh, basis, u0_func):
    """Elementwise L²-projection of u0_func into V_h^p.

    u0_func must accept a 2-D array of shape (N, n_quad) and return the
    same shape.
    """
    p      = basis.p
    n_loc  = p + 1
    N      = mesh.N
    n_quad = 2 * p + 4
    xi_q, w_q = gauss_legendre_ref(n_quad)
    phi_q = basis.phi(xi_q)

    h0 = mesh.h[0]; J0 = h0 / 2.0
    M_loc_ref = J0 * np.einsum('q,qi,qj->ij', w_q, phi_q, phi_q)

    x_L = mesh.x[:-1]; x_R = mesh.x[1:]; h_K = x_R - x_L
    x_phys = (0.5 * (x_L[:, None] + x_R[:, None])
              + 0.5 * h_K[:, None] * xi_q[None, :])
    u0_all = np.asarray(u0_func(x_phys), dtype=float)

    U0 = np.zeros(N * n_loc)
    for K in range(N):
        J     = h_K[K] / 2.0
        M_loc = J / J0 * M_loc_ref
        b_loc = J * np.einsum('q,q,qi->i', w_q, u0_all[K], phi_q)
        U0[K * n_loc:(K + 1) * n_loc] = np.linalg.solve(M_loc, b_loc)
    return U0


# ══════════════════════════════════════════════════════════════════════════════
# Picard (fully implicit) solver
# ══════════════════════════════════════════════════════════════════════════════

class PicardSolver:
    """Fully-implicit backward Euler time stepper with Picard inner iteration.

    At each time step the nonlinear system
        (M + dt * K(U^n, t_n)) U^n = M U^{n-1} + dt F^n(U^n)
    is solved by the fixed-point iteration:
        U^{n,k+1} = (M + dt * K(U^{n,k}))^{-1} (M U^{n-1} + dt F^n(U^{n,k}))

    Initialization: U^{n,0} = U^{n-1}.

    Stopping criterion:
        ‖U^{n,k+1} - U^{n,k}‖ / (‖U^{n,k+1}‖ + 1e-14) < tol
    """

    def __init__(self, mesh, basis, config):
        self.mesh     = mesh
        self.basis    = basis
        self.eta_C    = config.get('penalty_C', 10)
        self.p        = basis.p
        self.tol      = config.get('picard_tol', 1e-8)
        self.max_iter = config.get('max_picard_iter', 20)

    # ------------------------------------------------------------------
    # One time step
    # ------------------------------------------------------------------

    def solve_step(self, U_prev, t_n, dt, f_func, g_D_func,
                   r, Psi_func, Psi_x_func, sigma_H, sigma_L, eps,
                   M=None):
        """Picard iteration for a single time step.

        Parameters
        ----------
        U_prev           : DOF vector at t_{n-1}
        t_n              : physical time of the new step
        dt               : time-step size
        f_func           : source f(x_batch, t) → same-shape array
        g_D_func         : Dirichlet BC g_D(x_scalar, t) → float
        r, Psi_func, Psi_x_func, sigma_H, sigma_L, eps : PDE parameters
        M                : mass matrix (assembled internally when None)

        Returns
        -------
        (U_new, n_iters, converged)
        """
        mesh  = self.mesh
        basis = self.basis
        eta_C = self.eta_C
        p     = self.p
        n_loc = p + 1
        N     = mesh.N
        tol   = self.tol

        if M is None:
            M = assemble_mass_matrix(mesh, basis)

        # ── Static quadrature geometry ─────────────────────────────────
        n_quad    = 2 * p + 4
        xi_q, w_q = gauss_legendre_ref(n_quad)
        phi_q     = basis.phi(xi_q)

        x_L_arr = mesh.x[:-1]; x_R_arr = mesh.x[1:]
        h_K_arr = x_R_arr - x_L_arr; J_K_arr = h_K_arr / 2.0
        x_phys  = (0.5 * (x_L_arr[:, None] + x_R_arr[:, None])
                   + 0.5 * h_K_arr[:, None] * xi_q[None, :])

        # ── BC geometry ────────────────────────────────────────────────
        eta     = eta_C * (p + 1) ** 2
        x_bc_L  = float(mesh.faces[0]);  x_bc_R  = float(mesh.faces[N])
        h_bc_L  = float(mesh.h[0]);      h_bc_R  = float(mesh.h[N - 1])
        phi_m1  = basis.phi(np.array([-1.0]))[0]
        phi_p1  = basis.phi(np.array([ 1.0]))[0]
        dphi_m1 = basis.dphi_dx(np.array([-1.0]), h_bc_L)[0]
        dphi_p1 = basis.dphi_dx(np.array([ 1.0]), h_bc_R)[0]
        g_L     = float(g_D_func(x_bc_L, t_n))
        g_R     = float(g_D_func(x_bc_R, t_n))

        # ── Constant-across-iterations parts of the RHS ────────────────
        f_all = np.asarray(f_func(x_phys, t_n), dtype=float)
        F_vol = (J_K_arr[:, None]
                 * np.einsum('q,Kq,qi->Ki', w_q, f_all, phi_q)).ravel()
        b_mass = M @ U_prev

        # ── Picard loop ────────────────────────────────────────────────
        U_k   = U_prev.copy()
        delta = np.inf

        for k in range(self.max_iter):
            a_w, beta_w, gamma_w = _build_frozen_callables(
                mesh, basis, U_k, t_n,
                r, Psi_func, Psi_x_func, sigma_H, sigma_L, eps,
            )

            K_stiff = assemble_stiffness(mesh, basis, a_w, beta_w, gamma_w,
                                          r, eta_C, p, t_n)
            A_lu = splu((M + dt * K_stiff).tocsc())

            # Nitsche BC contribution (iterate-dependent)
            a_bc_L = float(a_w(np.array([x_bc_L]))[0])
            a_bc_R = float(a_w(np.array([x_bc_R]))[0])
            pen_L  = eta * a_bc_L / h_bc_L
            pen_R  = eta * a_bc_R / h_bc_R
            c_bc_L = -(a_bc_L * dphi_m1 + pen_L * phi_m1)
            c_bc_R =   a_bc_R * dphi_p1 - pen_R * phi_p1

            F = F_vol.copy()
            if g_L:
                F[:n_loc]  += c_bc_L * g_L
            if g_R:
                F[-n_loc:] += c_bc_R * g_R
            F += assemble_drift_inflow_bc(mesh, basis, beta_w, g_D_func, t_n)

            U_new = A_lu.solve(b_mass + dt * F)

            delta = np.linalg.norm(U_new - U_k)
            denom = np.linalg.norm(U_new) + 1e-14
            if delta / denom < tol:
                return U_new, k + 1, True

            U_k = U_new

        print(f"WARNING: Picard not converged at t={t_n:.4f}  "
              f"rel_change={delta / (np.linalg.norm(U_k) + 1e-14):.2e}  "
              f"after {self.max_iter} iters.")
        return U_k, self.max_iter, False

    # ------------------------------------------------------------------
    # Full time integration
    # ------------------------------------------------------------------

    def solve(self, u0_func, f_func, g_D_func, T, dt,
              r=0.0, Psi_func=None, Psi_x_func=None,
              sigma_H=None, sigma_L=None, eps=None,
              diagnostics=None, on_step=None,
              picard_stats=None):
        """Full time integration with Picard inner loop at each step.

        Parameters
        ----------
        u0_func, f_func, g_D_func, T, dt : same as BackwardEulerSolver.solve
        r, Psi_func, Psi_x_func, sigma_H, sigma_L, eps : PDE parameters
        diagnostics  : DiagnosticsRecorder or None
        on_step      : callable(step, t, U) or None
        picard_stats : list; per-step dicts {n_iters, converged, t} are appended

        Returns [(0.0, U0), (T, U_T)]
        """
        if Psi_x_func is None:
            Psi_x_func = lambda x, t: np.zeros_like(np.asarray(x, float))

        mesh  = self.mesh
        basis = self.basis
        M     = assemble_mass_matrix(mesh, basis)
        U     = _project_l2(mesh, basis, u0_func)
        U0    = U.copy()

        n_steps = int(round(T / dt))

        for step in range(n_steps):
            t_n    = (step + 1) * dt
            U_prev = U.copy()

            U, n_iters, converged = self.solve_step(
                U_prev, t_n, dt, f_func, g_D_func,
                r, Psi_func, Psi_x_func, sigma_H, sigma_L, eps,
                M=M,
            )

            if picard_stats is not None:
                picard_stats.append({'n_iters': n_iters,
                                     'converged': converged, 't': t_n})

            if diagnostics is not None:
                diagnostics.record_step(step + 1, t_n, U, mesh, basis,
                                        U_prev=U_prev, dt=dt, M=M)

            if on_step is not None:
                on_step(step + 1, t_n, U)

        return [(0.0, U0), (T, U.copy())]


# ══════════════════════════════════════════════════════════════════════════════
# Newton diffusion Jacobian correction
# ══════════════════════════════════════════════════════════════════════════════

def _assemble_newton_diffusion_correction(mesh, basis, U, t,
                                           Psi_func, sigma_H, sigma_L, eps,
                                           eta_C, p):
    """Assemble the extra Jacobian block N = d(K_diff · U)/dU  −  K_diff.

    This is the correction term arising from differentiating a_ε(u_h) inside
    the SIPG diffusion operator.  Covers volume, interior SIPG face, and
    Nitsche BC face terms.

    The full diffusion Newton Jacobian is  K_diff(U) + N(U).
    """
    from .coefficients import da_eps_dz as _da_dz, a_eps as _a_eps
    from scipy.sparse import csr_matrix as _csr

    N_el  = mesh.N
    n_loc = p + 1
    n_dof = N_el * n_loc

    n_quad    = 2 * p + 4
    xi_q, w_q = gauss_legendre_ref(n_quad)
    phi_q     = basis.phi(xi_q)
    dphi_q    = basis.dphi_dxi(xi_q)    # reference-domain derivatives

    eta    = eta_C * (p + 1) ** 2
    phi_m1 = basis.phi(np.array([-1.0]))[0]
    phi_p1 = basis.phi(np.array([ 1.0]))[0]

    rows, cols, vals = [], [], []

    def _add(gi, gj, mat):
        ii, jj = np.meshgrid(gi, gj, indexing='ij')
        rows.extend(ii.ravel())
        cols.extend(jj.ravel())
        vals.extend(np.asarray(mat).ravel())

    # ── Volume ──────────────────────────────────────────────────────────────
    for K in range(N_el):
        xL, xR = mesh.element_interval(K)
        hK     = xR - xL
        x_q    = 0.5 * (xL + xR) + 0.5 * hK * xi_q

        U_K      = U[K * n_loc:(K + 1) * n_loc]
        u_q      = phi_q  @ U_K
        ux_ref_q = dphi_q @ U_K      # reference gradient (not scaled by 2/hK)

        da_q = np.asarray(_da_dz(u_q, x_q, t, Psi_func, sigma_H, sigma_L, eps))

        # N_vol[i,l] = (2/hK) * Σ_q w_q da_q φ_q[l] dφ_q[i] ux_ref_q
        N_K = (2.0 / hK) * np.einsum(
            'q,q,ql,qi,q->il', w_q, da_q, phi_q, dphi_q, ux_ref_q)
        _add(np.arange(K * n_loc, (K + 1) * n_loc),
             np.arange(K * n_loc, (K + 1) * n_loc), N_K)

    # ── Interior SIPG faces ──────────────────────────────────────────────────
    # a_e = a_ε(u_L at face) depends only on U_L → only (*, K_L) columns corrected.
    for e_idx in range(len(mesh.interior_face_indices)):
        face_idx = mesh.interior_face_indices[e_idx]
        K_L = mesh.face_left_element[e_idx]
        K_R = mesh.face_right_element[e_idx]

        x_e = float(mesh.faces[face_idx])
        h_L = mesh.h[K_L];  h_R = mesh.h[K_R];  h_e = min(h_L, h_R)

        U_L = U[K_L * n_loc:(K_L + 1) * n_loc]
        U_R = U[K_R * n_loc:(K_R + 1) * n_loc]
        u_L_f = float(phi_p1 @ U_L)

        da_e = float(_da_dz(np.array([u_L_f]), np.array([x_e]),
                             t, Psi_func, sigma_H, sigma_L, eps)[0])

        dphi_L = basis.dphi_dx(np.array([ 1.0]), h_L)[0]
        dphi_R = basis.dphi_dx(np.array([-1.0]), h_R)[0]

        u_R_f = float(phi_m1 @ U_R)
        ux_L  = float(dphi_L @ U_L)
        ux_R  = float(dphi_R @ U_R)
        dpen  = eta / h_e

        # Vectorised correction: N_AB[i,l] = da_e * φ_p1[l] * vec_AB[i]
        vec_LL = (-0.5*(phi_p1*ux_L + dphi_L*u_L_f) + dpen*u_L_f*phi_p1)
        vec_LR = (-0.5*(phi_p1*ux_R - dphi_L*u_R_f) - dpen*u_R_f*phi_p1)
        vec_RL = ( 0.5*(phi_m1*ux_L - dphi_R*u_L_f) - dpen*u_L_f*phi_m1)
        vec_RR = ( 0.5*(phi_m1*ux_R + dphi_R*u_R_f) + dpen*u_R_f*phi_m1)

        dofs_L = np.arange(K_L * n_loc, (K_L + 1) * n_loc)
        dofs_R = np.arange(K_R * n_loc, (K_R + 1) * n_loc)
        _add(dofs_L, dofs_L, da_e * np.outer(vec_LL + vec_LR, phi_p1))
        _add(dofs_R, dofs_L, da_e * np.outer(vec_RL + vec_RR, phi_p1))

    # ── Left Nitsche BC ──────────────────────────────────────────────────────
    x_bc_L = float(mesh.faces[0]);      h_bc_L = float(mesh.h[0])
    U_0    = U[:n_loc];                 u_0_f  = float(phi_m1 @ U_0)
    da_0   = float(_da_dz(np.array([u_0_f]), np.array([x_bc_L]),
                           t, Psi_func, sigma_H, sigma_L, eps)[0])
    dp_m1  = basis.dphi_dx(np.array([-1.0]), h_bc_L)[0]
    vec_0  = (phi_m1 * float(dp_m1 @ U_0) + dp_m1 * u_0_f
              + (eta / h_bc_L) * u_0_f * phi_m1)
    _add(np.arange(0, n_loc), np.arange(0, n_loc),
         da_0 * np.outer(vec_0, phi_m1))

    # ── Right Nitsche BC ─────────────────────────────────────────────────────
    x_bc_R = float(mesh.faces[N_el]);   h_bc_R = float(mesh.h[N_el - 1])
    U_N    = U[(N_el - 1) * n_loc:];    u_N_f  = float(phi_p1 @ U_N)
    da_N   = float(_da_dz(np.array([u_N_f]), np.array([x_bc_R]),
                           t, Psi_func, sigma_H, sigma_L, eps)[0])
    dp_p1  = basis.dphi_dx(np.array([1.0]), h_bc_R)[0]
    vec_N  = (-phi_p1 * float(dp_p1 @ U_N) - dp_p1 * u_N_f
              + (eta / h_bc_R) * u_N_f * phi_p1)
    _add(np.arange((N_el - 1) * n_loc, N_el * n_loc),
         np.arange((N_el - 1) * n_loc, N_el * n_loc),
         da_N * np.outer(vec_N, phi_p1))

    return _csr((vals, (rows, cols)), shape=(n_dof, n_dof))


# ══════════════════════════════════════════════════════════════════════════════
# Newton solver
# ══════════════════════════════════════════════════════════════════════════════

class NewtonSolver:
    """Backward Euler with Newton inner iteration and Armijo line search.

    The Newton Jacobian includes the analytical derivative of the SIPG
    diffusion operator (volume + face + Nitsche BC) with respect to U,
    computed from da_eps_dz.  Drift and reaction corrections are neglected
    (they are small compared to the diffusion correction near the free
    boundary).
    """

    def __init__(self, mesh, basis, config):
        self.mesh        = mesh
        self.basis       = basis
        self.eta_C       = config.get('penalty_C', 10)
        self.p           = basis.p
        self.tol         = config.get('newton_tol', 1e-8)
        self.max_iter    = config.get('max_newton_iter', 20)

    def solve_step(self, U_prev, t_n, dt, f_func, g_D_func,
                   r, Psi_func, Psi_x_func, sigma_H, sigma_L, eps,
                   M=None):
        """Newton iteration for one time step.  Returns (U_new, n_iters, converged)."""
        mesh  = self.mesh
        basis = self.basis
        eta_C = self.eta_C
        p     = self.p
        n_loc = p + 1
        N     = mesh.N
        tol   = self.tol

        if M is None:
            M = assemble_mass_matrix(mesh, basis)

        n_quad    = 2 * p + 4
        xi_q, w_q = gauss_legendre_ref(n_quad)
        phi_q     = basis.phi(xi_q)
        x_L_arr   = mesh.x[:-1]; x_R_arr = mesh.x[1:]
        h_K_arr   = x_R_arr - x_L_arr; J_K_arr = h_K_arr / 2.0
        x_phys    = (0.5*(x_L_arr[:,None]+x_R_arr[:,None])
                     + 0.5*h_K_arr[:,None]*xi_q[None,:])

        eta     = eta_C * (p + 1) ** 2
        x_bc_L  = float(mesh.faces[0]);  x_bc_R  = float(mesh.faces[N])
        h_bc_L  = float(mesh.h[0]);      h_bc_R  = float(mesh.h[N - 1])
        phi_m1  = basis.phi(np.array([-1.0]))[0]
        phi_p1  = basis.phi(np.array([ 1.0]))[0]
        dphi_m1 = basis.dphi_dx(np.array([-1.0]), h_bc_L)[0]
        dphi_p1 = basis.dphi_dx(np.array([ 1.0]), h_bc_R)[0]
        g_L     = float(g_D_func(x_bc_L, t_n))
        g_R     = float(g_D_func(x_bc_R, t_n))

        f_all = np.asarray(f_func(x_phys, t_n), dtype=float)
        F_vol = (J_K_arr[:,None]
                 * np.einsum('q,Kq,qi->Ki', w_q, f_all, phi_q)).ravel()
        b_mass = M @ U_prev

        def _bc_load(a_w, beta_w):
            a_bc_L = float(a_w(np.array([x_bc_L]))[0])
            a_bc_R = float(a_w(np.array([x_bc_R]))[0])
            pen_L  = eta * a_bc_L / h_bc_L
            pen_R  = eta * a_bc_R / h_bc_R
            F = F_vol.copy()
            if g_L:
                F[:n_loc]  += -(a_bc_L * dphi_m1 + pen_L * phi_m1) * g_L
            if g_R:
                F[-n_loc:] += ( a_bc_R * dphi_p1 - pen_R * phi_p1) * g_R
            F += assemble_drift_inflow_bc(mesh, basis, beta_w, g_D_func, t_n)
            return F

        U_k   = U_prev.copy()
        delta = np.zeros_like(U_k)

        for k in range(self.max_iter):
            a_w, beta_w, gamma_w = _build_frozen_callables(
                mesh, basis, U_k, t_n,
                r, Psi_func, Psi_x_func, sigma_H, sigma_L, eps,
            )
            K_k     = assemble_stiffness(mesh, basis, a_w, beta_w, gamma_w,
                                          r, eta_C, p, t_n)
            b_rhs_k = b_mass + dt * _bc_load(a_w, beta_w)
            G_k     = (M + dt * K_k) @ U_k - b_rhs_k
            G_norm  = float(np.linalg.norm(G_k))

            # Newton Jacobian
            N_k  = _assemble_newton_diffusion_correction(
                mesh, basis, U_k, t_n, Psi_func, sigma_H, sigma_L, eps,
                eta_C, p,
            )
            J_k  = (M + dt * (K_k + N_k)).tocsc()
            delta = splu(J_k).solve(-G_k)

            # ── Step-size safeguard (cheap, no extra K assembly) ──────
            # Dampen the Newton step if it is larger than 3× the current
            # solution norm — prevents blow-up far from the solution.
            step_norm = float(np.linalg.norm(delta))
            U_norm    = float(np.linalg.norm(U_k)) + 1e-14
            lam = min(1.0, 3.0 * U_norm / step_norm) if step_norm > 3.0 * U_norm else 1.0

            U_new = U_k + lam * delta

            rel = float(np.linalg.norm(lam * delta) / (np.linalg.norm(U_new) + 1e-14))
            if rel < tol:
                return U_new, k + 1, True

            U_k = U_new

        print(f"WARNING: Newton not converged at t={t_n:.4f}  "
              f"rel_update={np.linalg.norm(delta)/(np.linalg.norm(U_k)+1e-14):.2e}  "
              f"after {self.max_iter} iters.")
        return U_k, self.max_iter, False

    def solve(self, u0_func, f_func, g_D_func, T, dt,
              r=0.0, Psi_func=None, Psi_x_func=None,
              sigma_H=None, sigma_L=None, eps=None,
              diagnostics=None, on_step=None,
              newton_stats=None):
        """Full time integration with Newton inner loop."""
        if Psi_x_func is None:
            Psi_x_func = lambda x, t: np.zeros_like(np.asarray(x, float))

        mesh  = self.mesh
        basis = self.basis
        M     = assemble_mass_matrix(mesh, basis)
        U     = _project_l2(mesh, basis, u0_func)
        U0    = U.copy()

        n_steps = int(round(T / dt))

        for step in range(n_steps):
            t_n    = (step + 1) * dt
            U_prev = U.copy()

            U, n_iters, converged = self.solve_step(
                U_prev, t_n, dt, f_func, g_D_func,
                r, Psi_func, Psi_x_func, sigma_H, sigma_L, eps,
                M=M,
            )

            if newton_stats is not None:
                newton_stats.append({'n_iters': n_iters,
                                     'converged': converged, 't': t_n})

            if diagnostics is not None:
                diagnostics.record_step(step + 1, t_n, U, mesh, basis,
                                        U_prev=U_prev, dt=dt, M=M)

            if on_step is not None:
                on_step(step + 1, t_n, U)

        return [(0.0, U0), (T, U.copy())]
