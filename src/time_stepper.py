"""Frozen-coefficient backward Euler time stepper for DG diffusion."""
import numpy as np
from scipy.sparse.linalg import splu
from .quadrature import gauss_legendre_ref
from .dg_operators import (assemble_mass_matrix, assemble_sipg_diffusion,
                            assemble_load_vector)


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

    def solve(self, u0_func, f_func, a_func, g_D_func, T, dt):
        """Frozen-coefficient backward Euler for the heat equation.

        Solves (M + dt K_diff) U^n = M U^{n-1} + dt F^n at each step.
        a_func is assumed constant in time — K_diff is assembled once.

        f_func(x, t) and a_func(x) must accept 2-D / 1-D numpy arrays.

        Returns list [(0, U^0), (T, U^T)].
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

        return [(0.0, U0), (T, U.copy())]
