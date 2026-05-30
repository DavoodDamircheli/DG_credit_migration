"""SIPG diffusion matrix, mass matrix, load vector, upwind drift, and reaction
assembly for 1D DG."""
import numpy as np
from scipy.sparse import csr_matrix
from .quadrature import gauss_legendre_ref


def assemble_mass_matrix(mesh, basis):
    """Global mass matrix M[i,j] = ∫ φ_i φ_j dx, block-diagonal over elements."""
    p = basis.p
    n_loc = p + 1
    N = mesh.N
    n_dof = N * n_loc

    n_quad = 2 * p + 4
    xi_q, w_q = gauss_legendre_ref(n_quad)
    phi_q = basis.phi(xi_q)  # (n_quad, p+1)

    rows, cols, vals = [], [], []
    for K in range(N):
        x_L, x_R = mesh.element_interval(K)
        J = (x_R - x_L) / 2.0
        M_loc = J * np.einsum('q,qi,qj->ij', w_q, phi_q, phi_q)
        off = K * n_loc
        ii, jj = np.meshgrid(np.arange(n_loc), np.arange(n_loc), indexing='ij')
        rows.extend((off + ii).ravel())
        cols.extend((off + jj).ravel())
        vals.extend(M_loc.ravel())

    return csr_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))


def assemble_sipg_diffusion(mesh, basis, a_func, eta_C, p):
    """SIPG diffusion matrix including Nitsche Dirichlet BC terms.

    Sign convention: [v]_e = v^- - v^+ at interior faces, where ^- is the left
    (lower-index) element and ^+ is the right element. At boundary faces, the
    interior element is always ^- and the ghost exterior value is ^+.
    """
    n_loc = p + 1
    N = mesh.N
    n_dof = N * n_loc

    n_quad = 2 * p + 4
    xi_q, w_q = gauss_legendre_ref(n_quad)
    dphi_q = basis.dphi_dxi(xi_q)  # (n_quad, p+1) reference derivatives

    eta = eta_C * (p + 1) ** 2

    # Face-endpoint basis values (shape: (p+1,))
    phi_at_m1 = basis.phi(np.array([-1.0]))[0]   # ξ = -1 (left end of element)
    phi_at_p1 = basis.phi(np.array([ 1.0]))[0]   # ξ = +1 (right end of element)

    rows, cols, vals = [], [], []

    def add(global_i, global_j, mat):
        """Append flattened (global_i x global_j) block."""
        ii, jj = np.meshgrid(global_i, global_j, indexing='ij')
        rows.extend(ii.ravel())
        cols.extend(jj.ravel())
        vals.extend(mat.ravel())

    # ------------------------------------------------------------------ #
    # Volume terms: ∫_K a(x) u'(x) v'(x) dx
    # ------------------------------------------------------------------ #
    for K in range(N):
        x_L, x_R = mesh.element_interval(K)
        h_K = x_R - x_L
        J = h_K / 2.0
        # Jacobian factor: J * (2/h_K)^2 = 2/h_K
        coef = 2.0 / h_K

        x_phys = 0.5 * (x_L + x_R) + 0.5 * h_K * xi_q
        a_vals = np.asarray(a_func(x_phys), dtype=float).ravel()

        K_vol = coef * np.einsum('q,q,qi,qj->ij', w_q, a_vals, dphi_q, dphi_q)

        dofs = np.arange(K * n_loc, (K + 1) * n_loc)
        add(dofs, dofs, K_vol)

    # ------------------------------------------------------------------ #
    # Interior face terms (consistency + symmetry + penalty)
    # ------------------------------------------------------------------ #
    for e_idx in range(len(mesh.interior_face_indices)):
        face_idx = mesh.interior_face_indices[e_idx]
        K_L = mesh.face_left_element[e_idx]
        K_R = mesh.face_right_element[e_idx]

        x_e = mesh.faces[face_idx]
        h_L = mesh.h[K_L]
        h_R = mesh.h[K_R]
        h_e = min(h_L, h_R)

        a_e = a_func(x_e)
        penalty = eta * a_e / h_e

        # Left element evaluated at ξ=+1 (right end), right element at ξ=-1 (left end)
        phi_L  = phi_at_p1
        phi_R  = phi_at_m1
        dphi_L = basis.dphi_dx(np.array([ 1.0]), h_L)[0]  # (p+1,) physical deriv
        dphi_R = basis.dphi_dx(np.array([-1.0]), h_R)[0]  # (p+1,)

        dofs_L = np.arange(K_L * n_loc, (K_L + 1) * n_loc)
        dofs_R = np.arange(K_R * n_loc, (K_R + 1) * n_loc)

        # Block (K_L, K_L): both test and trial in left element
        # [v_i]_e = phi_L[i], [u_j]_e = phi_L[j], {u_j'}_e = 0.5*dphi_L[j]
        K_LL = (-0.5 * a_e * np.outer(phi_L, dphi_L)
                - 0.5 * a_e * np.outer(dphi_L, phi_L)
                + penalty  * np.outer(phi_L, phi_L))

        # Block (K_L, K_R): test in left, trial in right
        # {u_j'}_e = 0.5*dphi_R[j], [u_j]_e = -phi_R[j]
        K_LR = (-0.5 * a_e * np.outer(phi_L, dphi_R)
                + 0.5 * a_e * np.outer(dphi_L, phi_R)
                - penalty  * np.outer(phi_L, phi_R))

        # Block (K_R, K_L): test in right, trial in left
        # [v_i]_e = -phi_R[i], [u_j]_e = phi_L[j]
        K_RL = (+ 0.5 * a_e * np.outer(phi_R, dphi_L)
                - 0.5 * a_e * np.outer(dphi_R, phi_L)
                - penalty  * np.outer(phi_R, phi_L))

        # Block (K_R, K_R): both test and trial in right element
        # [v_i]_e = -phi_R[i], [u_j]_e = -phi_R[j]
        K_RR = (+ 0.5 * a_e * np.outer(phi_R, dphi_R)
                + 0.5 * a_e * np.outer(dphi_R, phi_R)
                + penalty  * np.outer(phi_R, phi_R))

        add(dofs_L, dofs_L, K_LL)
        add(dofs_L, dofs_R, K_LR)
        add(dofs_R, dofs_L, K_RL)
        add(dofs_R, dofs_R, K_RR)

    # ------------------------------------------------------------------ #
    # Boundary face terms — Nitsche Dirichlet BC
    #
    # The outward unit normal from element K at its boundary face:
    #   LEFT  boundary (ξ=-1 of K=0):    n_K = -1  → signs FLIP vs. right
    #   RIGHT boundary (ξ=+1 of K=N-1):  n_K = +1
    #
    # General formula: -a(∇u·n_K)v - a(∇v·n_K)(u-g_D) + (η/h)(u-g_D)v
    #   n_K=-1 → +a u'v + a v'u + penalty uv  (LHS); RHS: -a v'g_D - penalty g_D v
    #   n_K=+1 → -a u'v - a v'u + penalty uv  (LHS); RHS: +a v'g_D - penalty g_D v
    # ------------------------------------------------------------------ #
    # Left boundary: element K=0 at ξ=-1, n_K = -1
    x_bc = mesh.faces[0]
    h_bc = mesh.h[0]
    a_bc = a_func(x_bc)
    penalty_bc = eta * a_bc / h_bc
    phi_bc   = phi_at_m1
    dphi_bc  = basis.dphi_dx(np.array([-1.0]), h_bc)[0]
    dofs_0   = np.arange(0, n_loc)

    K_bc_left = (+a_bc * np.outer(phi_bc, dphi_bc)   # sign flipped: n=-1
                 + a_bc * np.outer(dphi_bc, phi_bc)
                 + penalty_bc * np.outer(phi_bc, phi_bc))
    add(dofs_0, dofs_0, K_bc_left)

    # Right boundary: element K=N-1 at ξ=+1, n_K = +1
    x_bc = mesh.faces[N]
    h_bc = mesh.h[N - 1]
    a_bc = a_func(x_bc)
    penalty_bc = eta * a_bc / h_bc
    phi_bc   = phi_at_p1
    dphi_bc  = basis.dphi_dx(np.array([1.0]), h_bc)[0]
    dofs_N   = np.arange((N - 1) * n_loc, N * n_loc)

    K_bc_right = (-a_bc * np.outer(phi_bc, dphi_bc)   # n=+1, standard sign
                  - a_bc * np.outer(dphi_bc, phi_bc)
                  + penalty_bc * np.outer(phi_bc, phi_bc))
    add(dofs_N, dofs_N, K_bc_right)

    # scipy sums duplicate (i,j) entries automatically
    return csr_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))


def assemble_load_vector(mesh, basis, f_func, t,
                         a_func=None, g_D_func=None, eta_C=10, p=None):
    """Volume source F[i] = ∫ f(x,t) φ_i dx, plus Nitsche BC terms when provided.

    f_func must accept 2-D numpy arrays (shape (N, n_quad)) and return an
    array of the same shape.  a_func must accept 1-D arrays and return 1-D.
    """
    if p is None:
        p = basis.p
    n_loc = p + 1
    N = mesh.N

    n_quad = 2 * p + 4
    xi_q, w_q = gauss_legendre_ref(n_quad)
    phi_q = basis.phi(xi_q)  # (n_quad, p+1)

    # Batch all elements simultaneously  ─────────────────────────────────
    x_L = mesh.x[:-1]   # (N,)
    x_R = mesh.x[1:]    # (N,)
    h_K = x_R - x_L     # (N,)
    J_K = h_K / 2.0     # (N,)

    # Physical quadrature points for every element: shape (N, n_quad)
    x_phys = 0.5 * (x_L[:, None] + x_R[:, None]) + 0.5 * h_K[:, None] * xi_q[None, :]

    f_all = np.asarray(f_func(x_phys, t), dtype=float)   # (N, n_quad)

    # F_local[K, i] = J_K[K] * Σ_q w_q f(x_Kq) φ_i(ξ_q)
    F = (J_K[:, None] * np.einsum('q,Kq,qi->Ki', w_q, f_all, phi_q)).ravel()

    if a_func is None or g_D_func is None:
        return F

    eta = eta_C * (p + 1) ** 2
    phi_at_m1 = basis.phi(np.array([-1.0]))[0]
    phi_at_p1 = basis.phi(np.array([ 1.0]))[0]

    # Left boundary (n_K=-1): RHS = -a v'g_D - penalty*v*g_D
    x_bc = mesh.faces[0]
    h_bc = mesh.h[0]
    a_bc = a_func(x_bc)
    g_bc = g_D_func(x_bc, t)
    penalty_bc = eta * a_bc / h_bc
    dphi_bc = basis.dphi_dx(np.array([-1.0]), h_bc)[0]
    dofs_0 = np.arange(0, n_loc)
    F[dofs_0] += (-a_bc * dphi_bc - penalty_bc * phi_at_m1) * g_bc

    # Right boundary (n_K=+1): RHS = +a v'g_D - penalty*v*g_D
    x_bc = mesh.faces[N]
    h_bc = mesh.h[N - 1]
    a_bc = a_func(x_bc)
    g_bc = g_D_func(x_bc, t)
    penalty_bc = eta * a_bc / h_bc
    dphi_bc = basis.dphi_dx(np.array([1.0]), h_bc)[0]
    dofs_N = np.arange((N - 1) * n_loc, N * n_loc)
    F[dofs_N] += (+a_bc * dphi_bc - penalty_bc * phi_at_p1) * g_bc

    return F


def assemble_upwind_drift(mesh, basis, beta_func, gamma_func, t):
    """Upwind drift matrix for the nonconservative form β(x) ∂_x u_h.

    Bilinear form:
      c_h(u_h, v_h) = -Σ_K ∫_K β u_h v_h' dx
                     - Σ_K ∫_K γ u_h v_h dx        (γ = ∂_x β correction)
                     + Σ_e β̂_e u_h^up [v_h]_e       (upwind face flux)

    Face average: β̂_e = 0.5*(β^- + β^+).
    Jump: [v_h]_e = v_h^- - v_h^+ (left minus right trace).
    Upwind state: u^up = u^- if β̂_e ≥ 0, else u^+.

    Outflow boundary terms are included in the returned matrix.
    Inflow boundary Dirichlet contributions belong in the load vector;
    use assemble_drift_inflow_bc for those.
    """
    p = basis.p
    n_loc = p + 1
    N = mesh.N
    n_dof = N * n_loc

    n_quad = 2 * p + 4
    xi_q, w_q = gauss_legendre_ref(n_quad)
    phi_q  = basis.phi(xi_q)       # (n_quad, p+1)
    dphi_q = basis.dphi_dxi(xi_q)  # (n_quad, p+1) reference derivatives

    phi_at_m1 = basis.phi(np.array([-1.0]))[0]  # (p+1,)  ξ = -1
    phi_at_p1 = basis.phi(np.array([ 1.0]))[0]  # (p+1,)  ξ = +1

    rows, cols, vals = [], [], []

    def add(global_i, global_j, mat):
        ii, jj = np.meshgrid(global_i, global_j, indexing='ij')
        rows.extend(ii.ravel())
        cols.extend(jj.ravel())
        vals.extend(np.asarray(mat).ravel())

    # ------------------------------------------------------------------ #
    # Volume terms: -∫_K β u v_x dx  and  -∫_K γ u v dx
    # After change of variables ξ = 2(x-x_L)/h_K - 1, Jacobians cancel:
    #   ∫_K β u v_x dx = ∫_{-1}^1 β(x) φ_j (2/h_K) dφ_i/dξ (h_K/2) dξ
    #                  = Σ_q w_q β_q phi_j(ξ_q) dphi_i/dξ(ξ_q)
    # ------------------------------------------------------------------ #
    for K in range(N):
        x_L, x_R = mesh.element_interval(K)
        h_K = x_R - x_L
        J = h_K / 2.0
        x_phys = 0.5 * (x_L + x_R) + 0.5 * h_K * xi_q

        beta_q  = np.asarray(beta_func(x_phys),  dtype=float).ravel()
        gamma_q = np.asarray(gamma_func(x_phys), dtype=float).ravel()

        # K[i,j] = -Σ_q w_q β_q φ_j φ_i'  (i=test, j=trial; Jacobians cancel)
        K_vol   = -np.einsum('q,q,qj,qi->ij', w_q, beta_q,  phi_q, dphi_q)
        # K[i,j] = -J Σ_q w_q γ_q φ_j φ_i
        K_gamma = -J * np.einsum('q,q,qj,qi->ij', w_q, gamma_q, phi_q, phi_q)

        dofs = np.arange(K * n_loc, (K + 1) * n_loc)
        add(dofs, dofs, K_vol + K_gamma)

    # ------------------------------------------------------------------ #
    # Interior face upwind flux:  β̂_e * u^up * [v_h]_e
    # [v] = v^- - v^+  (left/K_L trace minus right/K_R trace)
    # ------------------------------------------------------------------ #
    for e_idx in range(len(mesh.interior_face_indices)):
        face_idx = mesh.interior_face_indices[e_idx]
        K_L = mesh.face_left_element[e_idx]
        K_R = mesh.face_right_element[e_idx]

        x_e = float(mesh.faces[face_idx])
        x_arr = np.array([x_e])
        beta_e = 0.5 * (float(np.asarray(beta_func(x_arr)).ravel()[0])
                        + float(np.asarray(beta_func(x_arr)).ravel()[0]))

        dofs_L = np.arange(K_L * n_loc, (K_L + 1) * n_loc)
        dofs_R = np.arange(K_R * n_loc, (K_R + 1) * n_loc)

        if beta_e >= 0:
            phi_up = phi_at_p1   # upwind state from left element (ξ=+1)
            dofs_up = dofs_L
        else:
            phi_up = phi_at_m1   # upwind state from right element (ξ=-1)
            dofs_up = dofs_R

        # v^- contribution (+): K[dofs_L, dofs_up] += β̂ outer(φ^-, φ^up)
        add(dofs_L, dofs_up,  beta_e * np.outer(phi_at_p1, phi_up))
        # v^+ contribution (-): K[dofs_R, dofs_up] -= β̂ outer(φ^+, φ^up)
        add(dofs_R, dofs_up, -beta_e * np.outer(phi_at_m1, phi_up))

    # ------------------------------------------------------------------ #
    # Boundary face outflow terms (inflow → load vector, see assemble_drift_inflow_bc)
    #
    # Left boundary (x_min): interior K=0 is to the RIGHT of the face.
    #   [v] = v^_ext - v^_int = -phi_at_m1   (exterior ghost = 0)
    #   outflow when β_L ≤ 0: u^up = interior trace, contributes to matrix.
    #
    # Right boundary (x_max): interior K=N-1 is to the LEFT of the face.
    #   [v] = v^_int - v^_ext = phi_at_p1
    #   outflow when β_R ≥ 0: u^up = interior trace, contributes to matrix.
    # ------------------------------------------------------------------ #
    x_L_bc  = float(mesh.faces[0])
    beta_L  = float(np.asarray(beta_func(np.array([x_L_bc]))).ravel()[0])
    dofs_0  = np.arange(0, n_loc)
    if beta_L <= 0:  # outflow: β̂ u^up [v] = β_L φ_m1[j] (-φ_m1[i])
        add(dofs_0, dofs_0, -beta_L * np.outer(phi_at_m1, phi_at_m1))

    x_R_bc  = float(mesh.faces[N])
    beta_R  = float(np.asarray(beta_func(np.array([x_R_bc]))).ravel()[0])
    dofs_N  = np.arange((N - 1) * n_loc, N * n_loc)
    if beta_R >= 0:  # outflow: β̂ u^up [v] = β_R φ_p1[j] φ_p1[i]
        add(dofs_N, dofs_N, beta_R * np.outer(phi_at_p1, phi_at_p1))

    return csr_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))


def assemble_drift_inflow_bc(mesh, basis, beta_func, g_D_func, t):
    """Load vector contribution from Dirichlet inflow BC for the drift operator.

    At inflow boundaries the upwind state is g_D.  The face flux
    β̂ g_D [v] is known data and must be moved to the right-hand side.

    Left  boundary: inflow when β(x_min) > 0  →  F[dofs_0] += β_L g_L φ(-1)
    Right boundary: inflow when β(x_max) < 0  →  F[dofs_N] -= β_R g_R φ(+1)
    """
    p = basis.p
    n_loc = p + 1
    N = mesh.N
    F = np.zeros(N * n_loc)

    phi_at_m1 = basis.phi(np.array([-1.0]))[0]
    phi_at_p1 = basis.phi(np.array([ 1.0]))[0]

    x_L_bc = float(mesh.faces[0])
    beta_L = float(np.asarray(beta_func(np.array([x_L_bc]))).ravel()[0])
    if beta_L > 0:
        g_L = float(g_D_func(x_L_bc, t))
        F[:n_loc] += beta_L * g_L * phi_at_m1

    x_R_bc = float(mesh.faces[N])
    beta_R = float(np.asarray(beta_func(np.array([x_R_bc]))).ravel()[0])
    if beta_R < 0:
        g_R = float(g_D_func(x_R_bc, t))
        F[-n_loc:] += -beta_R * g_R * phi_at_p1

    return F


def assemble_reaction(mesh, basis, r):
    """Reaction matrix: r_h(u_h, v_h) = r ∫ u_h v_h dx  =  r * M."""
    return r * assemble_mass_matrix(mesh, basis)


def assemble_stiffness(mesh, basis, a_func, beta_func, gamma_func, r, eta_C, p, t):
    """Full stiffness matrix K = K_diff(a) + K_drift(β, γ) + r * M."""
    K_diff  = assemble_sipg_diffusion(mesh, basis, a_func, eta_C, p)
    K_drift = assemble_upwind_drift(mesh, basis, beta_func, gamma_func, t)
    M       = assemble_mass_matrix(mesh, basis)
    return K_diff + K_drift + r * M
