"""Regularized PDE coefficients for the Wirth free-boundary model."""
import numpy as np
from .quadrature import gauss_legendre_ref


def H_eps(s, eps):
    """Smooth Heaviside: 0.5 * (1 + tanh(s / eps)).

    Clips s/eps to ±500 to avoid overflow in tanh for large arguments.
    """
    s = np.asarray(s, dtype=float)
    return 0.5 * (1.0 + np.tanh(np.clip(s / eps, -500.0, 500.0)))


def dH_eps(s, eps):
    """H_ε'(s) = sech²(s/eps) / (2*eps)."""
    s = np.asarray(s, dtype=float)
    s_safe = np.clip(s / eps, -500.0, 500.0)
    c = np.cosh(s_safe)
    return 0.5 / (eps * c * c)


def d2H_eps(s, eps):
    """H_ε''(s) = -tanh(s/eps) * sech²(s/eps) / eps²."""
    s = np.asarray(s, dtype=float)
    s_safe = np.clip(s / eps, -500.0, 500.0)
    t = np.tanh(s_safe)
    c = np.cosh(s_safe)
    return -t / (eps * eps * c * c)


def sigma_eps(z, x, t, Psi_func, sigma_H, sigma_L, eps):
    """σ_ε(z, x, t) = σ_H + (σ_L - σ_H) * H_ε(z - Ψ(x,t))."""
    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float)
    Psi = np.asarray(Psi_func(x, t), dtype=float)
    return sigma_H + (sigma_L - sigma_H) * H_eps(z - Psi, eps)


def a_eps(z, x, t, Psi_func, sigma_H, sigma_L, eps):
    """a_ε = 0.5 * σ_ε(z,x,t)²."""
    sig = sigma_eps(z, x, t, Psi_func, sigma_H, sigma_L, eps)
    return 0.5 * sig * sig


def da_eps_dz(z, x, t, Psi_func, sigma_H, sigma_L, eps):
    """∂_z a_ε = σ_ε * (σ_L - σ_H) * H_ε'(z - Ψ)."""
    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float)
    Psi = np.asarray(Psi_func(x, t), dtype=float)
    sig = sigma_eps(z, x, t, Psi_func, sigma_H, sigma_L, eps)
    return sig * (sigma_L - sigma_H) * dH_eps(z - Psi, eps)


def b_eps(z, x, t, r, Psi_func, sigma_H, sigma_L, eps):
    """b_ε(z,x,t) = r - a_ε(z,x,t)."""
    return r - a_eps(z, x, t, Psi_func, sigma_H, sigma_L, eps)


def beta_eps(z, q, Psi_x, x, t, r, Psi_func, sigma_H, sigma_L, eps):
    """β_ε(z, q, x, t) = ∂_z a_ε(z,x,t) * (q - Ψ_x(x,t)) - b_ε(z,x,t).

    z     : DG solution values at quadrature points (frozen state w)
    q     : DG gradient values ∂_x^h w at quadrature points
    Psi_x : ∂_x Ψ(x,t) at quadrature points
    """
    da = da_eps_dz(z, x, t, Psi_func, sigma_H, sigma_L, eps)
    b = b_eps(z, x, t, r, Psi_func, sigma_H, sigma_L, eps)
    return da * (np.asarray(q, dtype=float) - np.asarray(Psi_x, dtype=float)) - b


def gamma_eps_elementwise(w_vals, wx_vals, x_vals, t, mesh, basis, K, r,
                           Psi_func, Psi_x_func, sigma_H, sigma_L, eps):
    """γ_w = ∂_x β_w on element K via L2 projection + differentiation.

    Strategy: evaluate β_w at the n_quad Gauss–Legendre quadrature points
    already computed on element K, L2-project those values onto P^p(K),
    then differentiate the resulting polynomial analytically.  This avoids
    computing explicit second derivatives of U^{n-1} and stays consistent
    with the quadrature rule used in the upwind drift assembler.

    Returns γ_w values at the same n_quad quadrature points.
    """
    x_L, x_R = mesh.element_interval(K)
    h_K = x_R - x_L
    J = h_K / 2.0
    n_quad = len(x_vals)

    # Reference quadrature coordinates recovered from physical points
    xi_q = 2.0 * (x_vals - x_L) / h_K - 1.0
    _, w_q = gauss_legendre_ref(n_quad)   # weights matching this xi_q

    phi_q  = basis.phi(xi_q)              # (n_quad, n_loc)
    dphi_q = basis.dphi_dxi(xi_q)         # (n_quad, n_loc)

    Psi_x_vals = np.asarray(Psi_x_func(x_vals, t), dtype=float)
    beta_q = beta_eps(w_vals, wx_vals, Psi_x_vals, x_vals, t, r,
                      Psi_func, sigma_H, sigma_L, eps)

    # L2 projection: solve M_loc * c = b
    M_loc = J * np.einsum('q,qi,qj->ij', w_q, phi_q, phi_q)
    b_loc = J * np.einsum('q,q,qi->i',  w_q, beta_q, phi_q)
    c = np.linalg.solve(M_loc, b_loc)

    # ∂_x (Σ_j c_j φ_j) = (2/h_K) Σ_j c_j dφ_j/dξ
    return (2.0 / h_K) * (dphi_q @ c)    # (n_quad,)
