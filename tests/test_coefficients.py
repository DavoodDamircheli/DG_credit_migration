"""Tests for regularized coefficients a_eps, beta_eps, gamma_eps."""
import numpy as np
import pytest

from src.coefficients import (H_eps, dH_eps, d2H_eps, sigma_eps, a_eps,
                               da_eps_dz, b_eps, beta_eps,
                               gamma_eps_elementwise)
from src.mesh import Mesh1D
from src.basis import LagrangeBasis
from src.quadrature import gauss_legendre_ref


# ── Test 1: H_eps limits ──────────────────────────────────────────────────


def test_H_eps_limits():
    assert H_eps(np.array([-10.0]), 0.1) == pytest.approx(0.0, abs=1e-9)
    assert H_eps(np.array([ 10.0]), 0.1) == pytest.approx(1.0, abs=1e-9)
    # H_eps(0, eps) = 0.5 exactly because tanh(0) = 0 in IEEE 754
    assert H_eps(np.array([0.0]), 0.5) == pytest.approx(0.5, abs=1e-15)


# ── Test 2: dH_eps finite difference ─────────────────────────────────────


@pytest.mark.parametrize("s", [0.0, 0.1, -0.3])
def test_dH_eps_fd(s):
    eps = 0.2
    delta = 1e-6
    fd = (H_eps(s + delta, eps) - H_eps(s - delta, eps)) / (2.0 * delta)
    assert dH_eps(np.array([s]), eps)[0] == pytest.approx(float(fd), rel=1e-5)


# ── Test 3: sigma_eps range ───────────────────────────────────────────────


def test_sigma_eps_range():
    z = np.linspace(-2.0, 2.0, 50)
    x = np.zeros(50)
    Psi_func = lambda x, t: np.zeros_like(np.asarray(x, float))
    sig = sigma_eps(z, x, 0.0, Psi_func, sigma_H=0.4, sigma_L=0.1, eps=0.1)
    lo, hi = min(0.1, 0.4), max(0.1, 0.4)
    assert np.all(sig >= lo - 1e-12)
    assert np.all(sig <= hi + 1e-12)


# ── Test 4: a_eps positivity ──────────────────────────────────────────────


def test_a_eps_positive():
    z = np.linspace(-3.0, 3.0, 100)
    x = np.zeros(100)
    Psi_func = lambda x, t: np.zeros_like(np.asarray(x, float))
    a = a_eps(z, x, 0.0, Psi_func, sigma_H=0.4, sigma_L=0.1, eps=0.05)
    assert np.all(a > 0.0)


# ── Test 5: beta_eps constant-state check ─────────────────────────────────


def test_beta_eps_constant_state():
    """For w = Psi + delta (constant, above transition), q = 0, Psi_x = 0:
       beta_eps = da_eps_dz * 0 - (r - a_eps) = a_eps - r.
    """
    sigma_H, sigma_L, eps, r = 0.4, 0.1, 0.05, 0.02
    delta = 0.5   # well above Psi=0, so H_eps ≈ 1 and da_eps_dz ≈ 0

    Psi_func = lambda x, t: np.zeros_like(np.asarray(x, float))

    z     = np.array([delta])
    x     = np.array([0.5])
    q     = np.zeros(1)
    Psi_x = np.zeros(1)

    a_val    = a_eps(z, x, 0.0, Psi_func, sigma_H, sigma_L, eps)
    expected = a_val - r
    result   = beta_eps(z, q, Psi_x, x, 0.0, r, Psi_func, sigma_H, sigma_L, eps)

    assert result == pytest.approx(expected, rel=1e-10)


# ── Test 6: gamma_eps consistency — beta linear in x → gamma ≈ c ─────────


def test_gamma_eps_linear():
    """Engineer inputs so beta_eps(w, wx, Psi_x, x, t, r, ...) = c * x.

    Strategy: with z = Psi = 0, Psi_x = 0 and eps large (smooth regime),
    da_0 and b_0 are constants.  Setting wx = (c * x + b_0) / da_0 makes
    beta_eps = da_0 * wx - b_0 = c * x.

    gamma_eps_elementwise L2-projects c*x (degree 1) into P^p with p >= 1 —
    the projection is exact — then differentiates analytically to give c.
    """
    p     = 2
    N     = 4
    mesh  = Mesh1D(0.0, 1.0, N)
    xi_q_basis, _ = gauss_legendre_ref(2 * p + 4)
    basis = LagrangeBasis(p, xi_q_basis)

    sigma_H, sigma_L, eps, r, c = 0.4, 0.1, 1.0, 0.02, 3.0
    Psi_func   = lambda x, t: np.zeros_like(np.asarray(x, float))
    Psi_x_func = lambda x, t: np.zeros_like(np.asarray(x, float))

    K_elem = 1   # second element: [0.25, 0.50]
    x_L, x_R = mesh.element_interval(K_elem)
    h_K = x_R - x_L
    n_quad = 2 * p + 4
    xi_q2, _ = gauss_legendre_ref(n_quad)
    x_vals = 0.5 * (x_L + x_R) + 0.5 * h_K * xi_q2   # physical quad points

    # At z = 0 = Psi with eps = 1.0:
    #   sigma_0 = 0.4 + (0.1 - 0.4)*H_eps(0, 1) = 0.4 - 0.3*0.5 = 0.25
    #   da_0    = 0.25 * (0.1 - 0.4) * dH_eps(0, 1) = 0.25*(-0.3)*0.5 = -0.0375
    #   b_0     = r - 0.5*0.25^2 = 0.02 - 0.03125 = -0.01125
    z_vals = np.zeros(n_quad)
    sig0 = sigma_H + (sigma_L - sigma_H) * H_eps(z_vals, eps)
    da_0 = sig0 * (sigma_L - sigma_H) * dH_eps(z_vals, eps)
    b_0  = r - 0.5 * sig0 ** 2

    # wx chosen so that beta_eps = c * x_vals:
    #   beta = da_0 * wx - b_0  =>  wx = (c*x + b_0) / da_0
    wx_vals = (c * x_vals + b_0) / da_0

    gamma = gamma_eps_elementwise(z_vals, wx_vals, x_vals, 0.0, mesh, basis,
                                  K_elem, r, Psi_func, Psi_x_func,
                                  sigma_H, sigma_L, eps)

    assert gamma == pytest.approx(c * np.ones(n_quad), abs=1e-10)
