import numpy as np
from .quadrature import gauss_legendre_ref


def gauss_lobatto_points(n):
    """n Gauss-Lobatto-Legendre points on [-1, 1], hardcoded for n = 2..5."""
    if n == 2:
        return np.array([-1.0, 1.0])
    elif n == 3:
        return np.array([-1.0, 0.0, 1.0])
    elif n == 4:
        # Interior roots of P'_3: xi = ±1/sqrt(5)
        return np.array([-1.0, -1.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0), 1.0])
    elif n == 5:
        # Interior roots of P'_4: xi = 0, ±sqrt(3/7)
        s = np.sqrt(3.0 / 7.0)
        return np.array([-1.0, -s, 0.0, s, 1.0])
    else:
        raise ValueError(f"gauss_lobatto_points: n={n} not supported (must be 2..5)")


class LagrangeBasis:
    """Nodal Lagrange basis of degree p at Gauss-Lobatto-Legendre points."""

    def __init__(self, p, quadrature_points_ref):
        self.p = p
        self.nodes = gauss_lobatto_points(p + 1)
        self.n_nodes = p + 1
        self.quad_pts = np.asarray(quadrature_points_ref)

    def phi(self, xi):
        """Lagrange basis values. Returns array of shape (len(xi), p+1)."""
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        result = np.ones((len(xi), self.n_nodes))
        for j in range(self.n_nodes):
            for k in range(self.n_nodes):
                if k != j:
                    result[:, j] *= (xi - self.nodes[k]) / (self.nodes[j] - self.nodes[k])
        return result

    def dphi_dxi(self, xi):
        """Reference derivatives dφ_j/dξ. Returns array of shape (len(xi), p+1).

        Uses the standard product-rule sum for Lagrange derivatives:
          φ'_j(ξ) = Σ_{m≠j} [1/(ξ_j-ξ_m)] * Π_{k≠j,k≠m} (ξ-ξ_k)/(ξ_j-ξ_k)
        """
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        result = np.zeros((len(xi), self.n_nodes))
        for j in range(self.n_nodes):
            for m in range(self.n_nodes):
                if m == j:
                    continue
                term = np.full(len(xi), 1.0 / (self.nodes[j] - self.nodes[m]))
                for k in range(self.n_nodes):
                    if k != j and k != m:
                        term *= (xi - self.nodes[k]) / (self.nodes[j] - self.nodes[k])
                result[:, j] += term
        return result

    def dphi_dx(self, xi, h_K):
        """Physical derivatives dφ_j/dx = (2/h_K) * dφ_j/dξ. Returns (len(xi), p+1)."""
        return (2.0 / h_K) * self.dphi_dxi(xi)


def local_mass_matrix(p, n_quad_points, h_K=1.0):
    """Local DG mass matrix for a single element of length h_K.

    M_ij = (h_K/2) * Σ_q w_q φ_i(ξ_q) φ_j(ξ_q)

    Returns array of shape (p+1, p+1).
    """
    xi, w = gauss_legendre_ref(n_quad_points)
    basis = LagrangeBasis(p, xi)
    phi_vals = basis.phi(xi)                               # (n_quad, p+1)
    M = np.einsum('q,qi,qj->ij', w, phi_vals, phi_vals)
    return (h_K / 2.0) * M
