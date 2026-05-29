import numpy as np


class Mesh1D:
    """Uniform 1D mesh on [x_min, x_max] with N_elements elements."""

    def __init__(self, x_min, x_max, N_elements):
        self.N = N_elements
        self.x = np.linspace(x_min, x_max, N_elements + 1)
        self.h = np.diff(self.x)
        self.faces = self.x.copy()

        self.left_boundary_face = 0
        self.right_boundary_face = N_elements

        # Interior face e (1 <= e <= N-1): left element e-1, right element e
        self.interior_face_indices = np.arange(1, N_elements)
        self.face_left_element = self.interior_face_indices - 1
        self.face_right_element = self.interior_face_indices

    def element_midpoint(self, K):
        return 0.5 * (self.x[K] + self.x[K + 1])

    def element_interval(self, K):
        return self.x[K], self.x[K + 1]
