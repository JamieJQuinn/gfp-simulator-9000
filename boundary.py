import numpy as np

class Boundary:
    """class encapsulating the cell boundary and its evolution"""
    def __init__(self, n_elements, boundary_indices, km, kp):
        self.n_elements = n_elements
        self.bi = boundary_indices
        self.active_rac = np.zeros(n_elements)
        self.bound_gfp = np.zeros(n_elements)
        self.prev_ddt = np.zeros(n_elements)
        self.ddt = np.zeros(n_elements)
        self.km = km
        self.kp = kp

    def reset(self):
        self.active_rac = np.zeros(self.n_elements)
        self.bound_gfp = np.zeros(self.n_elements)
        self.prev_ddt = np.zeros(self.n_elements)
        self.ddt = np.zeros(self.n_elements)

    def set_active_rac(self, rac):
        self.active_rac[:] = rac

    def calc_kinetic_eqn(self, u):
        """Returns result of kinetic equation"""
        return self.kp * u[self.bi] * (self.active_rac[self.bi] - self.bound_gfp[self.bi]) \
             - self.km * self.bound_gfp[self.bi]

    def calc_ddt(self, u):
        """Uses Adams-Bashforth to calculated d/dt form kinetic equation"""
        self.ddt[self.bi] = 3.0/2.0 * self.calc_kinetic_eqn(u) - 1.0/2.0 * self.prev_ddt[self.bi]
        return self.ddt

    def update(self, dt):
        """Update state of boundary"""
        self.bound_gfp[self.bi] += dt*self.ddt[self.bi]
        self.prev_ddt[self.bi] = self.ddt[self.bi]
