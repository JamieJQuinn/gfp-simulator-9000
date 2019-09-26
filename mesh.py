"""Provides mesh class"""

import numpy as np
import triangle as tr
import matplotlib.pyplot as plt

class Mesh:
    """Triangulated mesh"""
    def __init__(self, n_points, min_size=0.05, min_angle=20):
        self.n_boundary_points = n_points
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        attributes = theta/(2*np.pi)

        vertices = dict(vertices=pts, vertex_attributes=attributes)
        self.triangulation = tr.triangulate(vertices,
                                            'q' + str(min_angle) + 'a' + str(min_size))
        self.n_triangles = len(self.triangulation['triangles'])
        self.n_vertices = len(self.triangulation['vertices'])
        self.boundary_vertices = np.where(self.triangulation['vertex_markers'] == [1])[0]
        self.boundary_info = {}

    def get_interp(self, vertex):
        """returns t in interpolation along boundary"""
        return self.triangulation['vertex_attributes'][vertex]

    def get_pos(self, vertex):
        """returns position of vertex"""
        return self.vertices()[vertex]

    def plot(self, soln, axis=None):
        """plot soln on mesh"""
        x, y = np.transpose(self.vertices())
        triangles = self.triangles()
        if axis:
            c = axis.tripcolor(x, y, triangles, soln)
        else:
            c = plt.tripcolor(x, y, triangles, soln)
        # plt.colorbar(c)

    def calculate_area(self, triangle):
        """Returns area of triangle"""
        p1, p2, p3 = self.get_vertices_from_triangle(triangle)
        return self.calculate_area_from_points(p1, p2, p3)

    def calculate_area_from_points(self, p1, p2, p3):
        """Returns area of triangle from points"""
        return 0.5*((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))

    def get_vertices_from_triangle(self, triangle):
        """Getter for vertices of triangle"""
        return self.get_pos(triangle[0]), self.get_pos(triangle[1]), self.get_pos(triangle[2])

    def calculate_stiffness(self, triangle):
        """Calculate local stiff matrix for triangle"""
        p1, p2, p3 = self.get_vertices_from_triangle(triangle)
        # Inverse of A' gives a + bx + cy representation of point
        A = [[1    , 1    , 1    ],
             [p1[0], p2[0], p3[0]],
             [p1[1], p2[1], p3[1]]]
        G = np.linalg.inv(np.transpose(A))

        # Take only b anc c
        grad = G[1:, :]

        tri_area = self.calculate_area_from_points(p1, p2, p3)
        stiffness = 0.5 * tri_area * np.transpose(grad).dot(grad)
        return stiffness

    def triangles(self):
        """Getter for triangle list"""
        return self.triangulation['triangles']

    def vertices(self):
        """Getter for vertex list"""
        return self.triangulation['vertices']

    def assemble_stiffness(self):
        """Assemble global stiffness matrix"""
        stiffness = np.zeros((self.n_vertices, self.n_vertices))
        for triangle in self.triangles():
            local_stiffness = self.calculate_stiffness(triangle)
            for local_i, i in enumerate(triangle):
                for local_j, j in enumerate(triangle):
                    stiffness[i, j] += local_stiffness[local_i, local_j]

        return stiffness
