"""Provides mesh class"""

import math
import numpy as np
import triangle as tr
import matplotlib.pyplot as plt

class Mesh2D:
    """Triangulated mesh"""
    def __init__(self, n_points, min_size=0.05, min_angle=20):
        self.min_size = min_size
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

    def plot(self, soln, axis=None, show_colourbar=False, vmax=None, vmin=None):
        """plot soln on mesh"""
        x, y = np.transpose(self.vertices())
        triangles = self.triangles()
        if axis:
            c = axis.tripcolor(x, y, triangles, soln, shading='gouraud', vmax=vmax, vmin=vmin)
        else:
            c = plt.tripcolor(x, y, triangles, soln, shading='gouraud', vmax=vmax, vmin=vmin)
        axis.set_yticklabels([])
        axis.set_xticklabels([])
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set(adjustable='box', aspect='equal')
        axis.axis('off')
        if show_colourbar:
            plt.colorbar(c, ax=axis, orientation='horizontal')
        return c

    def plot_triple(self, interior_soln, boundary_soln, axes):
        """ Plot interior, boundary and combined on separate plots"""
        if len(axes) != 3:
            return -1

        self.plot(interior_soln, axes[0], show_colourbar=True, vmin=0.0)
        self.plot(boundary_soln, axes[1], show_colourbar=True, vmin=0.0)
        combined_soln = interior_soln + boundary_soln
        self.plot(combined_soln, axes[2], show_colourbar=True, vmin=0.0)


    ### Area calculation

    def calculate_area(self, triangle):
        """Returns area of triangle"""
        p1, p2, p3 = self.get_vertices_from_triangle(triangle)
        return self.calculate_area_from_points(p1, p2, p3)

    def calculate_area_from_points(self, p1, p2, p3):
        """Returns area of triangle from points"""
        return 0.5*abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))

    def integrate(self, fn):
        sum = 0.0
        for tri in self.triangles():
            sum = sum + 1.0/3.0 * self.calculate_area(tri) * np.sum(fn[tri])
        return sum

    ### Getters

    def get_vertices_from_triangle(self, triangle):
        """Getter for vertices of triangle"""
        return self.get_pos(triangle[0]), self.get_pos(triangle[1]), self.get_pos(triangle[2])

    def get_pos(self, vertex):
        """returns position of vertex"""
        return self.vertices()[vertex]

    ### List getters

    def triangles(self):
        """Getter for triangle list"""
        return self.triangulation['triangles']

    def vertices(self):
        """Getter for vertex list"""
        return self.triangulation['vertices']

    def edges(self):
        """Getter for edge list"""
        return self.get_edges()

    ### Dirichlet boundary calculation

    def calculate_boundary_values(self, boundary_fn):
        """Returns vector of boundary_fn acted on all boundary points"""
        return list(map(lambda x: boundary_fn(self.get_pos(x)), self.boundary_vertices))

    ### Edge detection and utilities

    def contains_edge(self, triangle):
        """Does a triangle contain two edge points?"""
        edge_markers = self.get_markers(triangle)
        return len(edge_markers[edge_markers == 1]) == 2

    def get_markers(self, triangle):
        """Return vertex markers for given triangle"""
        return np.transpose(self.triangulation['vertex_markers'][triangle])[0]

    def get_edge_from_triangle(self, triangle):
        """Returns a list of edge points from a triangle"""
        return triangle[self.get_markers(triangle) == 1]

    def get_edges(self):
        """Returns a list of edges"""
        edge_triangles = filter(self.contains_edge, self.triangles())
        edges = list(map(self.get_edge_from_triangle, edge_triangles))
        return edges

    def calculate_edge_length(self, edge):
        """Returns length of edge"""
        p1, p2 = self.vertices()[edge]
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def integrate_around_edge(self, fn):
        # Note: fn is in full element representation
        integration_sum = 0.0
        for edge in self.edges():
            integration_sum += 0.5 * self.calculate_edge_length(edge) * np.sum(fn[edge])
        return integration_sum

    ### von Neumann assembly

    def calculate_von_neumann_boundary(self, edge, boundary_values):
        """Returns integral over edge of boundary function"""
        midpoint_value = (boundary_values[edge[0]] + boundary_values[edge[1]])/2.0
        return 1.0/2.0 * self.calculate_edge_length(edge) * midpoint_value

    def assemble_von_neumann_boundary(self, boundary_values):
        """Assembles von neumann vector"""
        boundary_vector = np.zeros(self.n_vertices)
        edges = self.get_edges()
        for edge in edges:
            boundary_vector[edge] += self.calculate_von_neumann_boundary(edge, boundary_values)
        return boundary_vector

    ### Body force assembly

    def assemble_body_force(self, force_fn):
        """Assembles body force vector"""
        force_vector = np.zeros(self.n_vertices)
        for triangle in self.triangles():
            area = self.calculate_area(triangle)
            centre_force = 0.0
            for index in triangle:
                centre_force += 1/3.0 * force_fn(self.get_pos(index))
            force_vector[triangle] += 1/3.0 * area * centre_force
        return force_vector

    ### Mass and stiffness assembly

    def assemble_matrix(self, generate_local_matrix_fn):
        """Generic global matrix assembly for both mass and stiffness"""
        matrix = np.zeros((self.n_vertices, self.n_vertices))
        for triangle in self.triangles():
            local_matrix = generate_local_matrix_fn(triangle)
            for local_i, i in enumerate(triangle):
                for local_j, j in enumerate(triangle):
                    matrix[i, j] += local_matrix[local_i, local_j]

        return matrix

    def calculate_mass(self, triangle):
        """Calculates local mass matrix"""
        return 1/12.0 * self.calculate_area(triangle) * np.array(
            [[2, 1, 1],
             [1, 2, 1],
             [1, 1, 2]])

    def calculate_stiffness(self, triangle):
        """Calculate local stiff matrix for triangle"""
        p1, p2, p3 = self.get_vertices_from_triangle(triangle)
        B_K_inv_trans = 1.0/(2.0*self.calculate_area(triangle)) * \
                np.array([[  p3[1] - p1[1], -(p2[1] - p1[1])],
                          [-(p3[0] - p1[0]),  p2[0] - p1[0]]])

        K11 = 1.0/2.0*np.array(\
                               [[1, -1, 0],
                                [-1, 1, 0],
                                [0, 0, 0 ]])
        K22 = 1.0/2.0*np.array(\
                               [[1, 0, -1],
                                [0, 0, 0],
                                [-1, 0, 1 ]])
        K12 = 1.0/2.0*np.array(\
                               [[1, 0, -1],
                                [-1, 0, 1],
                                [0, 0, 0 ]])

        C_K = np.transpose(B_K_inv_trans).dot(B_K_inv_trans)

        detB = 2.0*self.calculate_area(triangle)
        stiffness = detB * (C_K[0,0]*K11 + C_K[1,1]*K22 + C_K[0,1]*(K12 + np.transpose(K12)))
        return stiffness

    def assemble_mass(self):
        """Assemble global mass matrix"""
        return self.assemble_matrix(self.calculate_mass)

    def assemble_stiffness(self):
        """Assemble global stiffness matrix"""
        return self.assemble_matrix(self.calculate_stiffness)
