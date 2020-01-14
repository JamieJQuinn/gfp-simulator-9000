"""Provides mesh class"""

import math
import numpy as np
import triangle as tr
import matplotlib.pyplot as plt
import matplotlib
import meshio
import pymesh

class Mesh3D:
    """Triangulated mesh"""
    def __init__(self, input_mesh):
        self.input_mesh = input_mesh
        self.tetmesh = None

        # Calculate structures on the reference tetrahedron
        self.gradN = np.array([[-1,-1,-1], [ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]])
        self.gradN_transpose = np.transpose(self.gradN)

        self.K11 = 0.5 * np.outer(self.gradN_transpose[0], self.gradN_transpose[0])
        self.K22 = 0.5 * np.outer(self.gradN_transpose[1], self.gradN_transpose[1])
        self.K33 = 0.5 * np.outer(self.gradN_transpose[2], self.gradN_transpose[2])
        self.K12 = 0.5 * np.outer(self.gradN_transpose[0], self.gradN_transpose[1])
        self.K13 = 0.5 * np.outer(self.gradN_transpose[0], self.gradN_transpose[2])
        self.K23 = 0.5 * np.outer(self.gradN_transpose[1], self.gradN_transpose[2])

    ### Mesh tools

    def resize_input_mesh(self, tol):
        """Takes mesh & resizes triangles to tol size"""
        self.input_mesh, __ = pymesh.remove_degenerated_triangles(self.input_mesh, 100);
        self.input_mesh, _info = pymesh.split_long_edges(self.input_mesh, tol)
        self.input_mesh, __ = pymesh.collapse_short_edges(self.input_mesh, 1e-6);
        self.input_mesh, _info = pymesh.collapse_short_edges(self.input_mesh, tol, preserve_feature=True)

    def tetrahedralise(self, CELL_SIZE):
        """Tetrahedralise the mesh"""
        self.tetmesh = pymesh.tetrahedralize(self.input_mesh, CELL_SIZE)

    def write_input_mesh(self, filename, attributes=[]):
        """Ouput only the input mesh"""
        self.write_arbitrary_mesh(self.input_mesh, filename, attributes)

    def write_tet_mesh(self, filename, attributes=[]):
        """Ouput tetrahedralised mesh"""
        self.write_arbitrary_mesh(self.tetmesh, filename, attributes)

    def write_mesh(self, filename, attributes=[]):
        """Default mesh output"""
        self.write_tet_mesh(filename, attributes)

    def write_arbitrary_mesh(self, mesh, filename, attributes=[]):
        """detects mesh type and outputs it"""
        points = mesh.vertices
        cells = {"triangle": mesh.faces}
        if mesh.num_voxels>0:
            cells["tetra"] = mesh.voxels

        point_data = {name:mesh.get_attribute(name) for name in attributes}

        meshio.write_points_cells(
            filename,
            points,
            cells,
            # Optionally provide extra data on points, cells, etc.
            point_data=point_data
            # cell_data=cell_data,
            # field_data=field_data
        )

    def plot_mesh(self):
        ax = plt.axes(projection='3d')
        x = self.tetmesh.vertices[:,0]
        y = self.tetmesh.vertices[:,1]
        z = self.tetmesh.vertices[:,2]
        triangles = self.tetmesh.faces[:]

        ax.plot_trisurf(x,z,triangles,y, shade=True, color='white')

        ax.set_ylim(-2,2)
        ax.set_xlim(-2, 2)
        ax.set_zlim(0,4)

        plt.show()

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

    def calculate_stiffness(self, tetra):
        """Calculate local stiff matrix for triangle"""
        p1, p2, p3, p4 = self.get_vertices_from_tetra(tetra)
        BK = np.array([p2-p1, p3-p1, p4-p1])
        BK_inv = np.linalg.inv(BK)

        C_K = BK_inv.dot(np.transpose(BK_inv))
        detB = np.linalg.det(BK)

        stiffness = detB * (C_K[0,0]*self.K11 + C_K[1,1]*self.K22 + C_K[2,2]*self.K33
                            + C_K[0,1]*(self.K12 + np.transpose(self.K12))
                            + C_K[0,2]*(self.K13 + np.transpose(self.K13))
                            + C_K[1,2]*(self.K23 + np.transpose(self.K23))
                           )
        return stiffness

    def assemble_mass(self):
        """Assemble global mass matrix"""
        return self.assemble_matrix(self.calculate_mass)

    def assemble_stiffness(self):
        """Assemble global stiffness matrix"""
        return self.assemble_matrix(self.calculate_stiffness)
