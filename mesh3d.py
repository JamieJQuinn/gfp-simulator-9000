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
    def __init__(self, input_mesh, cell_size):
        self.input_mesh = input_mesh
        # Regularise input mesh
        regularise_mesh(self.input_mesh, cell_size)

        # Tetrahedralise input mesh
        self.tetmesh = tetrahedralise(self.input_mesh, cell_size)

        self.assembler = pymesh.Assembler(self.tetmesh)

        # save a list of boundary vertices
        self.boundary_mesh = compute_boundary_mesh(self.tetmesh)
        source_faces = self.boundary_mesh.get_attribute("face_sources").astype(int)
        self.boundary_vertices = np.unique(np.array(self.tetmesh.faces[source_faces]).flatten())

        self.n_vertices = self.tetmesh.num_vertices
        self.attributes_to_save = []

        self._faces = self.tetmesh.faces[source_faces]

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

    def write_input_mesh(self, filename):
        """Ouput only the input mesh"""
        self.write_arbitrary_mesh(self.input_mesh, filename, self.attributes_to_save)

    def write_tet_mesh(self, filename):
        """Ouput tetrahedralised mesh"""
        self.write_arbitrary_mesh(self.tetmesh, filename, self.attributes_to_save)

    def write_mesh(self, filename):
        """Default mesh output"""
        self.write_tet_mesh(filename)

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

    ### Attribute handling

    def set_attribute(self, name, value):
        """Sets attribute name to value"""
        if not self.tetmesh.has_attribute(name):
            self.tetmesh.add_attribute(name)
        self.tetmesh.set_attribute(name, value)
        if name not in self.attributes_to_save:
            self.attributes_to_save.append(name)

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

    def get_positions_from_tetra(self, tetra):
        """returns positions of all points making up tetra"""
        return self.get_pos(tetra[0]), self.get_pos(tetra[1]), self.get_pos(tetra[2]), self.get_pos(tetra[3])

    def get_pos(self, vertex):
        """returns position of vertex"""
        return self.vertices()[vertex]

    ### List getters

    def elements(self):
        """Getter for triangle list"""
        return self.tetmesh.voxels

    def vertices(self):
        """Getter for vertex list"""
        return self.tetmesh.vertices

    def faces(self):
        """Getter for boundary face list"""
        return self._faces

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

    def calculate_von_neumann_boundary(self, face, boundary_values):
        """Returns integral over edge of boundary function"""
        midpoint_value = (boundary_values[face[0]]
                          + boundary_values[face[1]]
                          + boundary_values[face[2]])/3.0
        return 1.0/3.0 * self.calculate_area_from_points(\
                                                        self.get_pos(face[0]),
                                                        self.get_pos(face[1]),
                                                        self.get_pos(face[2])) * midpoint_value

    def assemble_von_neumann_boundary(self, boundary_values):
        """Assembles von neumann vector"""
        boundary_vector = np.zeros(self.n_vertices)
        for face in self.faces():
            boundary_vector[face] += self.calculate_von_neumann_boundary(face, boundary_values)
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
        for element in self.elements():
            local_matrix = generate_local_matrix_fn(element)
            for local_i, i in enumerate(element):
                for local_j, j in enumerate(element):
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
        p1, p2, p3, p4 = self.get_positions_from_tetra(tetra)
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



def regularise_mesh(mesh, tol):
    """Takes mesh & resizes triangles to tol size"""
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, _info = pymesh.split_long_edges(mesh, tol)
    mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
    mesh, _info = pymesh.collapse_short_edges(mesh, tol, preserve_feature=True)

def tetrahedralise(mesh, cell_size):
    """Tetrahedralise input mesh"""
    return pymesh.tetrahedralize(mesh, cell_size)

def compute_boundary_mesh(mesh):
    """Returns outer hull of input mesh"""
    return pymesh.compute_outer_hull(mesh)
