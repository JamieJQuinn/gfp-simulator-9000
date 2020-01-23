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

    def calculate_area_from_points(self, p1, p2, p3):
        """Returns area of triangle from points"""
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        return 0.5*np.linalg.norm(np.cross(v1, v2))

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
