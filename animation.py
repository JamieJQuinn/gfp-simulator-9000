import matplotlib.pyplot as plt
import numpy as np

class Animation:
    """Holds and animates frames of a simulation"""

    def __init__(self, mesh, figsize=(10, 10)):
        self.mesh = mesh
        self.fig = None
        self.figsize = figsize
        self.ax = None
        self.solutions = []
        self.boundary_solutions = []
        self.anim = None
        self.max_interior = 0.0
        self.max_boundary = 0.0
        self.max_combined = 0.0

    def plot_frame(self, i):
        """Plots one single frame at frame number i"""
        self.plot.set_array(self.solutions[i])

    def plot_triple_frame(self, i, fix_colourbar=True):
        """Plots one single frame at frame number i"""
        self.interior_plot.set_array(self.solutions[i])
        self.boundary_plot.set_array(self.boundary_solutions[i])
        self.combined_plot.set_array(self.solutions[i] + self.boundary_solutions[i])

        if not fix_colourbar:
            self.interior_plot.set_clim(vmax=np.max(self.solutions[i]))
            self.boundary_plot.set_clim(vmax=np.max(self.boundary_solutions[i]))
            self.combined_plot.set_clim(vmax=np.max(self.solutions[i]+self.boundary_solutions[i]))

    def add_frame(self, solution, boundary_solution):
        """Adds solution to list"""
        self.solutions.append(np.zeros_like(solution))
        self.solutions[-1][:] = solution[:]
        self.boundary_solutions.append(np.zeros_like(boundary_solution))
        self.boundary_solutions[-1][:] = boundary_solution[:]
        self.max_interior = np.max(solution)
        self.max_boundary = np.max(boundary_solution)
        self.max_combined = np.max(solution+boundary_solution)

    def save_frames(self, fname, frames=[], is_triple=True, fix_colourbar=True):
        """Returns matplotlib animation object"""
        total_frames = len(self.solutions)

        if is_triple:
            self.fig, self.axes = plt.subplots(1, 3, figsize=self.figsize)
            plot_function = self.plot_triple_frame
            self.interior_plot, self.boundary_plot, self.combined_plot,\
                self.interior_cax, self.boundary_cax, self.combined_cax =\
                self.mesh.plot_triple(
                    self.solutions[0], self.boundary_solutions[0], self.axes)
            if fix_colourbar:
                self.interior_plot.set_clim(0.0, self.max_interior)
                self.boundary_plot.set_clim(0.0, self.max_boundary)
                self.combined_plot.set_clim(0.0, self.max_combined)
            plt.savefig(fname + '{:04d}'.format(0) + '.png')
        else:
            self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize)
            plot_function = self.plot_frame
            self.plot = self.mesh.plot(self.solutions[0], axis=self.ax, 
                    vmax=self.max_interior, vmin=0.0)
            plt.savefig(fname + '{:04d}'.format(0) + '.png')

        if frames == []:
            frame_list = range(1, total_frames)
        else:
            frame_list = frames

        for i in frame_list:
            plot_function(i)
            plt.savefig(fname + '{:04d}'.format(i) + '.png', fix_colourbar=fix_colourbar)

        # self.anim = FuncAnimation(self.fig, plot_function, frames=total_frames,
                                  # interval=int(1e3/fps), blit=True)

    def show(self):
        """Displays animation as HTML"""
        self.plot_frame(-1)
