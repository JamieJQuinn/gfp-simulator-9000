import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
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
        self.vmax = None
        self.vmin = None

    def plot_frame(self, i):
        """Plots one single frame at frame number i"""
        plot = self.mesh.plot(self.solutions[i], self.boundary_solutions[i],
                              self.ax, vmax=self.vmax, vmin=self.vmin)
        return plot,

    def add_frame(self, solution, boundary_solution):
        """Adds solution to list"""
        self.solutions.append(np.zeros_like(solution))
        self.solutions[-1][:] = solution[:]
        self.boundary_solutions.append(np.zeros_like(boundary_solution))
        self.boundary_solutions[-1][:] = boundary_solution[:]
        # self.vmax = np.max(solution)
        self.vmax = 1.0
        self.vmin = np.min(solution)

    def animate(self, fps=25):
        """Returns matplotlib animation object"""
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize)
        total_frames = len(self.solutions)
        self.anim = FuncAnimation(self.fig, self.plot_frame, frames=total_frames,
                                  interval=int(1e3/fps), blit=True)

    def show(self):
        """Displays animation as HTML"""
        self.plot_frame(-1)
