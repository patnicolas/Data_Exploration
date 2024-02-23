
import matplotlib
import matplotlib.colors as clrs
import matplotlib.image as mpg
import matplotlib.patches as mptch
import matplotlib.pyplot as plt
import geomstats.backend as gs

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Tuple, NoReturn, AnyStr
import numpy as np
import geomstats.visualization as visualization


class SpaceVisualization(object):
    def __init__(self, figure_size: Tuple[float, float], label: AnyStr, title: AnyStr):
        figure = plt.figure(figsize=figure_size)
        # fig = plt.figure(figsize=(5, 5))
        self.ax = figure.add_subplot(111, projection="3d")
        self.label = label
        # self.ax.set_title(title)


class Space2DVisualization(SpaceVisualization):
    def __init__(self, figure_size: Tuple[float, float], label: AnyStr, title: AnyStr):
        super(Space2DVisualization, self).__init__(figure_size, label, title)

    def scatter(self, data_points: np.array) -> NoReturn:
        self.ax.scatter(x=data_points[:, 0], y=data_points[:, 1], label=self.label)
        # self.ax.plot(x=data_points[:, 0],y=data_points[:, 1], label=f'{self.label}_s')
        self.ax.legend()
        plt.show()

    def plot_3d(self, data_points: np.array, is_manifold: bool) -> NoReturn:
        if is_manifold:
            visualization.plot(data_points, ax=self.ax, space="S2", label=self.label, s=80)
        self.ax.plot(
            data_points[:, 0],
            data_points[:, 1],
            data_points[:, 2],
           # linestyle="dashed",
            alpha=0.5)
        # self.ax.legend()
        self.ax.grid()
        plt.show()


