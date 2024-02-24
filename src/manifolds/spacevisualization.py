__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

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
from pydantic import BaseModel
from dataclasses import dataclass


@dataclass
class VisualizationParams:
    label: AnyStr
    title: AnyStr
    fig_size: Tuple[float, float]
    projection: AnyStr = None


class SpaceVisualization(object):
    def __init__(self, vParams: VisualizationParams):
        figure = plt.figure(figsize=vParams.fig_size)
        self.ax = figure.add_subplot(111, projection=vParams.projection) if vParams.projection is not None \
            else figure.add_subplot(111)
        self.label = vParams.label
        self.ax.set_title(vParams.title)

    def scatter(self, data_points: np.array) -> NoReturn:
        self.ax.scatter(x=data_points[:, 0], y=data_points[:, 1], label=self.label)
        self.ax.legend()
        plt.show()

    def plot_3d(self, data_points: np.array, space: AnyStr = None) -> NoReturn:
        from geometricspace import GeometricSpace

        if space is not None and space in GeometricSpace.supported_manifolds:
            visualization.plot(data_points, ax=self.ax, space=space, label=self.label, s=80)
        self.ax.plot(
            data_points[:, 0],
            data_points[:, 1],
            data_points[:, 2],
           # linestyle="dashed",
            alpha=0.5)
        # self.ax.legend()
        self.ax.grid()
        self.ax.legend()
        plt.show()


