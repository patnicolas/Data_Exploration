__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import matplotlib
import matplotlib.colors as clrs
import matplotlib.image as mpg
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import geomstats.backend as gs

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Tuple, NoReturn, AnyStr, List
import numpy as np
import geomstats.visualization as visualization
from pydantic import BaseModel
from dataclasses import dataclass
from manifolds.geometricspace import ManifoldPoint
from manifolds.geometricexception import GeometricException


@dataclass
class VisualizationParams:
    label: AnyStr
    title: AnyStr
    fig_size: Tuple[float, float]
    kwargs: dict[AnyStr, AnyStr] = None
    projection: AnyStr = None


class SpaceVisualization(object):
    def __init__(self, vParams: VisualizationParams):
        figure = plt.figure(figsize=vParams.fig_size)
        self.style = vParams.kwargs

        self.ax = figure.add_subplot(111, projection=vParams.projection) if vParams.projection is not None \
            else figure.add_subplot(111)
        self.label = vParams.label
        self.ax.set_title(vParams.title)
        self.style = vParams.kwargs

    def scatter(self, data_points: np.array) -> NoReturn:
        self.ax.scatter(x=data_points[:, 0], y=data_points[:, 1], label=self.label)
        self.ax.legend()
        plt.show()

    def plot_3d(self, data_points: np.array, space: AnyStr = None) -> NoReturn:
        from geometricspace import GeometricSpace

        if space is not None:
            if space == 'S2':
                visualization.plot(data_points, ax=self.ax, space=space, label=self.label, s=80)
            elif space == 'S32' or space == 'M32' or space == 'S33':
                visualization.plot(data_points, ax=self.ax, space=space, label=self.label, s=5)
            else:
                raise GeometricException(f'Space {space} is not supported')

        if self.style is not None:
            self.ax.plot(
                data_points[:, 0],
                data_points[:, 1],
                data_points[:, 2],
                **self.style,
                alpha=0.5)
        self.ax.grid()
        self.ax.legend()
        plt.show()

    @staticmethod
    def plot_tangent_geodesic(manifold_points: List[ManifoldPoint], space: AnyStr) -> NoReturn:
        if space == 'S2':
            from hyperspherespace import HypersphereSpace

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")

            start_point = manifold_points[0]
            manifold = HypersphereSpace(True)
            tangent_vec_end_points = manifold.tangent_vectors([start_point])
            tgt_vec, end_pt = tangent_vec_end_points[0]
            ax = visualization.plot(start_point.data_point, ax=ax, space="S2", s=100 , alpha=0.8, label="Start")
            ax = visualization.plot(end_pt, ax=ax, space="S2", s=100, alpha=0.8, label="End point")
            arrow = visualization.Arrow3D(start_point, vector=tgt_vec)
            arrow.draw(ax, color="red")
            ax.legend()
            plt.show()
        else:
            raise GeometricException(f'Space {space} is not supported')




