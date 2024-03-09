__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from typing import NoReturn
import numpy as np

from spacevisualization import VisualizationParams, SpaceVisualization
from geometricspace import GeometricSpace


"""
Define the Hypersphere geometric space as a 2D manifold in a 3D Euclidean space
"""


class HypersphereSpace(GeometricSpace):
    def __init__(self):
        dim = 2
        super(HypersphereSpace, self).__init__(dim)
        GeometricSpace.manifold_type = 'Hypersphere'
        self.space = Hypersphere(dim=self.dimension, equip=False)

    def sample(self, num_samples: int) -> np.array:
        """
        Generate random data on this Hypersphere
        :param num_samples Number of sample data points on the Hypersphere
        :return Numpy array of random data points
        """
        return self.space.random_uniform(num_samples)

    @staticmethod
    def show(vParams: VisualizationParams, data_points: np.array) -> NoReturn:
        """
        Visualize the data points in 3D
        :param vParams Parameters for the visualization
        :param data_points Data points to visualize
        """
        space_visualization = SpaceVisualization(vParams)
        space_visualization.plot_3d(data_points, "S2")

