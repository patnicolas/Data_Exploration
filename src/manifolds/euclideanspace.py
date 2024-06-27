__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from geomstats.geometry.euclidean import Euclidean
from geometricspace import GeometricSpace
from typing import NoReturn
import numpy as np
from spacevisualization import VisualizationParams, SpaceVisualization

"""
    Define the Euclidean space and its components

    Class attributes:
    manifold_type: Type of manifold
    supported_manifolds: List of supported manifolds

    Object attributes:
    dimension: Dimension of this Euclidean space

    Methods:
    sample (pure abstract): Generate random data on a manifold
    show (abstract method): Display the Euclidean space in 3 dimension
"""

class EuclideanSpace(GeometricSpace):
    def __init__(self, dimension: int) -> None:
        super(EuclideanSpace, self).__init__(dimension)
        self.space = Euclidean(dim=self.dimension, equip=False)
        GeometricSpace.manifold_type = 'Euclidean'

    def sample(self, num_samples: int) -> np.array:
        """
        Generate random data on the Euclidean space
        :param num_samples Number of sample data points on the Euclidean space
        :return Numpy array of random data points
        """
        return self.space.random_point(num_samples)

    @staticmethod
    def show(vParams: VisualizationParams, data_points: np.array) -> NoReturn:
        """
        Visualize the data points in 3D
        :param vParams Parameters for the visualization
        :param data_points Data points to visualize
        """
        space_visualization = SpaceVisualization(vParams)
        space_visualization.plot_3d(data_points)
