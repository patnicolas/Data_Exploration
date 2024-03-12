__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from typing import NoReturn, List
import numpy as np

from spacevisualization import VisualizationParams, SpaceVisualization
from geometricspace import GeometricSpace, ManifoldPoint
from dataclasses import dataclass


"""
    Define the Hypersphere geometric space as a 2D manifold in a 3D Euclidean space.
    The key functions are:
    sample: Select uniform data point on the hypersphere
    tangent_vectors: Define a tangent vector from a vector in Euclidean space and a
                     location on the hypersphere
    show: Display the hypersphere and related components in 3D
"""


class HypersphereSpace(GeometricSpace):
    def __init__(self, equip: bool = False):
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

    def tangent_vectors(self, manifold_points: List[ManifoldPoint]) -> List[np.array]:
        """
            Compute the tangent vectors for a set of manifold point as pair
            (location, vector). The tangent vectors are computed by projection to the
            tangent plane.
            :param manifold_points List of pair (location, vector) on the manifold
            :return List of tangent vector for each location
        """
        import geomstats.backend as gs

        hypersphere_metric = HypersphereMetric(self.space)
        def tangent_vector(point: ManifoldPoint):
            vector = gs.array(point.vector)
            tangent_v = self.space.to_tangent(vector, base_point=point.data_point)
            return hypersphere_metric.exp(tangent_vec=tangent_v, base_point=point.data_point)

        return [tangent_vector(point) for point in manifold_points]

    @staticmethod
    def show(vParams: VisualizationParams, data_points: np.array) -> NoReturn:
        """
        Visualize the data points in 3D
        :param vParams Parameters for the visualization
        :param data_points Data points to visualize
        """
        space_visualization = SpaceVisualization(vParams)
        space_visualization.plot_3d(data_points, "S2")

