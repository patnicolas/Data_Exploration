__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from typing import NoReturn, List
import numpy as np

from spacevisualization import VisualizationParams, SpaceVisualization
from geometricspace import GeometricSpace, ManifoldPoint, ManifoldDisplay
from dataclasses import dataclass
import geomstats.backend as gs


"""
    Define the Hypersphere geometric space as a 2D manifold in a 3D Euclidean space.
    The key functions are:
    sample: Select uniform data point on the hypersphere
    tangent_vectors: Define a tangent vector from a vector in Euclidean space and a
                     location on the hypersphere
    show: Display the hypersphere and related components in 3D
    
    :param equip Specified that the Hypersphere instance has to be equiped
"""


class HypersphereSpace(GeometricSpace):
    def __init__(self, equip: bool = False):
        dim = 2
        super(HypersphereSpace, self).__init__(dim)
        GeometricSpace.manifold_type = 'Hypersphere'
        self.space = Hypersphere(dim=self.dimension, equip=equip)

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

        def tangent_vector(point: ManifoldPoint) -> (np.array, np.array):
            vector = gs.array(point.vector)
            tangent_v = self.space.to_tangent(vector, base_point=point.data_point)
            end_point = hypersphere_metric.exp(tangent_vec=tangent_v, base_point=point.data_point)
            return tangent_v, end_point

        return [tangent_vector(point) for point in manifold_points]

    def geodesics(self,
                  manifold_points: List[ManifoldPoint],
                  tangent_vectors: List[np.array]) -> List[np.array]:
        """
            Compute the path (x,y,z) values for the geodesic
            :param manifold_points  Set of manifold points as pair (location, vector)
            :param tangent_vectors List of vectors associated with each location on the manifold
            :return List of geodesics as Numpy array of coordinates
        """
        return [self.__geodesic(point, tgt_vec) for point, tgt_vec in zip(manifold_points, tangent_vectors)]

    def show_manifold(self,
                      manifold_points: List[ManifoldPoint],
                      manifold_display: ManifoldDisplay) -> NoReturn:
        """
            Display the various components on a manifold such as data points, tangent vector,
            end point (exp. map), Geodesics
            :param manifold_points  Set of manifold points as pair (location, vector)
            :param manifold_display Type of components to be displayed in 3D
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # If the tangent vectors have to be displayed...
        if manifold_display.value >= ManifoldDisplay.TangentVector.value:
            tangent_vec_end_points = self.tangent_vectors(manifold_points)
            index = 0
            # Walk through all the points on the manifold
            for tgt_vec, end_pt in tangent_vec_end_points:
                # Display the data point on the Hypersphere
                ax = visualization.plot(
                    manifold_points[index].data_point,
                    ax=ax,
                    space="S2",
                    s=100,
                    alpha=0.8,
                    label=f'Start {index}')
                # Show the end point
                ax = visualization.plot(end_pt, ax=ax, space="S2", s=100, alpha=0.8, label=f'End {index}')

                # Display the tangent vector
                arrow = visualization.Arrow3D(manifold_points[index].data_point, vector=tgt_vec)
                arrow.draw(ax, color="red")

                # Compute and display the geodesic associated to each point on the manifold
                if manifold_display == ManifoldDisplay.Geodesics:
                    geodesics = self.__geodesic(manifold_points[index], tgt_vec)
                    geodesics_pts = geodesics(gs.linspace(0.0, 1.0, 40))
                    ax = visualization.plot(geodesics_pts, ax=ax, space="S2", color="blue", label=f'Geodesic {index}')
                index += 1
        ax.legend()
        plt.show()

        # ------------------  Helper method  -------------------
    def __geodesic(self, manifold_point: ManifoldPoint, tangent_vec: np.array) -> np.array:
        hypersphere_metric = HypersphereMetric(self.space)
        return hypersphere_metric.geodesic(
               initial_point=manifold_point.data_point,
               initial_tangent_vec=tangent_vec
           )

    @staticmethod
    def show(vParams: VisualizationParams, data_points: np.array) -> NoReturn:
        """
        Visualize the data points in 3D
        :param vParams Parameters for the visualization
        :param data_points Data points to visualize
        """
        space_visualization = SpaceVisualization(vParams)
        space_visualization.plot_3d(data_points, "S2")

