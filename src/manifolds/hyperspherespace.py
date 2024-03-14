__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from typing import NoReturn, List
import numpy as np

from spacevisualization import VisualizationParams, SpaceVisualization
from geometricspace import GeometricSpace, ManifoldPoint
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
        self.hypersphere_metric = HypersphereMetric(self.space)

    def belongs(self, point: List[float]) -> bool:
        assert len(point) == 3, f'Point {point} should have 3 dimension'
        """
        Test if a point belongs to this hypersphere
        :param point defined as a list of 3 values
        :return True if the point belongs to the manifold, False otherwise
        """
        return self.space.belongs(point)

    def frechet_mean(self, manifold_points: List[ManifoldPoint]) -> np.array:
        """
        Compute the mean of multiple points on a manifold
        :param manifold_points Data points on a manifold with optional tangent vector and geodesic
        :return mean value as a Numpy array
        """
        from geomstats.learning.frechet_mean import FrechetMean

        frechet_mean = FrechetMean(self.space)
        frechet_mean.fit([manifold_pt.location for manifold_pt in manifold_points)
        return frechet_mean.estimate_


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
        return [self.__tangent_vector(point) for point in manifold_points]

    def geodesics(self,
                  manifold_points: List[ManifoldPoint],
                  tangent_vectors: List[np.array]) -> List[np.array]:
        """
        Compute the path (x,y,z) values for the geodesic
        :param manifold_points  Set of manifold points as pair (location, vector)
        :param tangent_vectors List of vectors associated with each location on the manifold
        :return List of geodesics as Numpy array of coordinates
        """
        return [self.__geodesic(point, tgt_vec)
                for point, tgt_vec in zip(manifold_points, tangent_vectors) if point.geodesic]

    def show_manifold(self, manifold_points: List[ManifoldPoint]) -> NoReturn:
        """
        Display the various components on a manifold such as data points, tangent vector,
        end point (exp. map), Geodesics
        :param manifold_points  Set of manifold points as pair (location, vector)
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Walk through the list of data point on the manifold
        for manifold_pt in manifold_points:
            ax = visualization.plot(
                manifold_pt.location,
                ax=ax,
                space="S2",
                s=100,
                alpha=0.8,
                label=manifold_pt.id)

            # If the tangent vector has to be extracted and computed
            if manifold_pt.tgt_vector is not None:
                tgt_vec, end_pt = self.__tangent_vector(manifold_pt)
                # Show the end point
                ax = visualization.plot(end_pt, ax=ax, space="S2", s=100, alpha=0.8, label=f'End {manifold_pt.id}')
                arrow = visualization.Arrow3D(manifold_pt.location, vector=tgt_vec)
                arrow.draw(ax, color="red")

                # If the geodesic is to be computed and displayed
                if manifold_pt.geodesic:
                    geodesics = self.__geodesic(manifold_pt, tgt_vec)

                    # Arbitrary plot 40 data point for the geodesic from the tangent vector
                    geodesics_pts = geodesics(gs.linspace(0.0, 1.0, 40))
                    ax = visualization.plot(
                        geodesics_pts,
                        ax=ax,
                        space="S2",
                        color="blue",
                        label=f'Geodesic {manifold_pt.id}')

        ax.legend()
        plt.show()

        # ------------------  Helper methods  -------------------
    def __geodesic(self, manifold_point: ManifoldPoint, tangent_vec: np.array) -> np.array:
        return self.hypersphere_metric.geodesic(
               initial_point=manifold_point.location,
               initial_tangent_vec=tangent_vec
           )

    def __tangent_vector(self, point: ManifoldPoint) -> (np.array, np.array):
        import geomstats.backend as gs

        vector = gs.array(point.tgt_vector)
        tangent_v = self.space.to_tangent(vector, base_point=point.location)
        end_point = self.hypersphere_metric.exp(tangent_vec=tangent_v, base_point=point.location)
        return tangent_v, end_point

    @staticmethod
    def show(vParams: VisualizationParams, locations: np.array) -> NoReturn:
        """
        Visualize the data points in 3D
        :param vParams Parameters for the visualization
        :param locations Data points to visualize
        """
        space_visualization = SpaceVisualization(vParams)
        space_visualization.plot_3d(locations, "S2")

