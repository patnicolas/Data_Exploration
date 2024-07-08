__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from manifolds.hyperspherespace import HypersphereSpace
import geomstats.backend as gs
from typing import NoReturn, List
from manifolds.manifoldpoint import ManifoldPoint

"""
Define a generic Geometric Distribution on an Hypersphere using the Geomstats Python library
The purpose of this class is to display data points and associated tangent vectors on an
Hypersphere as defined in the class HypersphereSpace.
"""


class GeometricDistribution(object):
    _ZERO_TGT_VECTOR = [0.0, 0.0, 0.0]

    @staticmethod
    def zero_tgt_vector() -> List[float]:
        return GeometricDistribution._ZERO_TGT_VECTOR

    def __init__(self) -> None:
        """
        Constructor for the generic geometric distribution on a hypersphere
        """
        self.manifold = HypersphereSpace(True)

    def show_points(self, num_pts: int, tgt_vector: List[float] = _ZERO_TGT_VECTOR) -> int:
        """
        Display the data points on a manifold (Hypersphere). The tangent vector is displayed if
        is not defined as the extrinsic origin zero_tgt_vector = [0.0, 0.0, 0.0]
        @param num_pts: Number of points to be displayed on Hypersphere
        @type num_pts: int
        @param tgt_vector: Tangent vector extrinsic coordinate
        @type tgt_vector: List of float
        @return: Number of points from exponential map
        @rtype: int
        """
        manifold_pts = self._random_manifold_points(num_pts, tgt_vector)
        exp_map = self.manifold.tangent_vectors(manifold_pts)
        for v, end_pt in exp_map:
            print(f'Tangent vector: {v} End point: {end_pt}')
        self.manifold.show_manifold(manifold_pts)
        return len(exp_map)

    """ --------------------  Protected Helper Method ---------------------------  """
    def _random_manifold_points(self, num_pts: int, tgt_vector: List[float]) -> List[ManifoldPoint]:
        p = self.manifold.sample(num_pts)
        return [
            ManifoldPoint(
                id=f'data{index}',
                location=sample,
                tgt_vector=tgt_vector,
                geodesic=False) for index, sample in enumerate(p)
        ]





"""
normal_dist = NormalDistribution()
num_points = 2
num_pdfs = 40
normal_dist.show_points(num_points, [0.4, 0.7, 0.2])
normal_dist.show_distribution(num_pdfs, num_points)
"""

beta_dist = BetaHypersphere()
num_points = 10
num_pdfs = 200
beta_dist.show_points(num_points)
beta_dist.show_distribution(num_pdfs, num_points)