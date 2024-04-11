__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.base import LevelSet
import numpy as np
from manifoldpoint import ManifoldPoint


class RiemannianConnection(object):
    def __init__(self, riemannian_metric: RiemannianMetric):
        self.riemannian_metric = riemannian_metric.geodesic()

    def inner_product(self, tgt_vec1: np.array, tgt_vec_2: np.array) -> np.array:
        self.riemannian_metric.inner_production(tgt_vec1, tgt_vec_2)

    def manifold_point_inner_product(self, manifold_base_pt: ManifoldPoint, manifold_pt: ManifoldPoint) -> np.array:
        return self.riemannian_metric.inner_production(
            manifold_base_pt.tgt_vector,
            manifold_pt.tgt_vector,
            manifold_base_pt.location)

    def norm(self, vector: np.array) -> np.array:
        return self.riemannian_metric.norm(vector)

    def manifold_point_norm(self, manifold_base_pt: ManifoldPoint) -> np.array:
        return self.riemannian_metric.norm(manifold_base_pt.tgt_vector, manifold_base_pt.location)

    def geodesic(self, manifold_base_pt: ManifoldPoint, end_point: ManifoldPoint = None) -> np.array:
        return NotImplementedError("method not implemented")
