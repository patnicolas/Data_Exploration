__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from manifolds.hyperspherespace import HypersphereSpace
from sklearn.decomposition import PCA
from geomstats.learning.pca import TangentPCA
from geomstats.learning.frechet_mean import FrechetMean
import numpy as np
from typing import AnyStr
import geomstats.backend as gs


class PCAHypersphere(object):

    def __init__(self):
        self.hypersphere_space = HypersphereSpace(equip=True)

    def sample(self, num_samples: int) -> np.array:
        return self.hypersphere_space.sample(num_samples)

    @staticmethod
    def euclidean_pca_components(data: np.array) -> np.array:
        num_components = 3
        pca = PCA(num_components)
        pca.fit(data)
        return pca.components_

    @staticmethod
    def euclidean_pca_transform(data: np.array) -> np.array:
        num_components = 3
        pca = PCA(num_components)
        return pca.fit_transform(data)

    def tangent_pca(self, data: np.array) -> TangentPCA:
        sphere = self.hypersphere_space.space
        tgt_pca = TangentPCA(sphere)
        mean = FrechetMean(sphere, method="default")
        mean.fit(data)
        estimate = mean.estimate_
        tgt_pca.fit(data, base_point=estimate)
        return tgt_pca

    def tangent_pca_components(self, data: np.array) -> np.array:
        tgt_pca = self.tangent_pca(data)
        return tgt_pca.components_


    def tangent_pca_transform(self, data: np.array) -> np.array:
        tgt_pca = self.tangent_pca(data)
        tgt_pca.transform(data)