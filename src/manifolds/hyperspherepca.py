__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from src.manifolds.hyperspherespace import HypersphereSpace
from sklearn.decomposition import PCA
from geomstats.learning.pca import TangentPCA
from geomstats.learning.frechet_mean import FrechetMean
import numpy as np
import geomstats.backend as gs


class HyperspherePCA(object):

    def __init__(self) -> None:
        """
        Constructor for the PCA on Hypersphere tangent space
        """
        self.hypersphere_space = HypersphereSpace(equip=True)

    def sample(self, num_samples: int) -> np.array:
        """
        Generate a sample data point on this hypersphere
        @param num_samples: Number of samples
        @type num_samples: int
        @return: Numpy array of manifold points
        @rtype: Numpy array
        """
        assert(0 < num_samples < 1_200_000, f'Number of samples {num_samples} it out of range')
        return self.hypersphere_space.sample(num_samples)

    @staticmethod
    def euclidean_pca_components(data: np.array) -> (np.array, np.array):
        """
        Extract the principal components in the Euclidean space
        @param data: Data on the Hypersphere
        @type data: Numpy Array
        @return: Pair {singular values, eigenvectors}
        @rtype: Pair Numpy arrays
        """
        num_components = 3
        pca = PCA(num_components)
        pca.fit(data)
        return pca.singular_values_, pca.components_

    @staticmethod
    def euclidean_pca_transform(data: np.array) -> np.array:
        """
        Extract the principal components in Euclidean space and apply to transform
        data points on the hypersphere
        @param data: Data point on the manifold
        @type data: Numpy Array
        @return: Data point transformed
        @rtype: Numpy array
        """
        num_components = 3
        pca = PCA(num_components)
        return pca.fit_transform(data)

    def tangent_pca_components(self, data: np.array) -> np.array:
        """
        Compute the principal components on plane tangent to the hypersphere
        @param data: Data on the hypersphere
        @type data: Numpy array
        @return: Principal components on the tangent space
        @rtype: Numpy arrau
        """
        tgt_pca = self.__tangent_pca(data)
        return tgt_pca.components_

    def tangent_pca_transform(self, data: np.array) -> np.array:
        """
        Compute the principal components on the plan tangent to the hypersphere and
        apply to transform data on the hypersphere
        @param data: Data on the hypersphere
        @type data: Numpy array
        @return: Transformed data points
        @rtype: Numpy array
        """
        tgt_pca = self.__tangent_pca(data)
        tgt_pca.transform(data)

    ''' -------------------  Private Helper Method ----------------------  '''
    def __tangent_pca(self, data: np.array) -> TangentPCA:
        sphere = self.hypersphere_space.space
        tgt_pca = TangentPCA(sphere)
        mean = FrechetMean(sphere, method="default")
        mean.fit(data)
        estimate = mean.estimate_
        tgt_pca.fit(data, base_point=estimate)
        return tgt_pca
