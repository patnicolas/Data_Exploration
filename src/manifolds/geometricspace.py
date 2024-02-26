__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import geomstats.backend as gs

from typing import AnyStr
import numpy as np
import abc
from abc import ABC

"""
    Abstract class that defined the key components of a Geometric Space. It lists
    the various supported manifolds. 
    
    Class attributes:
    manifold_type: Type of manifold
    supported_manifolds: List of supported manifolds
    
    Object attributes:
    dimension: Dimension of the manifolds embedded in the Euclidean space
    
    Methods:
    sample (pure abstract): Generate random data on a manifold
    mean (static): Compute the mean value for a group of data on a manifold
    is_manifold_supported (static): Test if the manifold is supported
"""


class GeometricSpace(ABC):
    manifold_type: AnyStr
    supported_manifolds = [
        "SO3_GROUP",
        "SE3_GROUP",
        "SE2_GROUP",
        "S1",
        "S2",
        "H2_poincare_disk",
        "H2_poincare_half_plane",
        "H2_klein_disk",
        "poincare_polydisk",
        "S32",
        "M32",
        "S33",
        "M33",
        "SPD2",
    ]

    def __init__(self, dimension: int):
        self.dimension = dimension

    def __str__(self) -> AnyStr:
        return f'{GeometricSpace.manifold_type} with dimension: {self.dimension}'

    @abc.abstractmethod
    def sample(self, num_samples: int) -> np.array:
        """
        Generate random data on this manifold
        :param num_samples Number of sample data points on the manifold
        :return Numpy array of random data points
        """
        pass

    @staticmethod
    def mean(samples: np.array, axis: int = 0) -> float:
        """
        :param samples Sample of data on a manifold
        :param axis The axis the mean has to be computed
        :return the mean value
        """
        return float(gs.sum(samples, axis)/len(samples))

    @staticmethod
    def is_manifold_supported(manifold_type: AnyStr) -> bool:
        return manifold_type in supported_manifolds






