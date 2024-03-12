__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import geomstats.backend as gs

from typing import AnyStr, List, Callable
from dataclasses import dataclass
import numpy as np
import abc
from abc import ABC



@dataclass
class ManifoldPoint:
    data_point: np.array
    vector: List[float]



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
        "SO3_GROUP",             # Lie 3D rotation group
        "SE3_GROUP",             # 3D rotation and translation Euclidean group
        "SE2_GROUP",             # 2D rotation and translation group
        "S1",                    # Circle in 2D space
        "S2",                    # Hypersphere in 3D Euclidean space
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
        :param axis The index of axis (X=0, Y=1,...) used to compute the mean
        :return the mean value
        """
        return float(gs.sum(samples, axis)/len(samples))

    @staticmethod
    def is_manifold_supported(manifold_type: AnyStr) -> bool:
        """
        :param manifold_type Type of manifold
        :return True if manifold is supported, False otherwise
        """
        return manifold_type in supported_manifolds

    @staticmethod
    def load_csv(filename: AnyStr) -> np.array:
        return np.genfromtxt(filename, delimiter=',')











