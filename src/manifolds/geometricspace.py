__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import geomstats.backend as gs

from typing import AnyStr
import numpy as np
import abc
from abc import ABC

gs.random.seed(42)


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
        pass

    @staticmethod
    def mean(samples: np.array, axis: int = 0) -> np.array:
        return gs.sum(samples, axis)/len(samples)

    @staticmethod
    def is_manifold_valid(manifold_type: AnyStr) -> bool:
        return manifold_type in supported_manifolds






