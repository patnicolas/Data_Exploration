
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.euclidean import Euclidean
import geomstats.visualization as visualization
from pydantic import BaseModel
from typing import AnyStr
import numpy as np


class GeometricSpace(object):
    manifold_type: AnyStr

    def __init__(self, dimension: int):
        self.dimension = dimension

    def __str__(self) -> AnyStr:
        return f'{GeometricSpace.manifold_type} with dimension: {self.dimension}'

    @staticmethod
    def average(samples: np.array, axis: int = 0) -> np.array:
        return gs.sum(samples, axis)/len(samples)


class EuclideanSpace(GeometricSpace):

    def __init__(self, dimension: int):
        super(EuclideanSpace, self).__init__(dimension)
        self.space = Euclidean(self.dimension, equip=False)
        GeometricSpace.manifold_type = 'Euclidean'

    def sample(self, num_samples: int) -> np.array:
        return self.space.random_point(num_samples)



class HypersphereSpace(GeometricSpace):
    def __init__(self, dimension: int):
        """
        :param dimension Dimension for the hypersphere
        """
        super(HypersphereSpace, self).__init__(dimension)
        GeometricSpace.manifold_type = 'Hypersphere'
        self.space = Hypersphere(self.dimension, equip=False)

    def sample(self, num_samples: int) -> np.array:
        return self.space.random_uniform(num_samples)
