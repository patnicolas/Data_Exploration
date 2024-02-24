__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from geomstats.geometry.euclidean import Euclidean
from geometricspace import GeometricSpace
from typing import NoReturn
import numpy as np
from spacevisualization import VisualizationParams, SpaceVisualization


class EuclideanSpace(GeometricSpace):

    def __init__(self, dimension: int):
        super(EuclideanSpace, self).__init__(dimension)
        self.space = Euclidean(dim=self.dimension, equip=False)
        GeometricSpace.manifold_type = 'Euclidean'

    def sample(self, num_samples: int) -> np.array:
        return self.space.random_point(num_samples)

    @staticmethod
    def show(vParams: VisualizationParams, data_points: np.array) -> NoReturn:
        space_visualization = SpaceVisualization(vParams)
        space_visualization.plot_3d(data_points)
