__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from typing import NoReturn
import numpy as np

from spacevisualization import VisualizationParams, SpaceVisualization
from geometricspace import GeometricSpace


class HypersphereSpace(GeometricSpace):
    def __init__(self, dimension: int):
        """
        :param dimension Dimension for the hypersphere
        """
        super(HypersphereSpace, self).__init__(dimension)
        GeometricSpace.manifold_type = 'Hypersphere'
        self.space = Hypersphere(dim=self.dimension, equip=False)

    def sample(self, num_samples: int) -> np.array:
        return self.space.random_uniform(num_samples)

    @staticmethod
    def show(vParams: VisualizationParams, data_points: np.array) -> NoReturn:
        space_visualization = SpaceVisualization(vParams)
        space_visualization.plot_3d(data_points, "S2")

