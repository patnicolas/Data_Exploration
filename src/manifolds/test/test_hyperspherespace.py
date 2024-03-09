import unittest
import path
import sys
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from hyperspherespace import HypersphereSpace
from spacevisualization import VisualizationParams, SpaceVisualization
import numpy as np


class TestGeometricSpace(unittest.TestCase):

    def test_sample_hypersphere(self):
        num_samples = 20
        manifold = HypersphereSpace()
        data = manifold.sample(num_samples)
        print(f'Hypersphere:\n{str(data)}')

    def test_hypersphere(self):
        num_samples = 8
        style = {'color': 'red', 'linestyle': '--', 'label': 'Edges'}
        manifold = HypersphereSpace()
        print(str(manifold))
        data = manifold.sample(num_samples)
        visualParams = VisualizationParams("Data on Hypersphere", "locations", (8, 8), style, "3d")
        HypersphereSpace.show(visualParams, data)


if __name__ == '__main__':
    unittest.main()