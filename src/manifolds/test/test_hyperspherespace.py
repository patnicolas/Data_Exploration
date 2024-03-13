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

    @unittest.skip('ignore')
    def test_sample_hypersphere(self):
        num_samples = 180
        manifold = HypersphereSpace()
        data = manifold.sample(num_samples)
        print(f'Hypersphere:\n{str(data)}')

    @unittest.skip('ignore')
    def test_hypersphere(self):
        num_samples = 8
        style = {'color': 'red', 'linestyle': '--', 'label': 'Edges'}
        manifold = HypersphereSpace()
        print(str(manifold))
        data = manifold.sample(num_samples)
        visualParams = VisualizationParams("Data on Hypersphere", "locations", (8, 8), style, "3d")
        HypersphereSpace.show(visualParams, data)

    @unittest.skip('ignore')
    def test_tangent_vector(self):
        from geometricspace import GeometricSpace, ManifoldPoint

        filename = '../../../data/hypersphere_data_1.txt'
        data = GeometricSpace.load_csv(filename)
        manifold_points = [
            ManifoldPoint(data[1], [1.0, 0.4, 1.3]),
            ManifoldPoint(data[2], [1.0, 0.4, 1.3]),
            ManifoldPoint(data[3], [1.0, 0.4, 1.3])
        ]
        manifold = HypersphereSpace(True)
        tangent_vec = manifold.tangent_vectors(manifold_points)
        for vec, end_point in tangent_vec:
            print(f'Tangent vector: {vec} End point: {end_point}')

    def test_show_tangent_vector(self):
        from geometricspace import GeometricSpace, ManifoldPoint, ManifoldDisplay

       # filename = '../../../data/hypersphere_data_1.txt'
       # data = GeometricSpace.load_csv(filename)
        manifold = HypersphereSpace(True)
        samples = manifold.sample(4)
        manifold_points = [
            ManifoldPoint(samples[0], [1.0, 0.4, 1.3]),
            ManifoldPoint(samples[1], [1.0, 0.4, 1.3])
        ]
        manifold.show_manifolds(manifold_points, ManifoldDisplay.Geodesics)


if __name__ == '__main__':
    unittest.main()