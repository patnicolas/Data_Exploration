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
        exp_map = manifold.tangent_vectors(manifold_points)
        for vec, end_point in exp_map:
            print(f'Tangent vector: {vec} End point: {end_point}')


    def test_show_tangent_vector(self):
        from geometricspace import GeometricSpace, ManifoldPoint

       # filename = '../../../data/hypersphere_data_1.txt'
       # data = GeometricSpace.load_csv(filename)
        manifold = HypersphereSpace(True)
        samples = manifold.sample(3)
        manifold_points = [
            ManifoldPoint(
                id=f'data{index}',
                location=sample,
                tgt_vector=[0.5, 0.3, 0.5],
                geodesic=True) for index, sample in enumerate(samples)
        ]
        manifold.show_manifold(manifold_points)

    @unittest.skip('ignore')
    def test_mean(self):
        from geometricspace import GeometricSpace, ManifoldPoint

        manifold = HypersphereSpace(True)
        samples = manifold.sample(2)
        print(samples)
        assert(manifold.belongs(samples[0]))
        manifold_points = [
            ManifoldPoint(id='data-1', data_point=samples[0], tgt_vector=[0.8, 0.4, 0.7], geodesic=True)
        ]
        exp_map = manifold.tangent_vectors(manifold_points)
        tgt_vec, end_point = exp_map[0]
        assert(manifold.belongs(end_point))
        x = np.stack((samples[0], end_point), axis=0)
        frechet_mean = manifold.frechet_mean(x)
        print(f'Geodesic: {frechet_mean}')
        assert (manifold.belongs(frechet_mean))

        frechet_pt = ManifoldPoint(
            id='Frechet mean',
            data_point=frechet_mean,
            tgt_vector=[0.0, 0.0, 0.0],
            geodesic=False)
        manifold_points.append(frechet_pt)
        manifold.show_manifold(manifold_points)


if __name__ == '__main__':
    unittest.main()