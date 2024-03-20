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
        manifold = HypersphereSpace(True)
        samples = manifold.sample(3)
        manifold_points = [
            ManifoldPoint(
                id=f'data{index}',
                location=sample,
                tgt_vector=[0.5, 0.3, 0.5],
                geodesic=False) for index, sample in enumerate(samples)
        ]
        manifold = HypersphereSpace(True)
        exp_map = manifold.tangent_vectors(manifold_points)
        for vec, end_point in exp_map:
            print(f'Tangent vector: {vec} End point: {end_point}')
        manifold.show_manifold(manifold_points)

    @unittest.skip('ignore')
    def test_show_tangent_vector_geodesics(self):
        from geometricspace import GeometricSpace, ManifoldPoint

       # filename = '../../../data/hypersphere_data_1.txt'
       # data = GeometricSpace.load_csv(filename)
        manifold = HypersphereSpace(True)
        samples = manifold.sample(2)
        manifold_points = [
            ManifoldPoint(
                id=f'data{index}',
                location=sample,
                tgt_vector=[0.5, 0.3, 0.5],
                geodesic=True) for index, sample in enumerate(samples)
        ]
        manifold.show_manifold(manifold_points)

    @unittest.skip('ignore')
    def test_euclidean_mean(self):
        from geometricspace import ManifoldPoint
        manifold = HypersphereSpace(True)
        samples = manifold.sample(3)
        manifold_points = [
            ManifoldPoint(
                id=f'data{index}',
                location=sample) for index, sample in enumerate(samples)
        ]
        mean = manifold.euclidean_mean(manifold_points)
        print(mean)

    @unittest.skip('ignore')
    def test_frechet_mean(self):
        from geometricspace import ManifoldPoint

        manifold = HypersphereSpace(True)
        samples = manifold.sample(2)
        assert(manifold.belongs(samples[0]))   # Is True
        vector = [0.8, 0.4, 0.7]

        manifold_points = [
            ManifoldPoint(
                id=f'data{index}',
                location=sample,
                tgt_vector=vector,
                geodesic=False) for index, sample in enumerate(samples)
        ]
        euclidean_mean = manifold.euclidean_mean(manifold_points)
        manifold.belongs(euclidean_mean)   # Is False
        exp_map = manifold.tangent_vectors(manifold_points)
        tgt_vec, end_point = exp_map[0]
        assert(manifold.belongs(end_point))     # Is True
        frechet_mean = manifold.frechet_mean(manifold_points[0], manifold_points[1])
        print(f'Euclidean mean: {euclidean_mean}\nFrechet mean: {frechet_mean}')
        assert (manifold.belongs(frechet_mean))

        frechet_pt = ManifoldPoint(
            id='Frechet mean',
            location=frechet_mean,
            tgt_vector=[0.0, 0.0, 0.0],
            geodesic=False)
        manifold_points.append(frechet_pt)
        manifold.show_manifold(manifold_points, [euclidean_mean])

    def test_extrinsic_to_intrinsic(self):
        from geometricspace import ManifoldPoint

        manifold = HypersphereSpace(True)
        random_samples = manifold.sample(2)
        manifold_pts = [
            ManifoldPoint(f'id{index}', value, None, False, False) for index, value in enumerate(random_samples)
        ]






if __name__ == '__main__':
    unittest.main()