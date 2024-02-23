import unittest
import path
import sys
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from geometricspace import EuclideanSpace, HypersphereSpace, GeometricSpace


class TestGeometricSpace(unittest.TestCase):

    def test_init_2_euclidean(self):
        dim = 2
        num_samples = 5
        euclidean_space = EuclideanSpace(dim)
        print(str(euclidean_space))
        data = euclidean_space.sample(num_samples)
        print(f'Euclidean:\n{str(data)}')

    def test_init_3_euclidean(self):
        dim = 3
        num_samples = 100
        euclidean_space = EuclideanSpace(dim)
        data = euclidean_space.sample(num_samples)
        print(f'Euclidean:\n{str(data)}')

    def test_init_euclidean_average(self):
        dim = 2
        num_samples = 100
        euclidean_space = EuclideanSpace(dim)
        data = euclidean_space.sample(num_samples)
        average_points = GeometricSpace.average(data)
        print(f'Euclidean average:\n{str(average_points)}')

    def test_init_hypersphere(self):
        dim = 2
        num_samples = 20
        manifold = HypersphereSpace(dim)
        print(str(manifold))
        data = manifold.sample(num_samples)
        print(f'Hypersphere:\n{str(data)}')


if __name__ == '__main__':
    unittest.main()