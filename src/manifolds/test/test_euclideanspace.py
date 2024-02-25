import unittest
import path
import sys
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from euclideanspace import EuclideanSpace
from geometricspace import GeometricSpace
from spacevisualization import VisualizationParams


class TestEuclideanSpace(unittest.TestCase):

    def test_sample_2_euclidean(self):
        dim = 2
        num_samples = 5
        euclidean_space = EuclideanSpace(dim)
        print(str(euclidean_space))
        data = euclidean_space.sample(num_samples)
        print(f'Euclidean:\n{str(data)}')

    def test_sample_3_euclidean(self):
        dim = 3
        num_samples = 100
        euclidean_space = EuclideanSpace(dim)
        data = euclidean_space.sample(num_samples)
        print(f'Euclidean:\n{str(data)}')

    def test_euclidean_average(self):
        dim = 2
        num_samples = 100
        euclidean_space = EuclideanSpace(dim)
        data = euclidean_space.sample(num_samples)
        average_points = GeometricSpace.mean(data)
        print(f'Euclidean average:\n{str(average_points)}')


if __name__ == '__main__':
    unittest.main()