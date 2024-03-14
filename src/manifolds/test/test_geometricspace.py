import unittest
import path
import sys
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

import numpy as np
from geometricspace import GeometricSpace, ManifoldPoint, ManifoldDisplay
from typing import NoReturn, List


class TestClass(GeometricSpace):
    def __init__(self, dimension: int):
        super(TestClass, self).__init__(dimension)

    def sample(self, num_samples: int) -> np.array:
        return np.random.random(num_samples)

    def tangent_vectors(self, manifold_points: List[ManifoldPoint]) -> List[np.array]:
        return None

    def geodesics(self,
                  manifold_points: List[ManifoldPoint],
                  tangent_vectors: List[np.array]) -> List[np.array]:
        return None

    def show_manifold(self,
                      manifold_points: List[ManifoldPoint],
                      manifold_display: ManifoldDisplay) -> NoReturn:
        return None
    def frechet_mean(self, points: np.array) -> np.array:
        return None

    def belongs(self, point: List[float]) -> bool:
        return False


class TestGeometricSpace(unittest.TestCase):
    def test_euclidean_mean(self):
        test_class = TestClass(3)
        values = [[4.5, 1.4], [0.5, 3.5]]
        print(f'Mean value: {TestClass.euclidean_mean(np.array(values))}')

    def test_sample(self):
        test_class = TestClass(3)
        print(test_class.sample(20))

    def test_load_data(self):
        filename = '../../../data/hypersphere_data_1.txt'
        data = GeometricSpace.load_csv(filename)
        print(f'Loaded Array:\n{data}')


if __name__ == '__main__':
    unittest.main()