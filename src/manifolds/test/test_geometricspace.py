import unittest
import path
import sys
import os

directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

import numpy as np
from geometricspace import GeometricSpace
from manifoldpoint import ManifoldPoint
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

    def show_manifold(self,manifold_points: List[ManifoldPoint]) -> NoReturn:
        return None

    def frechet_mean(self, points: np.array) -> np.array:
        return None

    def belongs(self, point: List[float]) -> bool:
        return False


class TestGeometricSpace(unittest.TestCase):
    def test_euclidean_mean(self):
        test_class = TestClass(3)
        values = [[4.5, 1.4], [0.5, 3.5]]
        manifold_pts = [ ManifoldPoint(f'id{index}', np.array(value)) for index, value in enumerate(values)]
        print(f'Mean value: {TestClass.euclidean_mean(manifold_pts)}')
        #Extrinsic coordinates: [0.24628623 - 0.92608845  0.28583787]
        # Intrinsic coordinates: [1.28091567 4.97231511]

    def test_sample(self):
        test_class = TestClass(3)
        print(test_class.sample(20))

    def test_load_data(self):
        filename = '../../../data/hypersphere_data_1.txt'
        data = GeometricSpace.load_csv(filename)
        print(f'Loaded Array:\n{data}')

    def test_to_intrinsic(self):
        from hyperspherespace import HypersphereSpace

        manifold = HypersphereSpace(True)
        random_samples = manifold.sample(2)
        manifold_point = ManifoldPoint('id1', random_samples[0])
        print(f'Extrinsic coordinates: {random_samples[0]}\nIntrinsic coordinates: {manifold_point.to_intrinsic(manifold.space)}')


if __name__ == '__main__':
    unittest.main()
