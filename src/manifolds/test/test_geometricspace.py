import unittest
import path
import sys
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

import numpy as np
from geometricspace import GeometricSpace


class TestClass(GeometricSpace):
    def __init__(self, dimension: int):
        super(TestClass, self).__init__(dimension)

    def sample(self, num_samples: int) -> np.array:
        return np.random.random(num_samples)


class TestGeometricSpace(unittest.TestCase):
    def test_mean_geometric_space(self):
        test_class = TestClass(3)
        values: list[float] = [4.5, 1.4, 0.5, 3.5]
        print(f'Mean value: {test_class.mean(np.array(values))}')

    def test_sample(self):
        test_class = TestClass(3)
        print(test_class.sample(20))


if __name__ == '__main__':
    unittest.main()