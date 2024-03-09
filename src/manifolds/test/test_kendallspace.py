import unittest
import path
import sys
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from kendallspace import KendallSpace
from spacevisualization import VisualizationParams, SpaceVisualization
import numpy as np



class TestKendallSpace(unittest.TestCase):

    def test_sample_kendall_sphere(self):
        num_samples = 20
        manifold = KendallSpace()
        data = manifold.sample(num_samples)
        print(f'Kendall:\n{str(data)}')

    def test_kendall_sphere(self):
        num_samples = 2
        style = {'color': 'red', 'linestyle': '--', 'label': 'Edges'}
        manifold = KendallSpace()
        print(str(manifold))
        data = manifold.sample(num_samples)
        visualParams = VisualizationParams("Data on Kendall", "Data on Kendall S32", (8, 8))
        KendallSpace.show(visualParams, data, 'S32')

    """
    def test_kendall_sphere_m32(self):
        num_samples = 2
        style = {'color': 'red', 'linestyle': '--', 'label': 'Edges'}
        manifold = KendallSpace()
        print(str(manifold))
        data = manifold.sample(num_samples)
        visualParams = VisualizationParams("Data on Kendall M32 group", "Values", (8, 8), style)
        KendallSpace.show(visualParams, data, 'M32')
    """


if __name__ == '__main__':
    unittest.main()