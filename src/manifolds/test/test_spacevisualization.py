import unittest
import path
import sys
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

import numpy as np
from spacevisualization import Space2DVisualization
import matplotlib.pyplot as plt


class TestSpaceVisualization(unittest.TestCase):

    @unittest.skip
    def test_scatter(self):
        data_points = np.array([[-0.19963953, -0.90072907],
                     [-0.0292344,   0.77812525]])
        print(data_points[:, 0])
        print(data_points[:, 1])
        fig_size = (4, 4)
        label = 'Values'
        title = 'This is a test'
        space_visualization = Space2DVisualization(fig_size, label, title)
        space_visualization.scatter(data_points)

    @unittest.skip
    def test_plot(self):
        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)
        data_points = np.array([[0.12201818, - 0.80014098, 0.44830868],
                       [-0.76866486, 0.17708725, -0.28586653],
                       [0.24561599, -0.97369755, 0.63140498],
                       [-0.12494054, -0.31722046, 0.49782918],
                       [0.68541031, 0.6051049, 0.66552322],
                       [-0.24688761, 0.64412856, -0.68645193]])

        fig, ax = plt.subplots()
        print(data_points[:,0])
        ax.plot(data_points[:,0], data_points[:,1])

        ax.set(xlabel='time (s)', ylabel='voltage (mV)',
               title='About as simple as it gets, folks')
        ax.grid()

        fig.savefig("test.png")
        plt.show()


    @unittest.skip
    def test_plot_3d(self):
        data_points = np.array([[0.12201818, - 0.80014098,  0.44830868],
                        [-0.76866486, 0.17708725, -0.28586653],
                        [0.24561599, -0.97369755, 0.63140498],
                        [-0.12494054, -0.31722046,  0.49782918],
                        [0.68541031, 0.6051049, 0.66552322],
                        [-0.24688761,  0.64412856, -0.68645193]])
        fig_size = (4, 4)
        label = 'Values'
        title = 'This is a test'
        space_visualization = Space2DVisualization(fig_size, label, title)
        space_visualization.plot_3d(data_points, True)

    def test_3d_sphere(self):
        from geometricspace import HypersphereSpace
        dim = 2
        num_samples = 10
        manifold = HypersphereSpace(dim)
        data_points = manifold.sample(num_samples)
        print(f'Hypersphere:\n{str(data_points)}')

        fig_size = (8, 8)
        label = 'Values'
        title = 'This is a test'
        space_visualization = Space2DVisualization(fig_size, label, title)
        space_visualization.plot_3d(data_points, "S2")


if __name__ == '__main__':
    unittest.main()