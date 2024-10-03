
import unittest
import numpy as np
from filter.kalman import KalmanFilter

class KalmanFilterTest(unittest.TestCase):

    @unittest.skip('Ignored')
    def test_init_2d(self):
        x0 = np.array([0.0, 0.1])
        P0 = np.eye(x0.shape[0])
        A = np.array([[1.0, 0.5], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        cov_means = (0.6, 1.0)
        kf = KalmanFilter.build(x0, P0, A, H, cov_means)
        print(str(kf))
        self.assertTrue(kf.A.shape[0] == 2)

    @unittest.skip('Ignored')
    def test_init_3d(self):
        x0 = np.array([0.0, 0.1, 0.1])
        P0 = np.eye(x0.shape[0])
        A = np.array([[1.0, 0.5, 1.0], [0.0, 1.0, 0.5], [1.0, 0.0, 1.0]])
        H = np.array([[1.0, 0.0, 0.0]])
        cov_means = (0.6, 1.0)
        kf = KalmanFilter.build(x0, P0, A, H, cov_means)
        print(str(kf))
        self.assertTrue(kf.A.shape[0] == 3)

    @unittest.skip('Ignored')
    def test_simulate_2d(self):
        x0 = np.array([0.0, 0.1])
        P0 = np.eye(x0.shape[0])
        A = np.array([[1.0, 0.5], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        cov_means = (0.6, 1.0)
        num_points = 200
        rNoise = 3.4
        kf = KalmanFilter.build(x0, P0, A, H, cov_means)
        estimation = kf.simulate(num_points, lambda i: np.array([i + np.random.normal(rNoise, rNoise)]),
                                 np.array([0.4, 0.6]))

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(121)
        x: np.array = np.linspace(0, num_points, num_points)
        _x = x.tolist()
        ax1.plot(_x, [x[0] for x in estimation])
        ax1.plot(_x, [x[1][0] for x in estimation])

        ax2 = fig.add_subplot(122)
        # ax2.plot([x[0] for x in estimation], [x[1][1] for x in estimation])
        ax2.plot([x[1][1] for x in estimation], [x[0] for x in estimation])

        plt.legend()
        plt.tight_layout()
        plt.show()

    def test_simulate_3d(self):
        x0 = np.array([0.0, 0.1, 0.1])
        P0 = np.eye(x0.shape[0])
        A = np.array([[1.0, 0.5, 1.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
        H = np.array([[1.0, 0.2, 0.0]])
        cov_means = (0.6, 1.0)

        kf = KalmanFilter.build(x0, P0, A, H, cov_means)

        num_points = 200
        rNoise = 3.4
        estimation = kf.simulate(num_points,
                                 lambda i: np.array([i + np.random.normal(rNoise, rNoise)]),
                                 np.array([0.4, 0.6, 0.1]))

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(121)
        x: np.array = np.linspace(0, num_points, num_points)
        _x = x.tolist()
        ax1.plot(_x, [x[0] for x in estimation])
        ax1.plot(_x, [x[1][0] for x in estimation])

        ax2 = fig.add_subplot(122)
        # ax2.plot([x[0] for x in estimation], [x[1][1] for x in estimation])
        ax2.plot([x[1][1] for x in estimation], [x[0] for x in estimation])

        plt.legend()
        plt.tight_layout()
        plt.show()

