__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import numpy as np
from manifolds.spddatasetlimits import SPDDatasetLimits

"""
Generate symmetric positive definite matrices/manifolds with Gaussian distribution
"""


class SPDMatricesDataset(object):
    def __init__(self, num_spds: int, num_channels: int) -> None:
        self.num_spds = num_spds
        self.num_channels = num_channels
        self.target = np.concatenate([np.zeros(num_spds), np.ones(num_spds)])
        self.rs = np.random.RandomState(42)

    def __call__(self,
                 evals_low_1: int,
                 evals_low_2: int,
                 class_separation_ratio_1: float,
                 class_separation_ratio_2: float) -> np.array:
        spd_matrices_1 = self.__make_spd_matrices(evals_low_1)
        spd_matrices_2 = self.__make_spd_matrices(evals_low_2)

        return [
            (spd_matrices_1, self.target),
            (spd_matrices_2, self.target),
            self.__make_gaussian_blobs(class_separation_ratio_1),
            self.__make_gaussian_blobs(class_separation_ratio_2)
        ]

    @staticmethod
    def plot_datasets(features: np.array, target: np.array) -> (np.array, np.array, np.array):
        from sklearn.model_selection import train_test_split
        import matplotlib.pyplot as plt

        dataset_size = len(features[0])
        in_train, in_test, target_train, target_test = train_test_split(
            features,
            target,
            test_size=0.3,
            random_state=42
        )
        x = in_train[:, 0, 0]
        y = in_train[:, 0, 1]
        z = in_train[:, 1, 1]

        fig = plt.figure(figsize=(24, 14))
        # ax = plt.axes(projection="3d")
        ax = plt.subplot(dataset_size, 1, 1, projection='3d')
        my_cmap = plt.get_cmap('hsv')

        ax.grid(b=True, color='grey',linestyle='-.', linewidth=0.3, alpha=0.2)
        sc = ax.scatter3D(x, y, z, c=target_train, cmap=my_cmap)

        plt.title("Input data")
        ax.set_xlabel('X-axis', fontweight='bold')
        ax.set_ylabel('Y-axis', fontweight='bold')
        ax.set_zlabel('Z-axis', fontweight='bold')
        fig.colorbar(sc, ax=ax, shrink=0.3, aspect=5)

        x = in_test[:, 0, 0],
        y = in_test[:, 0, 1],
        z = in_test[:, 1, 1]
        sc2 = ax.scatter3D(x, y, z, c=target_test, marker='^')
        # fig.colorbar(sc2, ax=ax, shrink=0.3, aspect=5)

        spd_dataset_limits = SPDDatasetLimits(features)
        spd_dataset_limits.set_limits(ax)

        plt.show()
        return spd_dataset_limits.create_axis_values()

    """ --------------------  Private Helper Methods ------------------------ """

    def __make_spd_matrices(self, evals_low_2: int) -> np.array:
        from pyriemann.datasets import make_matrices

        evals_range = 4
        return np.concatenate([
            make_matrices(
                self.num_spds,
                self.num_channels,
                'spd',
                self.rs,
                evals_low=10,
                evals_high=10 + evals_range),
            make_matrices(
                self.num_spds,
                self.num_channels,
                'spd', self.rs,
                evals_low=evals_low_2,
                evals_high=evals_low_2 + evals_range),
        ])

    def __make_gaussian_blobs(self, class_separation_ratio: float) -> np.array:
        from pyriemann.datasets import make_gaussian_blobs
        return make_gaussian_blobs(
            2 * self.num_spds,
            self.num_channels,
            random_state=self.rs,
            class_sep=class_separation_ratio,
            class_disp=0.5,
            n_jobs=4
        )
