__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import numpy as np
from classifiers.spddatasetlimits import SPDDatasetLimits
from typing import List, AnyStr
from classifiers.spdtrainingdata import SPDTrainingData
from classifiers.spdmatricesconfig import SPDMatricesConfig
from matplotlib.axes import Axes


"""
Generate symmetric positive definite matrices/manifolds with Gaussian distribution
"""


class SPDMatricesDataset(object):
    def __init__(self, spd_matrices_config: SPDMatricesConfig) -> None:
        self.spd_matrices_config = spd_matrices_config
        self.target = np.concatenate([
            np.zeros(spd_matrices_config.n_spd_matrices), np.ones(spd_matrices_config.n_spd_matrices)
        ])
        self.rs = np.random.RandomState(42)

    def create(self) -> List[np.array]:
        spd_matrices_1 = self.__make_spd_matrices(self.spd_matrices_config.evals_lows_1)
        spd_matrices_2 = self.__make_spd_matrices(self.spd_matrices_config.evals_lows_2)

        return [
            (spd_matrices_1, self.target),
            (spd_matrices_2, self.target),
            self.__make_gaussian_blobs(self.spd_matrices_config.class_sep_ratio_1),
            self.__make_gaussian_blobs(self.spd_matrices_config.class_sep_ratio_2)
        ]

    @staticmethod
    def train_test_data(features: np.array, target: np.array) -> SPDTrainingData:
        from sklearn.model_selection import train_test_split

        train_X, test_X, train_y, test_y = train_test_split(
            features,
            target,
            test_size=0.3,
            random_state=42
        )
        return SPDTrainingData(train_X, test_X, train_y, test_y)

    def __str__(self) -> AnyStr:
        return f'\nConfig: {str(self.spd_matrices_config)}\nInitial target:\n{str(self.target)}'

    @staticmethod
    def plot(spd_training_data: SPDTrainingData, features: np.array) -> (np.array, np.array, np.array):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 8))

        ax = SPDMatricesDataset.create_scatter_plots(spd_training_data, (1, 1), fig)
        font_dict = {
            'family': 'serif',
            'color':  'darkred',
            'weight': 'bold',
            'size': 20,
        }
        plt.title('Training data set 3', fontdict=font_dict)
        spd_dataset_limits = SPDDatasetLimits(features)
        spd_dataset_limits.set_limits(ax)
        return spd_dataset_limits.create_axis_values()

    @staticmethod
    def create_scatter_plots(
            spd_training_data: SPDTrainingData,
            fig) -> Axes:
        import matplotlib.pyplot as plt

        ax = fig.add_subplot(111, projection='3d')
        my_cmap = plt.get_cmap('hsv')

        ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.3, alpha=0.2)
        x = spd_training_data.train_X[:, 0, 0]
        y = spd_training_data.train_X[:, 0, 1]
        z = spd_training_data.train_X[:, 1, 1]
        sc = ax.scatter3D(x, y, z, c=spd_training_data.train_y, cmap=my_cmap)

        ax.set_xlabel('X-axis', fontweight='bold')
        ax.set_ylabel('Y-axis', fontweight='bold')
        ax.set_zlabel('Z-axis', fontweight='bold')
        fig.colorbar(sc, ax=ax, shrink=0.3, aspect=5)

        x = spd_training_data.test_X[:, 0, 0],
        y = spd_training_data.test_X[:, 0, 1],
        z = spd_training_data.test_X[:, 1, 1]
        ax.scatter3D(x, y, z, c=spd_training_data.test_y, marker='^')
        return ax

    """ --------------------  Private Helper Methods ------------------------ """

    def __make_spd_matrices(self, evals_low_2: int) -> np.array:
        from pyriemann.datasets import make_matrices

        evals_range = 4
        return np.concatenate([
            make_matrices(
                self.spd_matrices_config.n_spd_matrices,
                self.spd_matrices_config.n_channels,
                'spd',
                self.rs,
                evals_low=10,
                evals_high=10 + evals_range),
            make_matrices(
                self.spd_matrices_config.n_spd_matrices,
                self.spd_matrices_config.n_channels,
                'spd',
                self.rs,
                evals_low=evals_low_2,
                evals_high=evals_low_2 + evals_range),
        ])

    def __make_gaussian_blobs(self, class_separation_ratio: float) -> np.array:
        from pyriemann.datasets import make_gaussian_blobs
        return make_gaussian_blobs(
            2 * self.spd_matrices_config.n_spd_matrices,
            self.spd_matrices_config.n_channels,
            random_state=self.rs,
            class_sep=class_separation_ratio,
            class_disp=0.5,
            n_jobs=4
        )
