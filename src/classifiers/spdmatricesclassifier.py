__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import Tuple, NoReturn, AnyStr
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from classifiers.spddatasetlimits import SPDDatasetLimits
from classifiers.spdtrainingdata import SPDTrainingData
from functools import partial
from enum import Enum


class SPDMetric(Enum):
    riemann = 'riemann'
    euclid = 'euclid'


"""
@partial(np.vectorize, excluded=['clf'])
def get_probability(cov_00: np.array, cov_01: np.array, cov_11: np.array, clf):
    cov = np.array([[cov_00, cov_01, 0.0, 0.0], [cov_01, cov_11, 0.0,  0.0], [0.0, 0.0, 0.0,  0.0], [0.0, 0.0, 0.0,  0.0]])
    u = cov[np.newaxis, ...]
    return clf.predict_proba(u)[0, 1]
"""


class SPDMatricesClassifier(object):
    def __init__(self,
                 classifier,
                 spd_metric: SPDMetric,
                 spd_training_data: SPDTrainingData) -> None:
        self.classifier = classifier
        self.spd_metric = spd_metric
        self.spd_training_data = spd_training_data

    def __str__(self) -> AnyStr:
        return f'\nClassifier: {str(self.classifier)}\nMetric: {self.spd_metric}\n{str(self.spd_training_data)}'

    def score(self) -> float:
        self.classifier.set_params(**{'metric': str(self.spd_metric.value)})
        self.classifier.fit(self.spd_training_data.train_X, self.spd_training_data.train_y)

        return self.classifier.score(
            self.spd_training_data.test_X,
            self.spd_training_data.test_y)

    def plot(self, indices: Tuple[int, int]) -> Axes:
        from classifiers.spdmatricesdataset import SPDMatricesDataset

        spd_dataset_limits = SPDDatasetLimits(self.spd_training_data.get_features())
        dataset_size = len(self.spd_training_data)
        ax = plt.subplot(dataset_size, indices[0], indices[1], projection='3d')
        score: float = self.score()
        print(f'Score: {score}')
        self.__create_contour(ax, spd_dataset_limits)
        SPDMatricesDataset.create_scatter_plots(
            self.spd_training_data,
            (1, 2),
            fig=plt.figure(figsize=(24, 14))
        )
        ax.text(
            1.3 * spd_dataset_limits.in_x_max,
            spd_dataset_limits.in_y_min,
            spd_dataset_limits.in_z_min,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
            verticalalignment="bottom"
        )
        return ax

    """ -----------------   Helper methods  ----------------------------- """

    def __create_contour(self,
                         ax: Axes,
                         spd_dataset_limits: SPDDatasetLimits) -> NoReturn:
        features = self.spd_training_data.get_features()
        axis_1, axis_2, axis_3 = spd_dataset_limits.create_axis_values()

        _x, _y = np.meshgrid(axis_1, axis_2)
        _z = features[:, 1, 1].mean() * np.ones_like(_x)
        _z = SPDMatricesClassifier.get_probability(_x, _y, _z, self.classifier)
        _z = np.ma.masked_where(~np.isfinite(_z), _z)
        ax.contour(_x, _y, _z, zdir='z', offset=spd_dataset_limits.in_z_min)

        _x, _z = np.meshgrid(axis_1, axis_3)
        _y = features[:, 0, 1].mean() * np.ones_like(_x)
        _y = SPDMatricesClassifier.get_probability(_x, _y, _z, clf=self.classifier)
        _y = np.ma.masked_where(~np.isfinite(_y), _y)
        ax.contour(_x, _y, _z, zdir='y', offset=spd_dataset_limits.in_y_max)

        _y, _z = np.meshgrid(axis_2, axis_3)
        _x = features[:, 0, 0].mean() * np.ones_like(_y)
        _x = SPDMatricesClassifier.get_probability(_x, _y, _z, clf=self.classifier)
        _x = np.ma.masked_where(~np.isfinite(_y), _y)
        ax.contour(_x, _y, _z, zdir='x', offset=spd_dataset_limits.in_y_min)

    @staticmethod
    @partial(np.vectorize, excluded=['clf'])
    def get_probability(cov_00: np.array, cov_01: np.array, cov_11: np.array, clf):
        cov = np.array(
            [[cov_00, cov_01, 0.0, 0.0], [cov_01, cov_11, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        u = cov[np.newaxis, ...]
        return clf.predict_proba(u)[0, 1]


if __name__ == '__main__':
    print(str(SPDMetric.riemann.value))
