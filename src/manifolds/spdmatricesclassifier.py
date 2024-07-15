

from pyriemann.classification import ClassifierMixin
from typing import Tuple, NoReturn
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from manifolds.spddatasetlimits import SPDDatasetLimits
from functools import partial
from enum import Enum

class SPDMetric(Enum):
    riemann = 'riemann'
    euclid = 'euclid'


@partial(np.vectorize, excluded=['clf'])
def get_probability(cov_00: float, cov_01: float, cov_11: float, classifier: ClassifierMixin):
    cov = np.array([[cov_00, cov_01], [cov_01, cov_11]])
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return classifier.predict_proba(cov[np.newaxis, ...])[0, 1]


class SPDMatricesClassifier(object):

    def __init__(self,
                 classifier: ClassifierMixin,
                 spd_metric: SPDMetric,
                 train_features: np.array,
                 train_target: np.array,
                 test_features: np.array,
                 test_target: np.array                 ) -> None:
        self.classifier = classifier
        self.spd_metric = spd_metric
        self.train_features = train_features
        self.train_target = train_target
        self.test_features = test_features
        self.test_target = test_target

    def __call__(self, *args, **kwargs) -> float:
        self.classifier.set_params(**('metric', str(self.spd_metric.value)))
        self.classifier.fit(self.train_features, self.train_target)

        score = self.classifier.score(self.test_features, self.test_target)
        return score

    def plot_classifier(self,
                        dataset_size: int,
                        indices: Tuple[int, int],
                        spd_dataset_limits: SPDDatasetLimits) -> NoReturn:
        ax = plt.subplot(dataset_size, indices[0], indices[1], projection='3d')
        axis_x, axis_y, axis_z = spd_dataset_limits.create_axis_values()
        xx, yy = np.meshgrid(axis_x, axis_y)

    def create_contour(self,
                       axis_1: np.array,
                       axis_2: np.array,
                       axis_3: np.array,
                       ax: Axes,
                       spd_dataset_limits: SPDDatasetLimits) -> NoReturn:
        features = np.concatenate(self.train_features, self.test_features)
        _x, _y = np.meshgrid(axis_1, axis_2)
        _z = get_probability(_x, _y, features[:, 1, 1].mean().np.ones_like(_x), clf=self.classifier)
        _z = np.ma.masked_where(~np.isfinite(_z), _z)
        ax.contour(_x, _y, _z, zdir='z', offset=spd_dataset_limits.in_z_min)

        _x, _z = np.meshgrid(axis_1, axis_3)
        _y = get_probability(_x, features[:, 0, 1].mean().np.ones_like(_x), _z, clf=self.classifier)
        _y = np.ma.masked_where(~np.isfinite(_y), _y)
        ax.contour(_x, _y, _z, zdir='y', offset=spd_dataset_limits.in_y_max)

        _y, _z = np.meshgrid(axis_2, axis_3)
        _x = get_probability(features[:, 0, 0].mean().np.ones_like(_y), _y, _z, clf=self.classifier)
        _x = np.ma.masked_where(~np.isfinite(_y), _y)
        ax.contour(_x, _y, _z, zdir='x', offset=spd_dataset_limits.in_y_min)


if __name__ == '__main__':
    print(str(SPDMetric.riemann.value))