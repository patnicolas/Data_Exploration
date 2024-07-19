__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import Tuple, NoReturn, AnyStr, Optional
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



class SPDMatricesClassifier(object):
    def __init__(self,
                 classifier,
                 spd_metric: SPDMetric,
                 spd_training_data: SPDTrainingData) -> None:
        """
        Constructor for the classifier of SPD matrices dataset
        @param classifier: Classifier (i.e. SVM, KNN,..)
        @param spd_metric: Metric enumerator {Riemann,...}
        @type spd_metric: SPDMetric enumeration
        @param spd_training_data: Data class containing training data
        @type spd_training_data: SPDTrainingData
        """
        self.classifier = classifier
        self.spd_metric = spd_metric
        self.spd_training_data = spd_training_data

    def __str__(self) -> AnyStr:
        return f'\nClassifier: {str(self.classifier)}\nMetric: {self.spd_metric}\n{str(self.spd_training_data)}'

    def score(self) -> float:
        """
        Select metric, train the classifier and compute the score for the given metric
        @return: Score [0, 1] interva;
        @rtype: float
        """
        self.classifier.set_params(**{'metric': str(self.spd_metric.value)})
        self.classifier.fit(self.spd_training_data.train_X, self.spd_training_data.train_y)

        return self.classifier.score(
            self.spd_training_data.test_X,
            self.spd_training_data.test_y)

    def plot(self, title: Optional[AnyStr] = None) -> NoReturn:
        """
        Train, score classifier on data set defined in spd_training_data.
        Create scatter plots for train and test data then visualize the
        various decision boundaries
        @param title: Optional title for the scatter plot
        @type title: Optional[AnyStr]
        """
        from classifiers.spdmatricesdataset import SPDMatricesDataset

        # Step 1: Train and compute the score for the classifier
        score: float = self.score()
        print(f'Score for {title}: {score}')

        # Step 2: Create scatter plots
        ax = SPDMatricesDataset.create_scatter_plots(
            self.spd_training_data,
            fig=plt.figure(figsize=(16, 8))
        )

        # Step 3: Create contour for decision boundaries
        self.__create_contour(ax, self.spd_training_data.get_spd_dataset_limits())

        font_dict =  {
            'family': 'serif',
            'color':  'darkred',
            'weight': 'bold',
            'size': 20,
        }
        plt.title(title, fontdict=font_dict)

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
        ax.contourf(_x, _y, _z, zdir='z', cmap= plt.get_cmap('RdGy'), offset=spd_dataset_limits.in_z_min)

        _x, _z = np.meshgrid(axis_1, axis_3)
        _y = features[:, 0, 1].mean() * np.ones_like(_x)
        _y = SPDMatricesClassifier.get_probability(_x, _y, _z, clf=self.classifier)
        _y = np.ma.masked_where(~np.isfinite(_y), _y)
        ax.contourf(_x, _y, _z, zdir='y', cmap= plt.get_cmap('RdGy'), offset=spd_dataset_limits.in_y_max)

        _y, _z = np.meshgrid(axis_2, axis_3)
        _x = features[:, 0, 0].mean() * np.ones_like(_y)
        _x = SPDMatricesClassifier.get_probability(_x, _y, _z, clf=self.classifier)
        _x = np.ma.masked_where(~np.isfinite(_y), _y)
        ax.contourf(_x, _y, _z, zdir='x', cmap= plt.get_cmap('RdGy'), offset=spd_dataset_limits.in_y_min)

    @staticmethod
    @partial(np.vectorize, excluded=['clf'])
    def get_probability(cov_00: np.array, cov_01: np.array, cov_11: np.array, clf):
        cov = np.array(
            [[cov_00, cov_01, 0.0, 0.0], [cov_01, cov_11, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        u = cov[np.newaxis, ...]
        return clf.predict_proba(u)[0, 1]