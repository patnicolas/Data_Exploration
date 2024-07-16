__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List
from classifiers.spdmatricesclassifier import SPDMetric
from classifiers.spdtrainingdata import SPDTrainingData
from classifiers.spdmatricesdataset import SPDMatricesDataset
from classifiers.spdmatricesconfig import SPDMatricesConfig

class SPDMatricesClassifierEval(object):

    def __init__(self,
                 classifiers: [],
                 spd_matrices_eval_config: SPDMatricesConfig) -> None:
        assert len(classifiers) > 1, f'Need at least two classifiers for evaluation'

        self.classifiers = classifiers
        self.metrics = [SPDMetric.euclid, SPDMetric.riemann]
        #n_spd_matrices = 48
        #n_channels = 4
        # evals_lows_1 = 13
        # evals_lows_2 = 11
        # class_sep_ratio_1 = 1.0
        # class_sep_ratio_2 = 0.5
        self.spd_matrices_dataset = SPDMatricesDataset(spd_matrices_eval_config)


    def plot(self):
        features, labels = self.spd_matrices_dataset.create()

        axis_1, axis_2, axis_3 = self.spd_matrices_dataset.plot(features, labels)


