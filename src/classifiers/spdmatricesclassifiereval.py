__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr
from classifiers.spdmatricesclassifier import SPDMetric
from classifiers.spdmatricesdataset import SPDMatricesDataset
from classifiers.spdmatricesconfig import SPDMatricesConfig


class SPDMatricesClassifierEval(object):

    def __init__(self,
                 classifiers: [],
                 spd_matrices_eval_config: SPDMatricesConfig) -> None:
        assert len(classifiers) > 1, f'Need at least two classifiers for evaluation'

        self.classifiers = classifiers
        self.metrics = [SPDMetric.euclid, SPDMetric.riemann]
        self.spd_matrices_dataset = SPDMatricesDataset(spd_matrices_eval_config)

    def plot(self):
        from classifiers.spdmatricesclassifier import SPDMatricesClassifier
        import matplotlib.pyplot as plt

        dataset = self.spd_matrices_dataset.create()
        print(f'Start processing {len(dataset)} data sets')
        # for dataset_idx in range(len(dataset)):
        for dataset_idx in range(2):
            features, labels = dataset[dataset_idx]
            training_data = SPDMatricesDataset.train_test_data(features, labels)

            self.spd_matrices_dataset.plot(training_data, features)
            for clf_index, classifier in enumerate(self.classifiers):
                for metric_index, metric in enumerate(self.metrics):
                    spd_matrices_classifier = SPDMatricesClassifier(classifier, metric, training_data)
                    spd_matrices_classifier.plot((clf_index+1, metric_index+2))
        plt.show()

    def __str__(self) -> AnyStr:
        return f"\nMetrics: {' '.join([str(metric) for metric in self.metrics])}\nData set: {self.spd_matrices_dataset}"

