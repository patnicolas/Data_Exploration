import unittest
from pyriemann.classification import SVC, KNearestNeighbor
from classifiers.spdmatricesclassifier import SPDMetric
from classifiers.spdmatricesdataset import SPDMatricesDataset
from classifiers.spdmatricesclassifier import SPDMatricesClassifier
from classifiers.spdmatricesclassifier import SPDTrainingData
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


class SPDMatricesClassifierTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init(self):
        spd_matrices_classifier = SPDMatricesClassifierTest.__instance()
        self.assertEqual(spd_matrices_classifier.spd_metric, SPDMetric.euclid)

    @unittest.skip('Ignore')
    def test_score(self):
        spd_matrices_classifier = SPDMatricesClassifierTest.__instance()
        score = spd_matrices_classifier.score()
        print(score)
        self.assertTrue(0.0 < score < 1.0)

    def test_plot(self):
        spd_matrices_classifier = SPDMatricesClassifierTest.__instance(1, SPDMetric.euclid, SVC(probability=True))
        spd_matrices_classifier.plot('SVC - Euclidean - Set 2')
        plt.show()

    def test_plot_2(self):
        spd_matrices_classifier = SPDMatricesClassifierTest.__instance(1, SPDMetric.riemann, SVC(probability=True))
        spd_matrices_classifier.plot('SVC - Riemann - Set 2')
        plt.show()

    @staticmethod
    def __instance(dataset_index: int, spd_metric: SPDMetric, classifier) -> SPDMatricesClassifier:
        from classifiers.spdmatricesconfig import SPDMatricesConfig

        n_spd_matrices = 48
        n_channels = 4
        evals_lows_1 = 13
        evals_lows_2 = 11
        class_sep_ratio_1 = 1.0
        class_sep_ratio_2 = 0.5

        spd_matrices_config = SPDMatricesConfig(
                n_spd_matrices,
                n_channels,
                evals_lows_1,
                evals_lows_2,
                class_sep_ratio_1,
                class_sep_ratio_2
            )
        spd_matrices_dataset = SPDMatricesDataset(spd_matrices_config)
        datasets = spd_matrices_dataset.create()
        features, target = datasets[dataset_index]

        in_train, in_test, target_train, target_test = train_test_split(
            features,
            target,
            test_size=0.3,
            random_state=42
        )
        spd_training_data = SPDTrainingData(in_train, in_test, target_train, target_test)
        return SPDMatricesClassifier(classifier, spd_metric, spd_training_data)
