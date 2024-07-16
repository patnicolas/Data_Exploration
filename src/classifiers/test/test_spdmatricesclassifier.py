import unittest
from pyriemann.classification import SVC
from classifiers.spdmatricesclassifier import SPDMetric
from classifiers.spdmatricesdataset import SPDMatricesDataset
from classifiers.spdmatricesclassifier import SPDMatricesClassifier
from classifiers.spdmatricesclassifier import SPDTrainingData
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class SPDMatricesClassifierTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init(self):
        spd_matrices_classifier = SPDMatricesClassifierTest.__instance()
        self.assertEqual(spd_matrices_classifier.spd_metric, SPDMetric.euclid)

    def test_score(self):
        spd_matrices_classifier = SPDMatricesClassifierTest.__instance()
        score = spd_matrices_classifier.score()
        print(score)
        self.assertTrue(0.0 < score < 1.0)

    def test_plot(self):
        spd_matrices_classifier = SPDMatricesClassifierTest.__instance()
        spd_matrices_classifier.plot((1, 2))
        plt.show()

    @staticmethod
    def __instance() -> SPDMatricesClassifier:
        classifier = SVC(probability=True)
        spd_metric = SPDMetric.euclid
        n_spd_matrices = 48
        n_channels = 4
        spd_matrices_dataset = SPDMatricesDataset(n_spd_matrices, n_channels)
        evals_lows_1 = 13
        evals_lows_2 = 11
        class_sep_ratio_1 = 1.0
        class_sep_ratio_2 = 0.5

        datasets = spd_matrices_dataset(evals_lows_1, evals_lows_2, class_sep_ratio_1, class_sep_ratio_2)
        features, target = datasets[0]

        in_train, in_test, target_train, target_test = train_test_split(
            features,
            target,
            test_size=0.3,
            random_state=42
        )
        spd_training_data = SPDTrainingData(in_train, in_test, target_train, target_test)
        return SPDMatricesClassifier(classifier, spd_metric, spd_training_data)
