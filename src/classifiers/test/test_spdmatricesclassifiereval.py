import unittest

from classifiers.spdmatricesconfig import SPDMatricesConfig
from classifiers.spdmatricesclassifiereval import SPDMatricesClassifierEval
from pyriemann.classification import KNearestNeighbor, SVC


class SPDMatricesClassifierEvalTest(unittest.TestCase):

    def test_init(self):
        spd_matrices_classifier_eval = SPDMatricesClassifierEvalTest.__instance()
        print(str(spd_matrices_classifier_eval))
        self.assertEqual(len(spd_matrices_classifier_eval.classifiers), 2)

    def test_plot(self):
        spd_matrices_classifier_eval = SPDMatricesClassifierEvalTest.__instance()
        spd_matrices_classifier_eval.plot()


    @staticmethod
    def __instance() -> SPDMatricesClassifierEval:
        n_spd_matrices = 48
        n_channels = 4
        evals_lows_1 = 13
        evals_lows_2 = 11
        class_sep_ratio_1 = 1.0
        class_sep_ratio_2 = 0.5

        spd_matrices_eval_config = SPDMatricesConfig(
            n_spd_matrices,
            n_channels,
            evals_lows_1,
            evals_lows_2,
            class_sep_ratio_1,
            class_sep_ratio_2
        )
        classifiers = [KNearestNeighbor(n_neighbors=4), SVC(probability=True)]
        return SPDMatricesClassifierEval(classifiers, spd_matrices_eval_config)
