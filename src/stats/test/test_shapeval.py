import unittest
import path
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
print(sys.path)
from shapeval import SHAPException, SHAPPlotType, SHAPEval
from modeleval import ModelEval


class TestSHAPEval(unittest.TestCase):


    def test_logistic_regression(self):
        features_names = ['x1', 'x2']
        X = [[0.5, 0.6], [0.6, 0.0], [0.1, 0.5], [0.3, 1.0], [0.9, 0.8], [0.2, 0.2], [0.1, 0.4], [0.1, 0.7]]

        try:
            model = LogisticRegression(solver='lbfgs', max_iter=1000, penalty='l2', multi_class='multinomial')
            shap_eval = SHAPEval(model.predict, SHAPPlotType.SUMMARY_PLOT)
            test_metrics = shap_eval(X, features_names)
            print(str(test_metrics))
        except Exception as e:
            print(str(e))

    def test_svm(self):
        import random

        features_names = ['x1', 'x2']
        num_data_points = 100
        X: List[List[float]] = [[x*0.01, 1 - x*0.01] for x in random.sample(range(0, 100), num_data_points)]

        try:
            model = SVC(kernel="rbf", decision_function_shape='ovo', random_state=ModelEval.random_state)
            shap_eval = SHAPEval(model.predict, SHAPPlotType.SUMMARY_PLOT)
            test_metrics = shap_eval(X, features_names)
            print(str(test_metrics))
        except Exception as e:
            print(str(e))

if __name__ == '__main__':
    unittest.main()