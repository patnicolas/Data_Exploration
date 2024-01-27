import unittest
import path
import sys
import os




class TestSHAPEval(unittest.TestCase):
    directory = path.Path(__file__).abspath()
    sys.path.append(directory.parent.parent)
    print(sys.path)
    from shapeval import SHAPEval

    def test_logistic_regression(self):
        test_filename = '../../../data/Philippine_Air_Quality.csv'
        test_drop_features = ['datetime', 'coord.lon', 'coord.lat', 'extraction_date_time', 'city_name']
        test_label = 'main.aqi'
        test_size = 0.15
        model = LogisticRegression(solver='lbfgs', max_iter=10000, penalty='l2', multi_class='multinomial')
        try:
            shap_eval = SHAPEval(model, )
            test_metrics = shap_eval('logistic_regression')
            print(str(test_metrics))
        except Exception as e:
            print(str(e))

    def test_svm(self):
        test_filename = '../../../data/Philippine_Air_Quality.csv'
        test_drop_features = ['datetime', 'coord.lon', 'coord.lat', 'extraction_date_time', 'city_name']
        test_label = 'main.aqi'
        test_size = 0.15
        try:
            shap_eval = SHAPEval(test_filename, test_drop_features, test_label, test_size)
            test_metrics = shap_eval('svm')
            print(str(test_metrics))
        except Exception as e:
            print(str(e))


if __name__ == '__main__':
    unittest.main()