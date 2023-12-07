from unittest import TestCase
from shap import SHAP
import sys
import os


class SHAPEvalTest(TestCase):

    def test_init(self):
        filename = 'data/Philippine_Air_Quality.csv'
        col_name = 'main.aqi'
        test_size = 0.15
        try:
            shap = SHAP(filename, col_name, test_size)
        except Exception as e:
            print(str(e))

    def test_logistic_regression(self):
        test_filename = '../../data/Philippine_Air_Quality.csv'
        test_drop_features = ['datetime', 'coord.lon', 'coord.lat', 'extraction_date_time', 'city_name']
        test_label = 'main.aqi'
        test_size = 0.15
        try:
            shap_eval = SHAPEval(test_filename, test_drop_features, test_label, test_size)
            test_metrics = shap_eval('logistic_regression')
            print(str(test_metrics))
        except Exception as e:
            print(str(e))

    def test_svm(self):
        test_filename = '../../data/Philippine_Air_Quality.csv'
        test_drop_features = ['datetime', 'coord.lon', 'coord.lat', 'extraction_date_time', 'city_name']
        test_label = 'main.aqi'
        test_size = 0.15
        try:
            shap_eval = SHAPEval(test_filename, test_drop_features, test_label, test_size)
            test_metrics = shap_eval('svm')
            print(str(test_metrics))
        except Exception as e:
            print(str(e))


