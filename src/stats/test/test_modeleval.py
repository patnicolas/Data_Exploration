import unittest
import path
import sys

directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
print(sys.path)
from modeleval import ModelEval, ModelType
from shapeval import SHAPException, SHAPPlotType


class TestModelEval(unittest.TestCase):
    def test_init(self):
        test_filename = '../../../data/Philippine_Air_Quality.csv'
        test_drop_features = ['datetime', 'coord.lon', 'coord.lat', 'extraction_date_time', 'city_name']
        test_label = 'main.aqi'
        test_validation_train_ratio = 0.15
        try:
            test_model_eval = ModelEval(test_filename, test_drop_features, test_label, test_validation_train_ratio)
            print(test_model_eval)
        except SHAPException as e:
            print(str(e))
        except Exception as e:
            print(str(e))

    def test_lr_shap_summary(self):
        test_filename = '../../../data/Philippine_Air_Quality.csv'
        test_drop_features = ['datetime', 'coord.lon', 'coord.lat', 'extraction_date_time', 'city_name']
        test_label = 'main.aqi'
        test_validation_train_ratio = 0.15
        try:
            test_model_eval = ModelEval(test_filename, test_drop_features, test_label, test_validation_train_ratio)
            test_model_eval(ModelType.LOGISTIC_REGRESSION, SHAPPlotType.SUMMARY_PLOT)
        except SHAPException as e:
            print(str(e))
        except Exception as e:
            print(str(e))

    def test_lr_shap_dependency(self):
        test_filename = '../../../data/Philippine_Air_Quality.csv'
        test_drop_features = ['datetime', 'coord.lon', 'coord.lat', 'extraction_date_time', 'city_name']
        test_label = 'main.aqi'
        test_validation_train_ratio = 0.15
        try:
            test_model_eval = ModelEval(test_filename, test_drop_features, test_label, test_validation_train_ratio)
            test_model_eval(ModelType.LOGISTIC_REGRESSION, SHAPPlotType.DEPENDENCY_PLOT)
        except SHAPException as e:
            print(str(e))
        except Exception as e:
            print(str(e))

    def test_svm_shap_dependency(self):
        test_filename = '../../../data/Philippine_Air_Quality.csv'
        test_drop_features = ['datetime', 'coord.lon', 'coord.lat', 'extraction_date_time', 'city_name']
        test_label = 'main.aqi'
        test_validation_train_ratio = 0.15
        try:
            test_model_eval = ModelEval(test_filename, test_drop_features, test_label, test_validation_train_ratio)
            test_model_eval(ModelType.SVM, SHAPPlotType.DEPENDENCY_PLOT)
        except SHAPException as e:
            print(str(e))
        except Exception as e:
            print(str(e))


if __name__ == '__main__':
    unittest.main()