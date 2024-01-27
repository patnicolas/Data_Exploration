__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2024  All rights reserved."

import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, auc
from typing import AnyStr, List
from enum import Enum
from shapeval import SHAPEval
from dataclasses import dataclass
from shapeval import SHAPPlotType, SHAPException


class ModelType(Enum):
    LOGISTIC_REGRESSION = 1,
    SVM = 2,
    MLP = 3

    @staticmethod
    def is_model_supported(self, model_type: int) -> bool:
        return 0 < model_type < 4


@dataclass
class TestMetric:
    accuracy: float
    f1: float
    mean_squared_error: float


"""
    Basic class to evaluate extraction of SHAP features
    :param filename Name of CSV file
    :param dropped_features List of features/columns to be dropped from data frame
    :param label Label variable 
    :param val_train_split Fraction of data set split between validation and training sets
"""


class ModelEval(object):
    random_state = 5713

    def __init__(self,
                 filename: AnyStr,
                 dropped_features: List[AnyStr],
                 label: AnyStr,
                 val_train_split: float):
        assert 0.0 < val_train_split < 0.5, f'Validation-train split ratio {val_train_split} should be [0.0, 0.50]'

        def set_label(x: float) -> int:
            return int(x) - 1

        df = pd.read_csv(filename)
        # Drop non features and label columns
        dropped_features.append(label)
        X = df.drop(dropped_features, axis=1)
        # Apply standard normalization
        X_scaled = StandardScaler().fit(X).transform(X)
        # Select column containing label
        y = df[label].apply(set_label)
        # Train - validation split
        self.feature_names = X.columns.values.tolist()
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(X_scaled, y, test_size=val_train_split, random_state=ModelEval.random_state)

    def __call__(self, model_type: ModelType, plot_type: SHAPPlotType) -> TestMetric:
        """
            Train model over self.X_train and compute quality metrics using self.X_val
            :param model_type Type (enumerator) of model (only logistic regression, SVM and
                            multi-layer perceptron are supported)
            :return Test metric element
        """
        match model_type:
            case ModelType.LOGISTIC_REGRESSION:
                model = LogisticRegression(solver='lbfgs', max_iter=1000, penalty='l2', multi_class='multinomial')
            case ModelType.SVM:
                model = SVC(kernel="rbf", decision_function_shape='ovo', random_state=ModelEval.random_state)
            case ModelType.MLP:
                model = MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    max_iter=500,
                    alpha=0.0001,
                    solver='adam',
                    random_state=ModelEval.random_state)
            case _:
                raise Exception(f'Model name {model_type} is not supported')
        print(f'{model_type} trained with {len(self.X_train)} validated with {len(self.X_val)} samples')
        model.fit(self.X_train, self.y_train)

        shap_eval = SHAPEval(model.predict, plot_type)
        shap_eval(self.X_val,  self.feature_names)
        y_predicted = model.predict(self.X_val)

        return TestMetric(
            accuracy_score(self.y_val, y_predicted),
            f1_score(self.y_val, y_predicted, average='weighted'),
            mean_squared_error(self.y_val, y_predicted)
        )

    def __str__(self):
        return f'Features: {self.feature_names}'


if __name__ == '__main__':
    test_filename = '../../data/Philippine_Air_Quality.csv'
    test_drop_features = ['datetime', 'coord.lon', 'coord.lat', 'extraction_date_time', 'city_name']
    test_label = 'main.aqi'
    test_size = 0.05
    try:
        model_eval = ModelEval(test_filename, test_drop_features, test_label, test_size)
        test_metrics = model_eval(ModelType.MLP, SHAPPlotType.DEPENDENCY_PLOT)
        print(str(test_metrics))
    except SHAPException as e:
        print(str(e))
    except Exception as e:
        print(str(e))
