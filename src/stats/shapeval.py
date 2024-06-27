__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2024  All rights reserved."

import pandas as pd
import shap
import sys
from typing import List, AnyStr, NoReturn
from sklearn.linear_model import LogisticRegression
from enum import Enum


def init() -> NoReturn:
    print(f'Python version: {sys.version}')
    python_version = sys.version.split(".")
    if int(python_version[1]) < 10:
        raise Exception(f'Python version {sys.version} not supported!')
    shap.initjs()


class SHAPException(Exception):
    def __init__(self, comment: AnyStr) -> None:
        super(Exception, self).__init__(comment)


def custom_masker(mask, x):
    # in this simple example we just zero out the features we are masking
    return (x * mask).reshape(1, len(x))


"""
    Enumerator for the type of SHAP output plot
"""


class SHAPPlotType(Enum):
    SUMMARY_PLOT = 1,
    DEPENDENCY_PLOT = 2,
    DECISION_PLOT = 3,
    FORCE_PLOT = 4


class SHAPEval(object):
    def __init__(self, model_prediction, plot_type: SHAPPlotType):
        init()
        self.model_prediction = model_prediction
        self.plot_type = plot_type

    def __call__(self, validation_data: pd.array, column_names: List[AnyStr]) -> NoReturn:
        """
            Compute SHAP values and execute a specific plot for a given model prediction and
            validation data. As the model are not random forest and decision tree, we used the
            kernel mode for the explainer.

            :param validation_data: Pandas data frame of validation data
            :param column_names Name of Data frame column (~ features names)
        """
        # Step 1: Compute SHAP values - Kernel model
        shap_descriptor = shap.KernelExplainer(self.model_prediction, validation_data)
        shap_values = shap_descriptor.shap_values(validation_data)
        # Step 2:
        match self.plot_type:
            case SHAPPlotType.SUMMARY_PLOT:
                shap.summary_plot(shap_values, validation_data, feature_names=column_names)

            case SHAPPlotType.DEPENDENCY_PLOT:
                shap.dependence_plot("o3", shap_values, validation_data, feature_names=column_names)

            case SHAPPlotType.FORCE_PLOT:
                data_point_rank = 8
                shap.force_plot(
                    shap_descriptor.expected_value,
                    shap_values[data_point_rank, :],
                    validation_data[data_point_rank, :],
                    feature_names=column_names,
                    matplotlib=True)

            case SHAPPlotType.DECISION_PLOT:
                shap.decision_plot(
                    shap_descriptor.expected_value,
                    shap_values,
                    feature_names=column_names,
                    link='logit')
            case _:
                raise SHAPException(f'Plot type {self.plot_type} is not supported')


if __name__ == '__main__':
    test_filename = '../../data/Philippine_Air_Quality.csv'
    test_drop_features = ['datetime', 'coord.lon', 'coord.lat', 'extraction_date_time', 'city_name']
    test_label = 'main.aqi'
    test_size_fraction = 0.15

    try:
        model = LogisticRegression(solver='lbfgs', max_iter=10000, penalty='l2', multi_class='multinomial')
    except Exception as e:
        print(str(e))
