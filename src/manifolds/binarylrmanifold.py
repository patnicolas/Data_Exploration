__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from dataclasses import dataclass
from typing import AnyStr
import numpy as np


@dataclass
class TrainingEvalData:
    X_train: np.array
    X_eval: np.array
    y_train: np.array
    y_eval:  np.array

    def shape(self) -> int:
        return self.X_train[0].shape()

    def n_train_samples(self) -> int:
        return len(self.X_train)

    def n_eval_samples(self) -> int:
        return len(self.X_eval)
    def __str__(self) -> AnyStr:
        return  f'X_train[0]: {self.X_train[0]}\nX_eval[0]: {self.X_eval[0]}' \
                f'\ny_train[0]: {self.y_train[0]}\ny_eval: {self.y_eval[0]}'


class BinaryLRManifold(object):
    def __init__(self, n_features: int, n_samples: int, train_eval_split: float = 0.9):
        """
        Constructor for the Binary Logistic Regression classifier
        @param n_features: Number of features
        @type n_features: int
        @param n_samples: Number of samples
        @type n_samples: int
        @param train_eval_split: Train and eval split
        @type train_eval_split: training/evaluation data set split ratio
        """
        assert(n_features > 1, f'Number of features {n_features} should be > 1')
        assert (n_samples > 0, f'Number of features {n_samples} should be > 0')
        assert (0.5 <= train_eval_split <= 0.95, f'Number of features {train_eval_split} should be [0.5, 0.95]')

        self.n_features = n_features
        self.n_samples = n_samples
        self.train_eval_split = train_eval_split

    def generate(self) -> TrainingEvalData:
        X, y = make_classification(
            n_samples = self.n_samples,
            n_features = self.n_features,
            n_redundant=10,
            n_clusters_per_class=1,
            flip_y=0.1,
            random_state=42)
        X_t, X_e, y_t, y_e = train_test_split(X, y, test_size=1.0 - self.train_eval_split)
        return TrainingEvalData(X_t, X_e, y_t, y_e)

    @staticmethod
    def train_euclidean(training_eval_data: TrainingEvalData) -> LogisticRegression:
        model = LogisticRegression()
        model.fit(training_eval_data.X_train, training_eval_data.y_train)
        return model

    @staticmethod
    def eval_euclidean(training_eval_data: TrainingEvalData, model: LogisticRegression) -> float:
        y_prediction = model.predict(training_eval_data.X_eval)
        return accuracy_score(training_eval_data.y_eval, y_prediction)


