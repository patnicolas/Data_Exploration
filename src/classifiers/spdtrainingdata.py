__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from dataclasses import dataclass
import numpy as np
from classifiers.spddatasetlimits import SPDDatasetLimits


@dataclass
class SPDTrainingData:
    train_X: np.array
    test_X: np.array
    train_y: np.array
    test_y: np.array

    def get_features(self) -> np.array:
        return np.concatenate([self.train_X, self.test_X], axis=0)

    def get_spd_dataset_limits(self) -> SPDDatasetLimits:
        return SPDDatasetLimits(self.get_features())

    def __len__(self) -> int:
        return len(self.train_X) + len(self.test_X)

    def __str__(self):
        return f'\nTrain features: {self.train_X[0]}\nTrain target: {self.train_y[0]}' \
               f'\nTest features: {self.test_X[0]}\nTest target: {self.test_y[0]}'
