__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import numpy as np
from typing import AnyStr, NoReturn
from matplotlib.axes import Axes


class SPDDatasetLimits(object):
    scale_factor: float = 0.02

    def __init__(self, features: np.array) -> None:
        self.in_x_min = features[:, 0, 0].min()
        self.in_x_max = features[:, 0, 0].max()
        self.in_y_min = features[:, 0, 1].min()
        self.in_y_max = features[:, 0, 1].max()
        self.in_z_min = features[:, 1, 1].min()
        self.in_z_max = features[:, 1, 1].max()

    def set_limits(self, ax: Axes) -> NoReturn:
        ax.set_xlim(self.in_x_min, self.in_x_max)
        ax.set_ylim(self.in_y_min, self.in_y_max)
        ax.set_zlim(self.in_z_min, self.in_z_max)

    def create_axis_values(self) -> (np.array, np.array, np.array):
        axis_x = SPDDatasetLimits.__axis_values(self.in_x_min, self.in_x_max)
        axis_y = SPDDatasetLimits.__axis_values(self.in_y_min, self.in_y_max)
        axis_z = SPDDatasetLimits.__axis_values(self.in_z_min, self.in_z_max)
        return axis_x, axis_y, axis_z

    def __str__(self) -> AnyStr:
        return f'Xmin: {self.in_x_min}, Xmax: {self.in_x_max}\nYmin: {self.in_y_min}, Ymax: {self.in_y_max}' \
               f'\nZmin: {self.in_z_min}, Zmax: {self.in_x_max}'

    """ --------------------  Private Helper Methods ----------------------------- """

    @staticmethod
    def __axis_values(x_min: float, x_max: float) -> np.array:
        return np.arange(x_min, x_max, (x_max - x_min) * SPDDatasetLimits.scale_factor)
