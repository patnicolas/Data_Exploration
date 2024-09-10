__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import Dict, List, AnyStr, NoReturn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar


class TAScatter(object):
    def __init__(self,
                 data: List[Dict[AnyStr, np.array]],
                 title: AnyStr,
                 annotation_data: np.array = []) -> None:
        self.data = data
        self.title = title
        self.annotation_data = annotation_data

    def visualize(self) -> NoReturn:
        match len(self.data):
            case 2:
                self.__visualize_2d()
            case 3:
                self.__visualize_3d()
            case 4:
                self.__visualize_4d()
            case _:
                raise Exception(f'Incorrect number of features {len(self.data)}')

    """ --------------------------- Private helper methods -----------------------   """

    def __visualize_2d(self) -> NoReturn:
        # Create the scatter plot
        plt.scatter(self.data[0]['values'], self.data[1]['values'], cmap='viridis')
        # Labels for axes
        plt.xlabel(self.data[0]['label'])
        plt.ylabel(self.data[1]['label'])
        plt.title(self.title)
        plt.legend()
        # Show plot
        plt.show()

    def __visualize_3d(self) -> NoReturn:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Create the scatter plot
        ax.scatter(self.data[0]['values'], self.data[1]['values'], self.data[2]['values'], cmap='viridis')
        self.__set_labels(ax)
        self.__add_annotation_points(ax)
        # Show plot
        plt.show()

    def __add_annotation_points(self, ax: Axes) -> NoReturn:
        if len(self.annotation_data) > 0:
            indices: List[int] = list(self.annotation_data)
            x_values = [x for idx, x in enumerate(self.data[0]['values']) if idx in indices]
            y_values = [x for idx, x in enumerate(self.data[1]['values']) if idx in indices]
            z_values = [x for idx, x in enumerate(self.data[2]['values']) if idx in indices]
            ax.scatter(
                x_values,
                y_values,
                z_values,
                color='red',
                s=120,
                edgecolor='red')
            x_text = min(x_values) - 16
            y_text = min(y_values) - 16
            z_text = min(z_values) - 16
            font = {'family': 'serif',
                    'color': 'red',
                    'weight': 'bold',
                    'size': 13,
                    }
            ax.text(x_text, y_text, z_text, 'Trade entries', fontdict =font, size=13, zorder=1)


    def __visualize_4d(self) -> NoReturn:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Create the scatter plot
        scatter = ax.scatter(
            self.data[0]['values'],
            self.data[1]['values'],
            self.data[2]['values'],
            c=self.data[3]['values'],
            cmap='viridis')

        # Add color bar which maps values to colors
        cbar = fig.colorbar(scatter, ax=ax, shrink=1.0, aspect=20)
        self.__set_labels(ax, cbar)
        self.__set_labels(ax)
        self.__add_annotation_points(ax)
        # Show plot
        plt.show()

    def __set_labels(self, ax: Axes, color_bar: Colorbar = None):
        if color_bar is not None:
            ax.set_zlabel(self.data[2]['label'])
            color_bar.set_label(self.data[3]['label'])
        elif len(self.data) > 2:
            ax.set_zlabel(self.data[2]['label'])

        ax.set_xlabel(self.data[0]['label'])
        ax.set_ylabel(self.data[1]['label'])
        ax.set_title(self.title)