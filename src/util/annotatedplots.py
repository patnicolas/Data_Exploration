__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2024  All rights reserved."


import numpy as np
from dataclasses import dataclass
from typing import AnyStr, Tuple, NoReturn, Optional


"""
Encapsulate the annotation for a given plot.
text: Description of the annotation
arrow_loc: Target of the arrow (point to) in coordinate
text_loc: Location of the text (start of the arrow)
"""


@dataclass
class PlotAnnotation:
    from matplotlib.axes import Axes

    text: AnyStr
    arrow_loc: Tuple[float, float]
    text_loc: Tuple[float, float]

    def show(self, ax: Axes) -> NoReturn:
        """
        Display the annotation with the parameters defined at member of the PlotAnnotation data class
        @param ax: Axes for displaying the annotation
        @type ax: axes.Axes
        """
        ax.annotate(
            text=self.text,
            xy=self.arrow_loc,
            xytext=self.text_loc,
            arrowprops={'width': 1,'facecolor': 'red', 'shrink': 0.1} )


"""
    Wraps the mechanism to annotate 2D/scatter plots 
"""


class AnnotatedPlots(object):
    def __init__(self, _x: np.array, _y: np.array) -> None:
        """
        Constructor for miscellaneous plots
        @param _x: First set of values
        @type _x: Numpy array
        @param _y: Second set of values
        @type _y: Numpy array
        """
        self.x = _x
        self.y = _y

    def plot(self, plt_annotation: Optional[PlotAnnotation] = None) -> NoReturn:
        """
        Display a 2D plot with x, y values and optional annotation.
        @param plt_annotation: Descriptor for the annotation
        @type plt_annotation: Optional PlotAnnotation
        """
        import matplotlib.pyplot as plt

        ax = plt.axes()
        ax.plot(self.x, self.y)
        if plt_annotation is not None:
            plt_annotation.show(ax)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    x = np.linspace(0.0, 100.0, 100)
    y = np.exp(-x*0.05)*np.sin(x)
    misc_plots = AnnotatedPlots(x, y)
    plot_annotation = PlotAnnotation('Spring relaxation', (10.0, 0.6), (2.0, 0.8))
    misc_plots.plot(plot_annotation)
