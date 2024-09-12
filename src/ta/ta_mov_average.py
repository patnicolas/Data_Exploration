__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import Self, AnyStr, List
import numpy as np
from enum import Enum
from ta_study import TAStudy
from ta_ticker import TATicker
import sys
sys.setrecursionlimit(2048)


class MovAverageType(Enum):
    simple = 'Simple'
    exponential = 'Exponential'


class TAMovAverage(TAStudy):
    def __init__(self, ticker: AnyStr, prices: np.array, mov_average: np.array) -> None:
        super(TAMovAverage, self).__init__(ticker, prices)
        self.mov_average = mov_average

    @classmethod
    def build_(cls, ticker: AnyStr, mov_average_type: MovAverageType, window_size: int, prices: np.array) -> Self:
        x = 0
        match mov_average_type:
            case MovAverageType.simple:
                x = TAMovAverage.__simple(window_size, prices)
            case MovAverageType.exponential:
                x = TAMovAverage.__exponential(window_size, prices)
        return cls(ticker=ticker, prices=prices, mov_average=x)

    @classmethod
    def build(cls, ta_ticker: TATicker, mov_average_type: MovAverageType, window_size: int) -> Self:
        return TAMovAverage.build_(ta_ticker.ticker, mov_average_type, window_size, ta_ticker.closes)

    """ ------------------- Private helper methods ---------------------  """

    @staticmethod
    def __simple(window_size: int, values: np.array) -> np.array:
        window = np.ones(window_size) / window_size
        return np.convolve(values, window, mode='valid')

    @staticmethod
    def __exponential(window_size: int, values: np.array) -> np.array:
        smooth_factor = 2 / (1 + window_size)

        def recurse(ema: np.array, index: int) -> np.array:
            if index >= len(ema):
                return ema
            ema[index] = smooth_factor * values[index] + (1 - smooth_factor) * ema[index - 1]
            return recurse(ema, index + 1)

        exp_mov = np.zeros_like(values)
        exp_mov[0] = values[0]
        return recurse(exp_mov, 0)[window_size - 1:]
