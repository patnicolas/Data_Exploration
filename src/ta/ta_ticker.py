__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List, Self, AnyStr
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class TATicker:
    ticker: AnyStr
    dates: List[AnyStr]
    highs: np.array
    lows: np.array
    closes: np.array
    volumes: np.array

    @classmethod
    def build(cls, ticker: AnyStr, df: pd.DataFrame) -> Self:
        print(df.head())
        path = f'../../../data/{ticker}.csv'
        df.to_csv(path)
        ds = pd.read_csv(path, delimiter=',')
        return TATicker(ticker=ticker,
                        dates=list(ds['Date'].values),
                        highs=ds['High'].values,
                        lows=ds['Low'].values,
                        closes=ds['Close'].values,
                        volumes=ds['Volume'].values)

    def __str__(self) -> AnyStr:
        return f'\n ----- Ticker: {self.ticker} -------\ndates:\n{str(self.dates)}\nCloses:\n{self.closes}' \
               f'\nHighs:\n{self.highs}\nLows:\n{self.lows}\nVolumes:\n{self.volumes}'

    """
    @staticmethod
    def scatter(labeled_data: List[Dict[AnyStr, np.array]], title: AnyStr, annotation_data: np.array):
        ta_scatter = TAScatter(labeled_data, title, annotation_data)
        ta_scatter.visualize()
    """
