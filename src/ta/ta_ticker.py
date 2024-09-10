__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List, Self, AnyStr, Dict, NoReturn
from dataclasses import dataclass
from ta_scatter import TAScatter
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class TATicker:
    ticker: AnyStr
    dates: List[AnyStr]
    closes: np.array
    volumes: np.array

    @classmethod
    def build(cls, ticker: AnyStr, df: pd.DataFrame) -> Self:
        print(df.head())
        path = f'../../data/{ticker}.csv'
        df.to_csv(path)
        ds = pd.read_csv(path, delimiter=',')
        return TATicker(ticker,
                        list(ds['Date'].values),
                        ds['Close'].values,
                        ds['Volume'].values)

    def __str__(self) -> AnyStr:
        return f'\nTicker: {self.ticker} --------------\ndates:\n{str(self.dates)}\nCloses:\n{self.closes}\nVolumes:\n{self.volumes}'

    @staticmethod
    def scatter(labeled_data: List[Dict[AnyStr, np.array]], title: AnyStr, annotation_data: np.array):
        ta_scatter = TAScatter(labeled_data, title, annotation_data)
        ta_scatter.visualize()
