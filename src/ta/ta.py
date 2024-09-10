__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List, Self, AnyStr, Dict
import yfinance as yf
from taticker import TATicker



class TA(object):
    csv_path: AnyStr = 'data'

    def __init__(self, ta_tickers: List[TATicker]):
        self.ta_tickers = ta_tickers

    @classmethod
    def build(cls, tickers: List[AnyStr], start_date: AnyStr, end_date: AnyStr) -> Self:
        ta_tickers = [TATicker.build(ticker, yf.download(ticker, start=start_date, end=end_date)) for ticker in tickers]
        return cls(ta_tickers)

    def __str__(self) -> AnyStr:
        return '\n\n'.join([str(ta_ticker) for ta_ticker in self.ta_tickers])



if __name__ == '__main__':

    ta = TA.build(['MSFT', 'MO'], '2021-01-01', '2024-08-01')
    print(str(ta))



