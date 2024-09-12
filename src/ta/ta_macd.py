__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, Self
import numpy as np
from ta_study import TAStudy
from ta_ticker import TATicker


"""
Implementation of the computation of the Moving Average Convergence/Divergence (MACD) technical indicator.
- 12-day exponential moving average price: ema12
- 26-day exponential moving average price: ema16
- MACD line macd_line = ema12 - ema26
- Signal line  signal_line = ema9(macd_line)  9-day Exponential moving average of MACD line
- MACD histogram macd_hist = macd_line - signal_line
"""

class TAMacd(TAStudy):
    def __init__(self,
                 ticker: AnyStr,
                 signal_line: np.array,
                 histogram: np.array,
                 prices: np.array) -> None:
        """
        Default constructor for the computation of MACD signal line and histogram
        @param ticker: Ticker symbol
        @type ticker: str
        @param signal_line: Signal line for this ticker
        @type signal_line: Numpy array
        @param histogram: MACD histogram for this ticker
        @type histogram: Numpy array
        @param prices: Closing prices for this ticker
        @type prices: Numpy array
        """
        super(TAMacd, self).__init__('M.A.C.D.', prices)
        self.ticker = ticker
        self.signal_line = signal_line
        self.histogram = histogram

    @classmethod
    def build(cls, _ta_ticker: TATicker) -> Self:
        """
        Alternative constructor using the fully defined TA ticker data
        @param _ta_ticker: Ticker instance containing ticker symbole, volume, high, low and closing prices
        @type _ta_ticker: TATicker class
        @return: Instance of this TAMacd
        @rtype: TAMacd class
        """
        signal_line, _macd_hist, offset = TAMacd._compute_hist(_ta_ticker)
        return cls(
            ticker=_ta_ticker.ticker,
            signal_line=signal_line,
            histogram=_macd_hist,
            prices=_ta_ticker.closes[offset:])

    @staticmethod
    def _compute_hist(_ta_ticker: TATicker) -> (np.array, np.array, int):
        """
        Protected method to compute the MACD histogram
        @param _ta_ticker: Ticker instance containing ticker symbol, volume, high, low and closing prices
        @type _ta_ticker: TATicker class
        @return: Tuple (signal_line, MACD histogram, offset signal
        @rtype: Tuple (np.array, np.array. int)
        """
        from ta_mov_average import TAMovAverage, MovAverageType

        ema_12 = TAMovAverage.build(_ta_ticker, MovAverageType.exponential, 12, )
        ema_26 = TAMovAverage.build(_ta_ticker, MovAverageType.exponential, 26)

        macd_line = ema_12.mov_average[14:] - ema_26.mov_average
        signal_line = TAMovAverage.build_('EMA-9', MovAverageType.exponential, 9, macd_line).mov_average

        macd_hist = macd_line[8:] - signal_line
        offset = len(_ta_ticker.closes) - len(macd_hist)
        return signal_line, macd_hist, offset

