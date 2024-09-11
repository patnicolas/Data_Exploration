__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, Self
import numpy as np
from ta_instrument import TAInstrument
from ta_ticker import TATicker


class TAMacd(TAInstrument):
    def __init__(self,
                 ticker: AnyStr,
                 signal_line: np.array,
                 histogram: np.array,
                 prices: np.array) -> None:
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
        signal_line, macd_hist, offset = TAMacd._compute_hist(_ta_ticker)
        return cls(_ta_ticker.ticker, signal_line, macd_hist, _ta_ticker.closes[offset:])

    @staticmethod
    def _compute_hist(_ta_ticker: TATicker) -> (np.array, np.array, int):
        from ta_mov_average import TAMovAverage, MovAverageType

        ema_12 = TAMovAverage.build(_ta_ticker, MovAverageType.exponential, 12, )
        ema_26 = TAMovAverage.build(_ta_ticker, MovAverageType.exponential, 26)

        macd_line = ema_12.mov_average[14:] - ema_26.mov_average
        signal_line = TAMovAverage.build_('EMA-9', MovAverageType.exponential, 9, macd_line).mov_average

        macd_hist = macd_line[8:] - signal_line
        offset = len(_ta_ticker.closes) - len(macd_hist)
        return signal_line, macd_hist, offset

