__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, Self
import numpy as np
from ta.ta_ticker import TATicker
from ta.ta_study import TAStudy

"""
Implementation Relative Strength Index (RSI)
    Average_gain = (old_average_gain * (window-1) + new_gain)/window
    Average_loss = (old_average_loss * (window-1) + new_loss)/window
    RSi = 100*(1 - 1/(1 + Average_gain/Average_loss)
"""


class TARsi(TAStudy):
    window: int = 14

    def __init__(self, ticker: AnyStr, rsi: np.array, prices: np.array) -> None:
        super(TARsi, self).__init__('RSI', prices)
        self.ticker = ticker
        self.rsi = rsi


    @classmethod
    def build(cls, _ta_ticker: TATicker) -> Self:
        """
        Alternative constructor using the fully defined TA ticker data
        @param _ta_ticker: Ticker instance containing ticker symbole, volume, high, low and closing prices
        @type _ta_ticker: TATicker class
        @return: Instance of this TARsi
        @rtype: TARsi class
        """
        rsi = TARsi.compute_rsi(_ta_ticker)
        return cls(ticker=_ta_ticker.ticker, rsi=rsi, prices=_ta_ticker.closes)

    @staticmethod
    def compute_rsi(_ta_ticker: TATicker) -> np.array:
        gains = []
        losses = []
        for i in range(1, TARsi.window):
            diff = _ta_ticker.closes[i] - _ta_ticker.closes[i - 1]
            if diff >= 0.0:
                gains.append(diff)
            else:
                losses.append(diff)
        ave_gain = sum(gains) / len(gains) if len(gains) > 0 else 0.0
        ave_loss = sum(losses) / len(losses) if len(losses) > 0 else 0.0
        rsi = []
        for i in range(15, len(_ta_ticker.closes)):
            diff = _ta_ticker.closes[i] - _ta_ticker.closes[i - 1]
            if diff > 0.0:
                av_gain = (ave_gain * (TARsi.window-1) + diff) / TARsi.window
                new_rsi = 100 - (100.0 / (1 + float(av_gain / ave_loss)))
            else:
                av_loss = (ave_loss * (TARsi.window-1) + diff) / TARsi.window
                new_rsi = 100 - (100.0 / (1 + float(ave_gain / av_loss)))
            rsi.append(new_rsi)

        return np.array(rsi)

