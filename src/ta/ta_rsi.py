__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from ta_instrument import TAInstrument
from typing import AnyStr, Self
import numpy as np
from ta_ticker import TATicker


class TARsi(TAInstrument):
    def __init__(self, ticker: AnyStr, rsi: np.array, prices: np.array) -> None:
        super(TARsi, self).__init__('RSI', prices)
        self.ticker = ticker
        self.rsi = rsi

    @staticmethod
    def compute_rsi(_ta_ticker: TATicker) -> np.array:
        gains = []
        losses = []
        for i in range(1, 14):
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
                av_gain = (ave_gain * 13 + diff) / 14
                new_rsi = 100 - (100.0 / (1 + float(av_gain / ave_loss)))
            else:
                av_loss = (ave_loss * 13 + diff) / 14
                new_rsi = 100 - (100.0 / (1 + float(ave_gain / av_loss)))
            rsi.append(new_rsi)

        return np.array(rsi)

    @classmethod
    def build(cls, _ta_ticker: TATicker) -> Self:
        rsi = TARsi.compute_rsi(_ta_ticker)
        return cls(_ta_ticker.ticker, rsi, _ta_ticker.closes)

