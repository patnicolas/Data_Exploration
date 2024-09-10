__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, NoReturn, Self
import numpy as np
from ta_macd import TAMacd
from ta_ticker import TATicker
from ta_rsi import TARsi


class TAMacdRsiVol(TAMacd):
    def __init__(self,
                 ticker: AnyStr,
                 signal_line: np.array,
                 histogram: np.array,
                 volumes: np.array,
                 rsi: np.array,
                 prices: np.array) -> None:
        super(TAMacdRsiVol, self).__init__('M.A.C.D. - Volume - Price', signal_line, histogram, prices)
        self.ticker = ticker
        self.volumes = volumes
        self.rsi = rsi

    @classmethod
    def build(cls, _ta_ticker: TATicker) -> Self:
        signal_line, macd_hist, offset = TAMacd._compute_hist(_ta_ticker)
        ta_rsi = TARsi.build(_ta_ticker)
        rsi = ta_rsi.rsi

        assert len(macd_hist) == len(rsi[offset-15:]), \
            f'MACD length {len(macd_hist)} does not match RSI length { len(rsi[offset-15:])}'
        assert len(macd_hist) == len(_ta_ticker.volumes[offset:]), \
            f'MACD length {len(macd_hist)} does not match RSI length {len(_ta_ticker.volumes[offset:])}'

        return cls(
            _ta_ticker.ticker,
            signal_line,
            macd_hist,
            _ta_ticker.volumes[offset:],
            rsi[offset-14:],
            _ta_ticker.closes[offset:])

    def __str__(self) -> AnyStr:
        return f'\nTicker: {self.ticker}\nSignal  line:\n{self.signal_line}\nHistogram:\n{self.histogram}' \
               f'\nVolume:\n{self.volumes}\nRSI:\n{self.rsi}'


if __name__ == '__main__':
    import yfinance as yf

    data = yf.download('MO', start='2020-01-01', end='2024-09-01')
    ta_ticker = TATicker.build('WBA', data)
    ta_macd = TAMacdRsiVol.build(ta_ticker)
    print(str(ta_macd))



