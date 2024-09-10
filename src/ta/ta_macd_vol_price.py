__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, NoReturn, Self
import numpy as np
from ta_instrument import TAInstrument
from ta_ticker import TATicker
from ta_macd import TAMacd


class TAMacdVolPrice(TAMacd):
    def __init__(self,
                 ticker: AnyStr,
                 signal_line: np.array,
                 histogram: np.array,
                 volumes: np.array,
                 prices: np.array) -> None:
        super(TAMacdVolPrice, self).__init__('M.A.C.D. - Volume - Price', signal_line, histogram, prices)
        self.ticker = ticker
        self.volumes = volumes

    @classmethod
    def build(cls, _ta_ticker: TATicker) -> Self:
        signal_line, macd_hist, offset = TAMacd._compute_hist(_ta_ticker)
        return cls(
            _ta_ticker.ticker,
            signal_line,
            macd_hist,
            _ta_ticker.volumes[offset:],
            _ta_ticker.closes[offset:])

    def scatter(self) -> NoReturn:
        from ta_scatter import TAScatter

        reversal_points = []
        _data = [
            {'label': 'MACD Histogram', 'values': self.histogram},
            {'label': 'Volume', 'values': self.volumes},
            {'label': 'Prices $', 'values': self.prices},
        ]
        ta_scatter = TAScatter(_data, f'{self.name} [{self.ticker}]', reversal_points)
        ta_scatter.visualize()

    def __str__(self) -> AnyStr:
        return f'\nTicker: {self.ticker}\nSignal  line:\n{self.signal_line}\nHistogram:\n{self.histogram}'


if __name__ == '__main__':
    import yfinance as yf

    data = yf.download('MO', start='2020-01-01', end='2024-09-01')
    ta_ticker = TATicker.build('WBA', data)
    ta_macd = TAMacdVolPrice.build(ta_ticker)
    print(str(ta_macd))
    ta_macd.scatter()



