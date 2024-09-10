__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, NoReturn, Self
import numpy as np
from tainstrument import TAInstrument
from taticker import TATicker


class TAMACD(TAInstrument):
    def __init__(self,
                 ticker: AnyStr,
                 signal_line: np.array,
                 histogram: np.array,
                 volumes: np.array,
                 prices: np.array) -> None:
        super(TAMACD, self).__init__('M.A.C.D. - Volume - Price', prices)
        self.ticker = ticker
        self.signal_line = signal_line
        self.histogram = histogram
        self.volumes = volumes

    @classmethod
    def build(cls, _ta_ticker: TATicker) -> Self:
        from tamovaverage import TAMovAverage, MovAverageType

        ema_12 = TAMovAverage.build(ta_ticker, MovAverageType.exponential, 12,)
        ema_26 = TAMovAverage.build(ta_ticker, MovAverageType.exponential, 26)

        macd_line = ema_12.mov_average[14:] - ema_26.mov_average
        signal_line = TAMovAverage.build_('EMA-9', MovAverageType.exponential, 9, macd_line).mov_average

        macd_hist = macd_line[8:] - signal_line
        offset = len(_ta_ticker.closes) - len(macd_hist)
        return cls(
            _ta_ticker.ticker,
            signal_line,
            macd_hist,
            _ta_ticker.volumes[offset:],
            _ta_ticker.closes[offset:])

    def scatter(self) -> NoReturn:
        from tascatter import TAScatter

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
    ta_macd = TAMACD.build(ta_ticker)
    print(str(ta_macd))
    ta_macd.scatter()



