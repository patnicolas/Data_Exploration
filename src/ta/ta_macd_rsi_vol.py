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
        """
        Default constructor for the computation of MACD histogram vs Volume vs RSI vs Price
        @param ticker: Ticker symbol
        @type ticker: str
        @param signal_line: Signal line for this ticker
        @type signal_line: Numpy array
        @param histogram: MACD histogram for this ticker
        @type histogram: Numpy array
        @param volumes: Daily volume for this ticker
        @type volumes: Numpy array
        @param rsi: Relative strength index for this ticker
        @type rsi: Numpy array
        @param prices: Closing prices for this ticker
        @type prices: Numpy array
        """
        super(TAMacdRsiVol, self).__init__('M.A.C.D. - RSI - Volume - Price', signal_line, histogram, prices)
        self.name = self.name + '- RSI - Volume - Price'
        self.ticker = ticker
        self.volumes = volumes
        self.rsi = rsi

    @classmethod
    def build(cls, _ta_ticker: TATicker) -> Self:
        """
        Alternative constructor using the fully defined TA ticker data
        @param _ta_ticker: Ticker instance containing ticker symbole, volume, high, low and closing prices
        @type _ta_ticker: TATicker class
        @return: Instance of this TAMacdRsiVol
        @rtype: TAMacdRsiVol class
        """
        signal_line, macd_hist, offset = TAMacd._compute_hist(_ta_ticker)
        ta_rsi = TARsi.build(_ta_ticker)
        rsi = ta_rsi.rsi

        assert len(macd_hist) == len(rsi[offset-15:]), \
            f'MACD length {len(macd_hist)} does not match RSI length { len(rsi[offset-15:])}'
        assert len(macd_hist) == len(_ta_ticker.volumes[offset:]), \
            f'MACD length {len(macd_hist)} does not match RSI length {len(_ta_ticker.volumes[offset:])}'

        return cls(
            ticker=_ta_ticker.ticker,
            signal_line=signal_line,
            histogram=macd_hist,
            volumes=_ta_ticker.volumes[offset:],
            rsi=rsi[offset-15:],
            prices=_ta_ticker.closes[offset:])

    def __str__(self) -> AnyStr:
        return f'\nTicker: {self.ticker}\nSignal  line:\n{self.signal_line}\nHistogram:\n{self.histogram}' \
               f'\nVolume:\n{self.volumes}\nRSI:\n{self.rsi}'

    def scatter(self, _annotated_data: np.array = None) -> np.array:
        """
        Scatter plot for this study with data point annotated by previous studies
        @param _annotated_data: Data point selected from previous studies, None if none were selected
        @type _annotated_data: Numpy Array
        @return: Newly annotated data point if any, None otherwise
        @rtype: Numpy array
        """
        from ta_scatter import TAScatter

        _data = [
            {'label': 'MACD Histogram', 'values': self.histogram},
            {'label': 'RSI', 'values': self.rsi},
            {'label': 'Volume', 'values': self.volumes},
            {'label': 'Prices $', 'values': self.prices},
        ]
        ta_scatter = TAScatter(_data, f'{self.name} [{self.ticker}]', _annotated_data)
        ta_scatter.visualize()
        return _annotated_data


if __name__ == '__main__':
    import yfinance as yf
    from ta_market_forecast import TAMarketForecast
    from ta_macd_vol_price import TAMacdVolPrice
    from ta_mfi import TAMfi

    data = yf.download('MO', start='2020-01-01', end='2024-09-01')
    ta_ticker = TATicker.build('WBA', data)

    ta_market_forecast = TAMarketForecast.build(ta_ticker)
    annotated_data = ta_market_forecast.scatter()

    ta_mfi = TAMfi.build(ta_ticker)
    print(str(ta_mfi))
    ta_mfi.scatter(annotated_data)




