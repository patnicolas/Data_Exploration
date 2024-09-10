__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, NoReturn, Self, List
import numpy as np
from ta_instrument import TAInstrument
from ta_ticker import TATicker


class TAMarketForecast(TAInstrument):
    def __init__(self,
                 ticker: AnyStr,
                 prices: np.array,
                 momentum: np.array,
                 near_term: np.array,
                 intermediate: np.array) -> None:
        super(TAMarketForecast, self).__init__('Market Forecast', prices)
        self.ticker = ticker
        self.momentum = momentum
        self.near_term = near_term
        self.intermediate = intermediate

    @classmethod
    def build(cls, _ta_ticker: TATicker, time_frames: List[int] = (2, 10, 40)) -> Self:
        from ta_mov_average import TAMovAverage, MovAverageType

        assert len(time_frames) == 3, f'Market forecast has {len(time_frames)} time frames It should be 3'
        assert time_frames[0] < time_frames[1] < time_frames[2], \
            f'Market forecast time frames {time_frames} are incorrect'

        simple_mov_averages = [TAMovAverage.build(_ta_ticker, MovAverageType.simple, time_frame) for
                               time_frame in time_frames]
        market_forecast = TAMarketForecast(
            _ta_ticker.ticker,
            _ta_ticker.closes[time_frames[2] - 1:],
            simple_mov_averages[0].mov_average[time_frames[2] - time_frames[0]:],
            simple_mov_averages[1].mov_average[time_frames[2] - time_frames[1]:],
            simple_mov_averages[2].mov_average
        )
        return market_forecast

    def scatter(self, normalize: bool) -> NoReturn:
        from ta_scatter import TAScatter

        reversal_points = []
        if normalize:
            self.__normalize()
            reversal_points = self.__annotation_points()
        _data = [
            {'label': 'Momentum %', 'values': self.momentum},
            {'label': 'Near Term %', 'values': self.near_term},
            {'label': 'Intermediate %', 'values': self.intermediate},
            {'label': 'Price $', 'values': self.prices}
        ]
        ta_scatter = TAScatter(_data, f'{self.name} [{self.ticker}]', reversal_points)
        ta_scatter.visualize()

    """ -----------------  Private helper methods ------------------------  """
    def __normalize(self) -> NoReturn:
        self.momentum = TAMarketForecast.__normalize_values(self.momentum)
        self.near_term = TAMarketForecast.__normalize_values(self.near_term)
        self.intermediate = TAMarketForecast.__normalize_values(self.intermediate)

    @staticmethod
    def __normalize_values(mov_ave: np.array) -> np.array:
        min_value = np.min(mov_ave)
        max_value = np.max(mov_ave)
        delta_value = 100.0 / (max_value - min_value)
        return (mov_ave - min_value)*delta_value

    def __annotation_points(self) -> np.array:
        low_momentum_idx = np.where(self.momentum < 20.0)
        low_near_term_idx = np.where(self.near_term < 20.0)
        low_intermediate_idx = np.where(self.intermediate < 20.0)
        shared_indices = np.intersect1d(np.intersect1d(low_momentum_idx, low_near_term_idx), low_intermediate_idx)
        if len(shared_indices) == 0:
            print('No reversal points')
        return shared_indices


if __name__ == '__main__':
    import yfinance as yf
    from ta_ticker import TATicker

    data = yf.download('MO', start='2020-01-01', end='2024-09-01')
    ta_ticker = TATicker.build('WBA', data)
    ta_market_forecast = TAMarketForecast.build(ta_ticker)
    ta_market_forecast.scatter(normalize=True)
