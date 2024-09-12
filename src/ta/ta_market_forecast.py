__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, NoReturn, Self, List
import numpy as np
from ta_study import TAStudy
from ta_ticker import TATicker


class TAMarketForecast(TAStudy):
    def __init__(self,
                 ticker: AnyStr,
                 prices: np.array,
                 momentum: np.array,
                 near_term: np.array,
                 intermediate: np.array,
                 normalize: bool = True) -> None:
        super(TAMarketForecast, self).__init__('Market Forecast', prices)
        self.name = self.name + ' - Price'
        self.ticker = ticker
        self.momentum = momentum
        self.near_term = near_term
        self.intermediate = intermediate
        self.normalize = normalize

    @classmethod
    def build(cls, _ta_ticker: TATicker, time_frames: List[int] = (2, 10, 40)) -> Self:
        """
        Alternative constructor using the fully defined TA ticker data
        @param _ta_ticker: Ticker instance containing ticker symbole, volume, high, low and closing prices
        @type _ta_ticker: TATicker class
        @return: Instance of this TAMarketForecast
        @type time_frames: The 3 time frames for the Market forecast moving average
        @rtype: List[int]
        """
        from ta_mov_average import TAMovAverage, MovAverageType

        assert len(time_frames) == 3, f'Market forecast has {len(time_frames)} time frames It should be 3'
        assert time_frames[0] < time_frames[1] < time_frames[2], \
            f'Market forecast time frames {time_frames} are incorrect'

        simple_mov_averages = [TAMovAverage.build(_ta_ticker, MovAverageType.simple, time_frame) for
                               time_frame in time_frames]
        market_forecast = TAMarketForecast(
            ticker=_ta_ticker.ticker,
            prices=_ta_ticker.closes[time_frames[2] - 1:],
            momentum=simple_mov_averages[0].mov_average[time_frames[2] - time_frames[0]:],
            near_term=simple_mov_averages[1].mov_average[time_frames[2] - time_frames[1]:],
            intermediate=simple_mov_averages[2].mov_average,
            normalize=True
        )
        return market_forecast

    def scatter(self, _annotated_data: np.array = None) -> np.array:
        """
        Scatter plot for this study with data point annotated by previous studies
        @param _annotated_data: Data point selected from previous studies, None if none were selected
        @type _annotated_data: Numpy Array
        @return: Newly annotated data point if any, None otherwise
        @rtype: Numpy array
        """
        from ta_scatter import TAScatter

        annotated_data = [] if _annotated_data is None else _annotated_data
        if self.normalize:
            self.__normalize()
            annotated_data = self.__annotation_points()
        _data = [
            {'label': 'Momentum %', 'values': self.momentum},
            {'label': 'Near Term %', 'values': self.near_term},
            {'label': 'Intermediate %', 'values': self.intermediate},
            {'label': 'Price $', 'values': self.prices}
        ]
        ta_scatter = TAScatter(_data, f'{self.name} [{self.ticker}]', annotated_data)
        ta_scatter.visualize()
        return annotated_data

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
