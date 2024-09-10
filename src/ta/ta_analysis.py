__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List
from tamarketforecast import TAMarketForecast
from taticker import TATicker
from tamacd import TAMACD
from tamovaverage import TAMovAverage, MovAverageType


class TAAnalysis(object):
    def __init__(self, _ta_ticker: TATicker):
        self.ta_ticker = _ta_ticker

    def moving_average(self, mov_ave_type: MovAverageType, window_size: int) -> TAMovAverage:
        return TAMovAverage.build(self.ta_ticker, mov_ave_type, window_size)

    def market_forecast(self, time_frames: List[int] = (2, 10, 40)) -> TAMarketForecast:
        assert len(time_frames) == 3, f'Market forecast has {len(time_frames)} time frames It should be 3'
        assert time_frames[0] < time_frames[1] < time_frames[2], \
            f'Market forecast time frames {time_frames} are incorrect'

        #   def build(cls, mov_average_type: MovAverageType, window_size: int, values: np.array)  -> Self:
        simple_mov_averages = [TAMovAverage.build(self.ta_ticker, MovAverageType.simple, time_frame)
                               for time_frame in time_frames]
        market_forecast = TAMarketForecast(
            self.ta_ticker.ticker,
            self.ta_ticker.closes[time_frames[2] - 1:],
            simple_mov_averages[0].mov_average[time_frames[2] - time_frames[0]:],
            simple_mov_averages[1].mov_average[time_frames[2] - time_frames[1]:],
            simple_mov_averages[1].mov_average
        )
        return market_forecast

    def macd(self) -> TAMACD:
        return TAMACD.build(self.ta_ticker)



if __name__ == '__main__':
    import yfinance as yf

    data = yf.download('MO', start='2020-01-01', end='2024-09-01')
    ta_ticker = TATicker.build('WBA', data)
    ta_analysis = TAAnalysis(ta_ticker)
    ta_market_forecast = ta_analysis.market_forecast()
    ta_market_forecast.scatter(normalize=True)
