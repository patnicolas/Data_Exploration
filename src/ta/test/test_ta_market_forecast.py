import unittest

from ta.ta_ticker import TATicker
from ta.ta_market_forecast import TAMarketForecast


class TAMarkerForecastTest(unittest.TestCase):

    def test_init(self):
        import yfinance as yf

        ticker = 'AFL'
        data = yf.download(ticker, start='2020-01-01', end='2024-09-01')
        ta_ticker = TATicker.build(ticker, data)
        ta_market_forecast = TAMarketForecast.build(ta_ticker)
        self.assertTrue(len(ta_market_forecast.momentum) + 39 == len(ta_ticker.closes))
        print(str(ta_market_forecast))

    def test_scatter(self):
        import yfinance as yf

        ticker = 'WBA'
        data = yf.download(ticker, start='2020-01-01', end='2024-09-01')
        ta_ticker = TATicker.build(ticker, data)
        ta_market_forecast = TAMarketForecast.build(ta_ticker)
        self.assertTrue(len(ta_market_forecast.momentum) + 39 == len(ta_ticker.closes))
        ta_market_forecast.scatter()
