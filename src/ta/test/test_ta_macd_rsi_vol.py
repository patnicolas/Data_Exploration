import unittest
from ta.ta_ticker import TATicker
from ta.ta_market_forecast import TAMarketForecast
from ta.ta_mfi import TAMfi


class TAMacdRsiVolTest(unittest.TestCase):
    def test_scatter(self):
        import yfinance as yf

        data = yf.download('MO', start='2020-01-01', end='2024-09-01')
        ta_ticker = TATicker.build('WBA', data)

        ta_market_forecast = TAMarketForecast.build(ta_ticker)
        annotated_data = ta_market_forecast.scatter()

        ta_mfi = TAMfi.build(ta_ticker)
        print(str(ta_mfi))
        ta_mfi.scatter(annotated_data)