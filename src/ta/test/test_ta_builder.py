import unittest

from ta.ta_ticker import TATicker
from ta.ta_builder import TABuilder

class TABuilderTest(unittest.TestCase):

    def test_init(self):
        import yfinance as yf
        ticker_symbol = 'MO'
        data = yf.download(ticker_symbol, start='2020-01-01', end='2024-09-01')
        ta_ticker = TATicker.build(ticker_symbol, data)
        ta_analysis = TABuilder(ta_ticker)
        self.assertTrue(len(ta_ticker.closes) > 0)
        print(str(ta_analysis))

    def test_scatter(self):
        import yfinance as yf

        ticker_symbol = 'WBA'
        data = yf.download(ticker_symbol, start='2020-01-01', end='2024-09-01')
        ta_ticker = TATicker.build(ticker_symbol, data)
        ta_analysis = TABuilder(ta_ticker)
        self.assertTrue(len(ta_ticker.closes) > 0)
        ta_analysis.scatter()