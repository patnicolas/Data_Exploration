import unittest

from ta.ta_ticker import TATicker
from ta.ta_mfi import TAMfi


class TAMfiTest(unittest.TestCase):
    def test_scatter(self):
        import yfinance as yf

        ticker = 'MO'
        data = yf.download(ticker, start='2020-01-01', end='2024-09-01')
        ta_ticker = TATicker.build(ticker, data)
        ta_mfi = TAMfi.build(ta_ticker)
        n = len(ta_mfi.prices)+TAMfi.window
        self.assertTrue(len(ta_mfi.prices)-TAMfi.window == len(ta_mfi.mfis))
        print(str(ta_mfi))
        ta_mfi.scatter()
