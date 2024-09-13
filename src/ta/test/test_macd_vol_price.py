import unittest

from ta.ta_ticker import TATicker
from ta.ta_macd_vol_price import TAMacdVolPrice


class TAMacdVolPriceTest(unittest.TestCase):
    def test_scatter(self):
        import yfinance as yf

        ticker = 'QQQ'
        data = yf.download(ticker, start='2020-01-01', end='2024-09-01')
        ta_ticker = TATicker.build(ticker, data)
        ta_macd = TAMacdVolPrice.build(ta_ticker)
        print(str(ta_macd))
        ta_macd.scatter()