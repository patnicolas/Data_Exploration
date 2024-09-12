__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import NoReturn, AnyStr, Self
from ta_ticker import TATicker


class TAAnalysis(object):
    def __init__(self, _ta_ticker: TATicker) -> None:
        self.ta_ticker = _ta_ticker

    @classmethod
    def build(cls, start_date: AnyStr, end_date: AnyStr) -> Self:
        """
        Alternative constructor for the TA analysis as sequence of predefined studies
        @param start_date: Starting date for ticker data (High, Low, close price, Volume)
        @type start_date: str
        @param end_date: End date for ticker data (High, Low, close price, Volume)
        @type end_date: str
        @return: Instance of TA analysis
        @rtype: TAAnalysis
        """
        import yfinance as yf

        ta_data = yf.download('MO', start=start_date, end=end_date)
        _ta_ticker = TATicker.build('WBA', ta_data)
        return cls(_ta_ticker)

    def scatter(self) -> NoReturn:
        from ta_market_forecast import TAMarketForecast
        from ta_macd_vol_price import TAMacdVolPrice
        from ta_macd_rsi_vol import TAMacdRsiVol
        from ta_mfi import TAMfi

        ta_market_forecast = TAMarketForecast.build(self.ta_ticker)
        annotated_data = ta_market_forecast.scatter()

        ta_macd = TAMacdVolPrice.build(self.ta_ticker)
        print(str(ta_macd))
        ta_macd.scatter(annotated_data)

        ta_macd_rsi_volume = TAMacdRsiVol.build(self.ta_ticker)
        print(str(ta_macd_rsi_volume))
        ta_macd_rsi_volume.scatter(annotated_data)

        ta_mfi = TAMfi.build(self.ta_ticker)
        print(str(ta_mfi))
        ta_mfi.scatter(annotated_data)


if __name__ == '__main__':
    import yfinance as yf

    data = yf.download('MO', start='2020-01-01', end='2024-09-01')
    ta_ticker = TATicker.build('WBA', data)
    ta_analysis = TAAnalysis(ta_ticker)
    ta_analysis.scatter()
