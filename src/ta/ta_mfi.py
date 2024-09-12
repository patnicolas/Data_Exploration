__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from ta_study import TAStudy
from ta_ticker import TATicker
from typing import AnyStr, Self, NoReturn
import numpy as np

"""
Implementation of the computation of Money Flow Index (MFI)
    typical_price = (High + Low + Close)/3
    money_flow = typical_price * volume
    positive_money_flow = money_flow[i) if money_flow[i] > money_flow[i-1])
    negative_money_flow = money_flow[i)) if money_flow[i] < money_flow[i-1])
    money_flow_ratio = positive_money_flow/negative_money_flow
    money_flow_index = 100*(1 - 1/(1 + money_flow_ratio)
"""


class TAMfi(TAStudy):
    window: int = 14
    def __init__(self,
                 ticker: AnyStr,
                 prices: np.array,
                 mfis: np.array,
                 volumes: np.array) -> None:
        """
        Constructor for the Money Flow Index
        @param ticker: Ticker symbol
        @type ticker: str
        @param prices: Ticker price at close
        @type prices: Numpy array
        @param mfis: Money flow indices
        @type mfis: Numpy array
        @param volumes: Trading volume for the day or period
        @type volumes: Numpy array
        """
        super(TAMfi, self).__init__('MFI - Price - Volume', prices)
        self.ticker = ticker
        self.mfis = mfis
        self.volumes = volumes

    @classmethod
    def build(cls, _ta_ticker: TATicker) -> Self:
        """
        Alternative constructor using the fully defined TA ticker data
        @param _ta_ticker: Ticker instance containing ticker symbole, volume, high, low and closing prices
        @type _ta_ticker: TATicker class
        @return: Instance of this TA Mif
        @rtype: TAMfi
        """
        money_flow = [float(0.3333*(h+l+p)*v) for h, l, p, v in
                      zip(_ta_ticker.highs, _ta_ticker.lows, _ta_ticker.closes, _ta_ticker.volumes)]

        pos_money_flows = [money_flow[0]]
        neg_money_flows = [money_flow[0]]
        for idx, mf in enumerate(money_flow[1:]):
            if mf >= money_flow[idx]:
                pos_money_flows.append(mf)
                neg_money_flows.append(neg_money_flows[idx])
            else:
                neg_money_flows.append(mf)
                pos_money_flows.append(pos_money_flows[idx])

        money_flow_ratios = []
        for idx in range(len(money_flow) - TAMfi.window):
            idx_end = idx + TAMfi.window
            money_flow_ratios.append(sum(pos_money_flows[idx:idx_end])/sum(neg_money_flows[idx:idx_end]))

        money_flow_indices = [100.0*(1 - 1/(1+mf)) for mf in money_flow_ratios]
        return cls(
            ticker=_ta_ticker.ticker,
            prices=_ta_ticker.closes,
            mfis=money_flow_indices,
            volumes=_ta_ticker.volumes)

    def __str__(self) -> AnyStr:
        return f'\nTicker: {self.ticker}\nMFIs:\n{self.mfis}\nVolume:\n{self.volumes}'

    def scatter(self, _annotated_data: np.array = None) -> np.array:
        """
        Scatter plot for this study with data point annotated by previous studies
        @param _annotated_data: Data point selected from previous studies, None if none were selected
        @type _annotated_data: Numpy Array
        @return: Newly annotated data point if any, None otherwise
        @rtype: Numpy array
        """
        if _annotated_data is None:
            _annotated_data = []
        from ta_scatter import TAScatter

        _data = [
            {'label': 'MFI', 'values': self.mfis},
            {'label': 'Volume', 'values': self.volumes[TAMfi.window:]},
            {'label': 'Prices $', 'values': self.prices[TAMfi.window:]},
        ]
        ta_scatter = TAScatter(_data, f'{self.name} [{self.ticker}]', _annotated_data)
        ta_scatter.visualize()
        return _annotated_data


if __name__ == '__main__':
    import yfinance as yf

    data = yf.download('MO', start='2020-01-01', end='2024-09-01')
    ta_ticker = TATicker.build('WBA', data)
    ta_mfi = TAMfi.build(ta_ticker)
    print(ta_mfi)
    ta_mfi.scatter()