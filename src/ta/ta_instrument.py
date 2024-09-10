__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr
import numpy as np


class TAInstrument(object):
    def __init__(self, name: AnyStr, prices: np.array) -> None:
        self.name = name
        self.prices = prices

