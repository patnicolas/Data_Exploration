__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import numpy as np
from typing import Self, List, AnyStr

SIGMOID: AnyStr = 'sigmoid'
ADD: AnyStr = 'add'
MUL: AnyStr = 'mul'
DOT: AnyStr = 'dot'


class LinearAlgebraEval(object):
    def __init__(self, x: np.array, shape: List[int]) -> None:
        self.x = x.reshape(shape)

    @classmethod
    def build(cls, size: int) -> Self:
        _x = np.random.uniform(0.0, 100.0, size)
        return cls(_x, [10, 100, -1])

    def __str__(self) -> AnyStr:
        return str(self.x)

    def scalar_mul(self, a: float) -> np.array:
        return self.x * a

    def sigmoid(self, a: float) -> np.array:
        return 1.0/(1.0+ np.exp(-self.x * a))

    def add(self, y: np.array) -> np.array:
        assert len(self.x) == len(y), f'x and y have different dimension'
        return self.x + y.reshape(self.x.shape)

    def mul(self, y: np.array) -> np.array:
        assert len(self.x) == len(y), f'x and y have different dimension'
        z = 2.0* self.x * y.reshape(self.x.shape)
        return z * z

    def dot(self, y: np.array) -> np.array:
        assert len(self.x) == len(y), f'x and y have different dimension'
        return np.dot(self.x.reshape(-1), y.reshape(-1))


