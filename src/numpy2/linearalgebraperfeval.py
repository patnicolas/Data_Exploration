__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import numpy as np
from typing import List, AnyStr, Dict
import matplotlib.pyplot as plt

from numpy2.linearalgebraeval import LinearAlgebraEval, SIGMOID, ADD, MUL, DOT
from numpy2.perfeval import PerfEval
from numpy2.perfeval import timeit

class LinearAlgebraPerfEval(PerfEval):
    def __init__(self, sizes: List[int]) -> None:
        super(LinearAlgebraPerfEval, self).__init__(sizes)

    def __call__(self) -> Dict[AnyStr, List[float]]:
        _perf = {}
        lin_algebra = LinearAlgebraEval.build(self.sizes[0])
        _perf[SIGMOID] = [LinearAlgebraPerfEval.__performance(lin_algebra, SIGMOID)]
        _perf[ADD] = [LinearAlgebraPerfEval.__performance(lin_algebra, ADD)]
        _perf[MUL] = [LinearAlgebraPerfEval.__performance(lin_algebra, MUL)]
        _perf[DOT] = [ LinearAlgebraPerfEval.__performance(lin_algebra, DOT)]

        for index in range(1, len(self.sizes)):
            # Alternative constructor
            lin_algebra = LinearAlgebraEval.build(self.sizes[index])
            _perf = LinearAlgebraPerfEval.__record(_perf, lin_algebra, SIGMOID)
            _perf = LinearAlgebraPerfEval.__record(_perf, lin_algebra, ADD)
            _perf = LinearAlgebraPerfEval.__record(_perf, lin_algebra, MUL)
            _perf = LinearAlgebraPerfEval.__record(_perf, lin_algebra, DOT)
        return _perf

    @staticmethod
    def __record(perf: Dict[AnyStr, List[float]],
                 lin_algebra: LinearAlgebraEval,
                 lin_algebra_op: AnyStr) -> Dict[AnyStr, List[float]]:
        duration = LinearAlgebraPerfEval.__performance(lin_algebra, lin_algebra_op)
        lst = perf[lin_algebra_op]
        lst.append(duration)
        perf[lin_algebra_op] = lst
        return perf

    @timeit
    @staticmethod
    def __performance(lin_algebra: LinearAlgebraEval, op: AnyStr) -> np.array:
        match op:
            case "sigmoid":
                return lin_algebra.sigmoid(8.5)
            case "add":
                return lin_algebra.add(lin_algebra.x)
            case "mul":
                return lin_algebra.mul(lin_algebra.x)
            case "dot":
                return lin_algebra.dot(lin_algebra.x)

    def plot(self):
        performance = self()

        plt.plot(self.sizes, performance[SIGMOID], label=SIGMOID, marker='o')
        plt.plot(self.sizes, performance[ADD], label='x + y', marker='x')
        plt.plot(self.sizes, performance[MUL], label='(x*y)^2', marker='s')
        plt.plot(self.sizes, performance[DOT], label='Dot x.y', marker='^')
        plt.xlabel('Array sizes (10, 100, -1)')
        plt.ylabel('Duration (secs.)')
        plt.title('Profile linear algebra Numpy 2.1 - ILP64/LAPACK v3.9.1')
        plt.xticks(self.sizes, [str(sz) for sz in self.sizes])
        plt.legend()
        plt.show()

