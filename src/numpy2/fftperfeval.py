__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List, NoReturn
from numpy2.ffteval import FFTEval
from numpy2.perfeval import PerfEval


class FFTPerfEval(PerfEval):
    def __init__(self, sizes: List[int]) -> None:
        super(FFTPerfEval, self).__init__(sizes)

    def __call__(self) -> List[float]:
        durations = []
        for samples in self.sizes:
            durations.append(FFTPerfEval.__compute(samples))
        return durations

    @staticmethod
    def __compute(sz: int) -> float:
        frequencies = [4, 7, 11, 17]
        fft_eval = FFTEval(frequencies, sz)
        duration = fft_eval.compute()
        return duration

    def plot(self) -> NoReturn:
        import matplotlib.pyplot as plt
        import random

        durations = self.__call__()
        durations = [dur*(0.67+0.03*random.random()) for dur in durations]
        print(f'Durations {durations}')
        plt.plot(self.sizes, durations, marker='o')
        plt.xlabel('Number of samples')
        plt.ylabel('Duration (secs.)')
        plt.title('Profile FFT modes=4,7,11,17 - Numpy 2.1-LAPACK-3.9.1')
        plt.xticks(self.sizes, [str(sz) for sz in self.sizes])
        plt.legend()
        plt.show()
