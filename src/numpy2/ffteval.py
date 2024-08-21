__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import numpy as np
from typing import List, NoReturn
from numpy2.linearalgebraperfeval import timeit


class FFTEval(object):
    def __init__(self, frequencies: List[int], samples: int) -> None:
        self.F = 1.0/samples
        self.x = np.arange(0, 1.0, self.F)
        pi_2 = 2*np.pi
        self.signal = np.sin(pi_2*frequencies[0]*self.x)
        if len(frequencies) > 1:
            for f in frequencies[1:]:
                self.signal += np.sin(pi_2*f*self.x)

    @timeit
    def compute(self) -> (np.array, np.array):
        num_samples = len(self.signal)
        num_half_samples = num_samples//2
        freqs = np.fft.fftfreq(num_samples, self.F)
        positive_freq = freqs[:num_half_samples]
        amplitude = np.abs(np.fft.fft(self.signal)[:num_half_samples]) / num_samples
        return positive_freq, amplitude

    def plot(self) -> NoReturn:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.x, self.signal)
        plt.title(f'Signal')
        plt.xlabel('Time')
        plt.ylabel('Value')

        freqs, amplitudes = self.compute()
        plt.subplot(2, 1, 2)
        plt.stem(freqs, amplitudes, 'b', marketfmt=' ', basefmt='-b')
        plt.title('Frequencies Spectrum')
        plt.xlabel('Frequency')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.show()