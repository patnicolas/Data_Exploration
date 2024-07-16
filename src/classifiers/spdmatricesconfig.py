__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from dataclasses import dataclass
from typing import AnyStr


@dataclass
class SPDMatricesConfig:
    n_spd_matrices: int
    n_channels: int
    evals_lows_1: int
    evals_lows_2: int
    class_sep_ratio_1: float
    class_sep_ratio_2: float

    def __str__(self) -> AnyStr:
        return f'\nNum matrices: {self.n_spd_matrices}\nNum channels: {self.n_channels}\n' \
               f'Eval low 1: {self.evals_lows_1}\nEval low 2: {self.evals_lows_2}\n' \
               f'Class separation ratio 1: {self.class_sep_ratio_1}\nClass separation ratio 2: {self.class_sep_ratio_2}'