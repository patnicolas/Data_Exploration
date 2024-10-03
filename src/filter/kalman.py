__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import numpy as np
from typing import Self, NoReturn, Callable, List, Tuple, AnyStr


class KalmanFilter(object):

    def __init__(self,
                 _x0: np.array,
                 _P0: np.array,
                 _A: np.array,
                 _H: np.array,
                 Q: np.array,
                 R: np.array,
                 u0: np.array = None,
                 B: np.array = None) -> None:
        self.x = _x0
        self.P = _P0
        self.A = _A
        self.H = _H
        self.Q = Q
        self.R = R
        self.u0 = u0
        self.B = B

    @classmethod
    def build(cls, _x0: np.array, _P0: np.array, _A: np.array, _H: np.array, qr: (float, float)) -> Self:
        dim = len(_x0)
        Q = np.eye(dim)*qr[0]
        R = np.eye(1)*qr[1]
        return cls(_x0, _P0, _A, _H, Q, R)

    def __str__(self) -> AnyStr:
        return f'\nA state transition:\n{self.A}\nH observations:\n{self.H}\nP covariance:{self.P}\nQ:\n{self.Q}\nR:\n{self.R}\nx state:\n{self.x}'

    def predict(self, v: np.array) -> NoReturn:
        # State:  x[n] = A.x~[n-1] + B.u[n-1] + v
        self.x = self.A @ self.x + v if self.B is None else  self.A @ self.x + self.B @ self.u0 + v
        # Error covariance:  P[n] = A[n].P[n-1].A[n]^T + Q[n]
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z: np.array) -> NoReturn:
        # Innovation:  S[n] = H.P[n-1].H^T + R[n]
        S = self.H @ self.P @ self.H.T + self.R
        # Gain: G[n] = P[n-1].H^T/S[n]
        G = self.P @ self.H.T @ np.linalg.inv(S)
        # State estimate y[n] = z[n] - H.x
        y = z - self.H @ self.x
        self.x = self.x + G @ y
        g = np.eye(self.P.shape[0]) - G @ self.H
        self.P = g @ self.P

    def simulate(self,
                 num_measurements: int,
                 measure: Callable[[int], np.array],
                 cov_means: np.array) -> List[Tuple[np.array]]:
        return [self.__estimate_next_state(i, measure, cov_means) for i in range(num_measurements)]

    """ -------------------------------------   Private supporting methods ------------------- """
    def __estimate_next_state(self, state_index: int, measure: Callable[[int], np.array], noise: np.array) -> (np.array, np.array):
        z = measure(state_index)
        self.predict(noise)
        self.update(z)
        return z, self.x

