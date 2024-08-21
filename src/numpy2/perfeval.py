__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List


def timeit(func):
    import time

    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        return duration
    return wrapper


class PerfEval(object):
    def __init__(self, sizes: List[int]) -> None:
        self.sizes = sizes
