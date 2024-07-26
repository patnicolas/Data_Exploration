

from enum import Enum
from sklearn.datasets import load_digits, load_iris

"""
    Enumerator for the Source of data set
"""


class DataSrc(Enum):
    MNIST = 'mnist'
    IRIS = 'iris'

class DatasetLoader(object):
    def __init__(self, dataset_src: DataSrc) -> None:
        try:
            match dataset_src:
                case DataSrc.MNIST:
                    digits = load_digits()
                    self.data = digits.data
                    self.color = digits.target.astype(int)
                case DataSrc.IRIS:
                    images = load_iris()
                    self.data = images.data
                    self.names = images.target_names
                    self.color = images.target.astype(int)
        except Exception as e:
            raise Exception(f'Failed to load {str(dataset_src)} with {str(e)}')
        self.dataset_src = dataset_src
