__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import umap
import numpy as np
from sklearn.datasets import load_digits, load_iris
from typing import NoReturn, AnyStr
from umap.datasetloader import DataSrc


class UMAPEval(object):

    def __init__(self, dataset_src: DataSrc, n_neighbors: int, min_dist: float) -> None:
        """
        Constructor for the evaluation of the Uniform Manifold Approximation and Projection
        algorithm.
        @param dataset_src: Source of the data set {i.e. IRIS)
        @type dataset_src: DataSrc
        @param n_neighbors: Number of neighbors associated with each alss
        @type n_neighbors: int
        @param min_dist: Minimum distance for UMAP
        @type min_dist: float
        """
        assert(2 <= n_neighbors <= 128, f'Number of neighbors {n_neighbors} is out of range [2, 128]')
        assert(1e-5 < min_dist < 0.2, f'Minimum distance {min_dist} is out of range')

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
        # Instantiate the UMAP model
        self.umap = umap.UMAP(random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)

    def __call__(self, cmap: AnyStr) -> NoReturn:
        """
        Visualize the UMAP data cluster given a cmap, minimum distance and estimated number of neighbors
        @param cmap: Color theme or map identifier used in display
        @type cmap: str
        """
        import matplotlib.pyplot as plt

        embedding = self.umap.fit_transform(self.data)
        x = embedding[:, 0]
        y = embedding[:, 1]
        n_ticks = 10
        plt.scatter(x=x, y=y, c=self.color, cmap=cmap, s=4.0)
        plt.colorbar(boundaries=np.arange(n_ticks+1) - 0.5).set_ticks(np.arange(n_ticks))
        plt.title(f'UMAP {self.dataset_src} {self.umap.n_neighbors} neighbors, min_dist: {self.umap.min_dist}')
        plt.show()