__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import umap
import numpy as np
from typing import NoReturn, AnyStr
from umap.datasetloader import DataSrc
from umap.datasetloader import DatasetLoader

"""
Class that wraps the evaluation of the Uniform Manifold Approximation and Projection (UMAP) algorithm.
It relies on umap-learn Python module 
Versions: Python  3.11,  SciKit-learn 1.5.1,  Matplotlib 3.9.1, umap-learn 0.5.6
The dunda method __call__ is used to visualize the umap generated clusters in 2 dimension
"""


class UMAPEval(DatasetLoader):

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

        # Instantiate the selected data set
        super(UMAPEval, self).__init__(dataset_src)
        # Instantiate the UMAP model
        self.umap = umap.UMAP(random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)

    def __call__(self, cmap: AnyStr) -> bool:
        """
        Visualize the UMAP data cluster given a cmap, minimum distance and estimated number of neighbors
        @param cmap: Color theme or map identifier used in display
        @type cmap: str
        """
        import matplotlib.pyplot as plt
        try:
            embedding = self.umap.fit_transform(self.data)
            x = embedding[:, 0]
            y = embedding[:, 1]
            n_ticks = 10
            plt.scatter(x=x, y=y, c=self.color, cmap=cmap, s=4.0)
            plt.colorbar(boundaries=np.arange(n_ticks+1) - 0.5).set_ticks(np.arange(n_ticks))
            plt.title(f'UMAP {self.dataset_src} {self.umap.n_neighbors} neighbors, min_dist: {self.umap.min_dist}')
            plt.show()
            return True

        except Exception as e:
            print(f'UMAP failed to visualize for {self.dataset_src}:  {str(e)}')
            return False
