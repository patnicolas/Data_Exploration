__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from sklearn.manifold import TSNE
from typing import NoReturn, AnyStr
from umap.umapeval import DataSrc
from umap.datasetloader import DatasetLoader


class TSneEval(DatasetLoader):
    def __init__(self, dataset_src: DataSrc, n_components: int) -> None:
        """
        Constructor for the evaluation of the t-distributed Stochastic Neighbor Embedding. An assert error is
        thrown if the number of components is not 2 or 3
        @param dataset_src: Source for the dataset (MNIST, IRIS,...)
        @type dataset_src: DataSrc
        @param n_components: Dimension of the tSNE analysis
        @type n_components: int
        """
        assert (n_components > 3 or n_components < 2, f'Number of components {n_components} is out of range')

        super(TSneEval, self).__init__(dataset_src)
        # Instantiate the Sk-learn tSNE model
        self.t_sne = TSNE(n_components=n_components)


    def __call__(self, cmap: AnyStr) -> NoReturn:
        """
        Visualize the tSNE data cluster in 2 or 3 dimension given a cmap
        @param cmap: Color theme or map identifier used in display
        @type cmap: str
        """
        import matplotlib.pyplot as plt

        embedded = self.t_sne.fit_transform(self.data)
        fig = plt.figure()
        if self.t_sne.n_components == 2:
            plt.scatter(embedded[:, 0], embedded[:, 1], c=self.color, cmap=cmap)
            plt.colorbar()
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*zip(*embedded[:, :2]), c=self.color, cmap=cmap)
        plt.title(f'tSNE {str(self.dataset_src)} {self.t_sne.n_components} components')
        plt.show()

