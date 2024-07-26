import unittest

from umap.umapeval import DataSrc
from umap.tsneeval import TSneEval


class TSneEvalTest(unittest.TestCase):

    def test_mnist_3(self):
        tsne_eval = TSneEval(dataset_src=DataSrc.MNIST, n_components=3)
        tsne_eval('Spectral')

    def test_mnist_2(self):
        tsne_eval = TSneEval(dataset_src=DataSrc.MNIST, n_components=2)
        tsne_eval('Spectral')

    def test_iris_2(self):
        tsne_eval = TSneEval(dataset_src=DataSrc.IRIS, n_components=2)
        tsne_eval('Spectral')

    def test_iris_3(self):
        tsne_eval = TSneEval(dataset_src=DataSrc.IRIS, n_components=3)
        tsne_eval('Spectral')



if __name__ == '__main__':
    unittest.main()