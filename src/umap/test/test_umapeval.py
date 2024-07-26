import unittest
from umap.umapeval import UMAPEval
from umap.umapeval import DataSrc


class UMAPEvalTest(unittest.TestCase):
    @unittest.skip
    def test_mnist(self):
        n_neighbors = 4
        min_dist = 0.8
        umap_eval = UMAPEval(dataset_src=DataSrc.MNIST, n_neighbors=n_neighbors, min_dist=min_dist)
        succeeded = umap_eval(cmap='Spectral')
        self.assertTrue(succeeded)

    def test_iris(self):
        n_neighbors = 40
        min_dist = 0.001
        umap_eval = UMAPEval(dataset_src=DataSrc.IRIS,n_neighbors=n_neighbors, min_dist=min_dist)
        succeeded = umap_eval(cmap='Set1')
        self.assertTrue(succeeded)


if __name__ == '__main__':
    unittest.main()
