import unittest
from src.manifolds.hyperspherepca import HyperspherePCA
import numpy as np

from src.manifolds.hyperspherespace import HypersphereSpace


class HyperspherePCATest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init(self):
        num_samples = 256
        pca_hypersphere = HyperspherePCA()
        self.assertEqual(num_samples, len(pca_hypersphere.sample(num_samples)))

    @unittest.skip('Ignore')
    def test_euclidean_pca_components(self):
        num_samples = 256
        pca_hypersphere = HyperspherePCA()
        data = pca_hypersphere.sample(num_samples)
        eigenvalues, components = HyperspherePCA.euclidean_pca_components(data)
        print(f'\nPrincipal components:\n{components}\nEigen values: {eigenvalues}')
        self.assertEqual(3, len(components))

    @unittest.skip('Ignore')
    def test_euclidean_pca_transform(self):
        num_samples = 256
        pca_hypersphere = HyperspherePCA()
        data = pca_hypersphere.sample(num_samples)
        transformed = pca_hypersphere.euclidean_pca_transform(data)
        self.assertEqual(len(transformed), num_samples)
        print(f'\nTransformed data:\n{transformed}')

    @unittest.skip('Ignore')
    def test_tangent_pca_components(self):
        num_samples = 256
        pca_hypersphere = HyperspherePCA()
        data = pca_hypersphere.sample(num_samples)
        components = pca_hypersphere.tangent_pca_components(data)
        print(f'\nTangent PCA components:\n{components}')
        self.assertEqual(len(components), 3)

    def test_pca_vs_tangent_pca_1(self):
        num_samples = 1024
        components, tangent_components = HyperspherePCATest.pca_and_tangent_pca(num_samples)
        print(f'\nEuclidean PCA components:\n{components}\nTangent Space PCA components:\n{tangent_components}')
        self.assertEqual(len(tangent_components), 3)

    def test_pca_vs_tangent_pca_2(self):
        num_samples = 64
        components, tangent_components = HyperspherePCATest.pca_and_tangent_pca(num_samples)
        print(f'\nEuclidean PCA components:\n{components}\nTangent Space PCA components:\n{tangent_components}')
        self.assertEqual(len(tangent_components), 3)


    @staticmethod
    def pca_and_tangent_pca(num_samples: int) -> (np.array, np.array):
        pca_hypersphere = HyperspherePCA()
        data = pca_hypersphere.sample(num_samples)
        _, components = HyperspherePCA.euclidean_pca_components(data)
        tangent_components = pca_hypersphere.tangent_pca_components(data)
        return components, tangent_components


if __name__ == '__main__':
    unittest.main()