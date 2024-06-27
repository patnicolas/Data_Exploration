import unittest
from manifolds.hyperspherepca import HyperspherePCA


class HyperspherePCATest(unittest.TestCase):

    def test_init(self):
        num_samples = 256
        pca_hypersphere = HyperspherePCA()
        self.assertEqual(num_samples, len(pca_hypersphere.sample(num_samples)))


    def test_euclidean_pca_components(self):
        num_samples = 256
        pca_hypersphere = HyperspherePCA()
        data = pca_hypersphere.sample(num_samples)
        eigenvalues, components = HyperspherePCA.euclidean_pca_components(data)
        print(f'\nPrincipal components:\n{components}\nEigen values: {eigenvalues}')
        self.assertEqual(3, len(components))

    def test_euclidean_pca_transform(self):
        num_samples = 256
        pca_hypersphere = HyperspherePCA()
        data = pca_hypersphere.sample(num_samples)
        transformed = pca_hypersphere.euclidean_pca_transform(data)
        self.assertEqual(len(transformed), num_samples)
        print(f'\nTransformed data:\n{transformed}')

    def test_tangent_pca_components(self):
        num_samples = 256
        pca_hypersphere = HyperspherePCA()
        data = pca_hypersphere.sample(num_samples)
        components = pca_hypersphere.tangent_pca_components(data)
        print(f'\nTangent PCA components:\n{components}')
        self.assertEqual(len(components), 3)

    def test_pca_vs_tangent_pca(self):
        num_samples = 256
        pca_hypersphere = HyperspherePCA()
        data = pca_hypersphere.sample(num_samples)
        _, components = HyperspherePCA.euclidean_pca_components(data)
        tangent_components = pca_hypersphere.tangent_pca_components(data)
        print(f'\nEuclidean PCA components:\n{components}\nTangent Space PCA components:\n{tangent_components}')
        self.assertEqual(len(tangent_components), 3)


if __name__ == '__main__':
    unittest.main()