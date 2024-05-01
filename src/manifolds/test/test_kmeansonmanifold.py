import unittest
from manifolds.kmeansonmanifold import KMeansOnManifold


class KMeansOnManifoldTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init(self):
        num_samples = 100
        num_clusters = 3
        kmeans = KMeansOnManifold(num_samples, num_clusters)
        print(str(kmeans))

    def test_euclidean(self):
        num_samples = 100
        num_clusters = 3
        kmeans = KMeansOnManifold(num_samples, num_clusters)
        kmeans_result = kmeans.euclidean_clustering()
        print('\nEuclidean --------')
        print('\n'.join([f'Cluster #{idx+1}\n{str(cluster_result)}' for idx, cluster_result in enumerate(kmeans_result)]))

    def test_riemannian(self):
        num_samples = 100
        num_clusters = 3
        kmeans = KMeansOnManifold(num_samples, num_clusters)
        kmeans_result = kmeans.riemannian_clustering()
        print('\nRiemannian --------')
        print('\n'.join([f'Cluster #{idx+1}\n{str(cluster_result)}' for idx, cluster_result in enumerate(kmeans_result)]))


if __name__ == '__main__':
    unittest.main()