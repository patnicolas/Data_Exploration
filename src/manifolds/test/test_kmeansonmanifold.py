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
        kmeans_result = kmeans.euclidean()
        print('\n'.join([f'Cluster: {idx}\n{str(cluster_result)}' for idx, cluster_result in enumerate(kmeans_result)]))


if __name__ == '__main__':
    unittest.main()