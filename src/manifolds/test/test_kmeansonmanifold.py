import unittest
from manifolds.kmeansonmanifold import KMeansOnManifold, KMeansResult
from typing import List, AnyStr


class KMeansOnManifoldTest(unittest.TestCase):
    num_samples = 500
    num_clusters = 4
    @unittest.skip('Ignore')
    def test_init(self):
        num_samples = 100
        num_clusters = 3
        kmeans = KMeansOnManifold(num_samples, num_clusters)
        print(str(kmeans))

    def test_euclidean(self):
        kmeans = KMeansOnManifold(KMeansOnManifoldTest.num_samples, KMeansOnManifoldTest.num_clusters)
        kmeans_result = kmeans.euclidean_clustering()
        print(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters in Euclidean space --------\n{KMeansOnManifoldTest.__str(kmeans_result)}')

    def test_riemannian_von_mises_fisher(self):
        kmeans = KMeansOnManifold(
            KMeansOnManifoldTest.num_samples,
            KMeansOnManifoldTest.num_clusters,
            'random_von_mises_fisher')
        kmeans_result = kmeans.riemannian_clustering()
        print(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters with von-mises-Fisher distribution --------\n{KMeansOnManifoldTest.__str(kmeans_result)}')

    @unittest.skip('Ignore')
    def test_riemannian_random_normal(self):
        kmeans = KMeansOnManifold(
            KMeansOnManifoldTest.num_samples,
            KMeansOnManifoldTest.num_clusters,
            'random_riemann_normal')
        kmeans_result = kmeans.riemannian_clustering()
        print(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters with normal distribution  --------\n{KMeansOnManifoldTest.__str(kmeans_result)}')

    def test_riemannian_random_uniform(self):
        kmeans = KMeansOnManifold(KMeansOnManifoldTest.num_samples, KMeansOnManifoldTest.num_clusters, 'random_uniform')
        kmeans_result = kmeans.riemannian_clustering()
        print(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters with uniform distribution  --------\n{KMeansOnManifoldTest.__str(kmeans_result)}')

    def test_riemannian_constrained_random_uniform(self):
        kmeans = KMeansOnManifold(KMeansOnManifoldTest.num_samples, KMeansOnManifoldTest.num_clusters, 'constrained_random_uniform')
        kmeans_result = kmeans.riemannian_clustering()
        print(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters with constrained uniform distribution  --------\n{KMeansOnManifoldTest.__str(kmeans_result)}')

    def test_evaluate(self):
        kmeans = KMeansOnManifold(KMeansOnManifoldTest.num_samples, KMeansOnManifoldTest.num_clusters)
        print(f'Evaluate: {kmeans.evaluate()}')

    @staticmethod
    def __str(cluster_result: List[KMeansResult])-> AnyStr:
        return '\n'.join([f'Cluster #{idx + 1}\n{str(cluster_res)}' for idx, cluster_res in enumerate(cluster_result)])

if __name__ == '__main__':
    unittest.main()