import unittest
from src.manifolds.kmeansonmanifold import KMeansOnManifold, KMeansCluster
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

    @unittest.skip('Ignore')
    def test_euclidean(self):
        kmeans = KMeansOnManifold(KMeansOnManifoldTest.num_samples, KMeansOnManifoldTest.num_clusters)
        kmeans_cluster = kmeans.euclidean_clustering()
        print(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters in Euclidean space --------\n{KMeansOnManifoldTest.__str(kmeans_cluster)}')

    def test_euclidean_2(self):
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        import numpy as np
        from geomstats.geometry.hypersphere import Hypersphere

        # Generate num_samples data points on a Hypersphere
        # (uniform random generation)
        n_clusters = 4
        num_samples = 256
        hypersphere = Hypersphere(dim=2, equip=True)
        clusters = hypersphere.random_uniform(n_samples=num_samples)

        # Uses Sk-learn to create cluster on the Euclidean space
        kmeans = KMeans(n_clusters=n_clusters,
                        init='k-means++',
                        algorithm='elkan',
                        max_iter=140)
        data = np.concatenate(clusters, axis=0).reshape(256, 3)
        kmeans.fit(data)

        # Display the data points on Hypersphere manifold in
        # 3D Euclidean space
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs=data[:, 0],
                   ys=data[:, 1],
                   zs=data[:, 2],
                   c=data[:, 2],
                   cmap='viridis',
                   marker='o')

        # Plot cluster centers
        ax.scatter(xs=kmeans.cluster_centers_[:, 0],
                   ys=kmeans.cluster_centers_[:, 1],
                   zs=kmeans.cluster_centers_[:, 2],
                   c='red',
                   s=260,
                   marker='X')

        plt.title("k-Means Clustering Visualization Euclidean 3D")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.ylabel("Z")
        plt.legend()
        plt.grid(True)
        plt.show()


    @unittest.skip('Ignore')
    def test_riemannian_von_mises_fisher(self):
        kmeans = KMeansOnManifold(
            KMeansOnManifoldTest.num_samples,
            KMeansOnManifoldTest.num_clusters,
            'random_von_mises_fisher')
        kmeans_cluster = kmeans.riemannian_clustering()
        print(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters with von-mises-Fisher distribution --------\n{KMeansOnManifoldTest.__str(kmeans_cluster)}')

    @unittest.skip('Ignore')
    def test_riemannian_von_mises_fisher_2(self):
        num_samples = 1048
        kmeans = KMeansOnManifold(
            num_samples,
            KMeansOnManifoldTest.num_clusters,
            'random_von_mises_fisher')
        kmeans_cluster = kmeans.riemannian_clustering()
        print(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters with von-mises-Fisher distribution --------\n{KMeansOnManifoldTest.__str(kmeans_cluster)}'
              f'{num_samples} samples')

    @unittest.skip('Ignore')
    def test_riemannian_random_normal(self):
        kmeans = KMeansOnManifold(
            KMeansOnManifoldTest.num_samples,
            KMeansOnManifoldTest.num_clusters,
            'random_riemann_normal')
        kmeans_cluster = kmeans.riemannian_clustering()
        print(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters with normal distribution  --------\n{KMeansOnManifoldTest.__str(kmeans_cluster)}')

    @unittest.skip('Ignore')
    def test_riemannian_random_uniform(self):
        kmeans = KMeansOnManifold(KMeansOnManifoldTest.num_samples, KMeansOnManifoldTest.num_clusters, 'random_uniform')
        kmeans_clusters = kmeans.riemannian_clustering()
        print(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters with uniform distribution  --------\n{KMeansOnManifoldTest.__str(kmeans_clusters)}')

    @unittest.skip('Ignore')
    def test_riemannian_constrained_random_uniform(self):
        kmeans = KMeansOnManifold(KMeansOnManifoldTest.num_samples, KMeansOnManifoldTest.num_clusters, 'constrained_random_uniform')
        kmeans_clusters = kmeans.riemannian_clustering()
        print(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters with constrained uniform distribution  --------\n{KMeansOnManifoldTest.__str(kmeans_clusters)}')

    @unittest.skip('Ignore')
    def test_evaluate(self):
        kmeans = KMeansOnManifold(KMeansOnManifoldTest.num_samples, KMeansOnManifoldTest.num_clusters)
        print(f'Evaluate: {kmeans.evaluate()}')

    @staticmethod
    def __str(cluster_result: List[KMeansCluster])-> AnyStr:
        return '\n'.join([f'Cluster #{idx + 1}\n{str(cluster_res)}' for idx, cluster_res in enumerate(cluster_result)])


if __name__ == '__main__':
    unittest.main()