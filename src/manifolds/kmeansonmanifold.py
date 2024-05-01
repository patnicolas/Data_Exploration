__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from typing import AnyStr, List
import numpy as np
from dataclasses import dataclass
import geomstats.backend as gs

@dataclass
class KMeansResult:
    center: np.array
    label: np.array
    predicted: np.array

    def __str__(self) -> AnyStr:
        return f'Center: {self.center}, Label: {self.label}, Predicted: {self.predicted}'


class KMeansOnManifold(object):
    def __init__(self, num_samples: int, num_clusters: int, random_gen: AnyStr = 'random_von_mises_fisher'):
        """
        Constructor for the evaluation of KMeans on Euclidean space and Riemann manifold (Hypersphere)
        :param num_samples: Number of random samples
        :type num_samples: int
        :param num_clusters: Number of clusters used in KMeans
        :type num_clusters: int
        :param random_gen: Random generator identifier on the hypersphere
        :type random_gen: AnyStr
        """
        self.hypersphere = Hypersphere(dim=2, equip=True)
        match random_gen:
            case 'random_von_mises_fisher':
                cluster = self.hypersphere.random_von_mises_fisher(kappa=20, n_samples=num_samples)
            case 'random_rand_riemann_normal':
                cluster = self.hypersphere.random_riemannian_normal(n_samples=num_samples)
            case 'random_uniform':
                cluster = self.hypersphere.random_uniform(n_samples=num_samples)
            case _:
                cluster = self.hypersphere.random_uniform(n_samples=num_samples)
        # Use the symmetric rotation Lie group in dimension 3
        so3_lie_group = SpecialOrthogonal(3, equip=False)
        # Generate the clusters
        self.clusters = [cluster @ so3_lie_group.random_uniform() for _ in range(num_clusters)]

    def __str__(self) -> AnyStr:
        return '\n'.join([f'Cluster: {idx+1}\n{str(cluster)}' for idx, cluster in enumerate(self.clusters)])

    def euclidean_clustering(self) -> List[KMeansResult]:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=len(self.clusters))
        data = np.concatenate(self.clusters, axis=0)
        kmeans.fit(data)
        predictions = kmeans.predict(data)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        return [KMeansResult(center, label, prediction)
                for center, label, prediction in zip(centers, labels, predictions)]

    def riemannian_clustering(self) -> List[KMeansResult]:
        from geomstats.learning.kmeans import RiemannianKMeans

        kmeans = RiemannianKMeans(space=self.hypersphere, n_clusters=len(self.clusters))
        data = gs.concatenate(self.clusters, axis =0)
        kmeans.fit(data)
        centers = kmeans.centroids_
        labels = kmeans.labels_
        predictions = kmeans.predict(data)
        return [KMeansResult(center, label, prediction)
                for center, label, prediction in zip(centers, labels, predictions)]
