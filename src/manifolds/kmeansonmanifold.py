__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from typing import AnyStr, List
from sklearn.cluster import KMeans
import numpy as np
from dataclasses import dataclass

@dataclass
class KMeansResult:
    center: np.array
    predicted: np.array

    def __str__(self) -> AnyStr:
        return f'Center: {self.center}\nPredicted: {self.predicted}'


class KMeansOnManifold(object):
    def __init__(self, num_samples: int, num_clusters: int):
        hypersphere = Hypersphere(dim=2, equip=False)
        cluster = hypersphere.random_von_mises_fisher(kappa=20, n_samples=num_samples)
        so3_lie = SpecialOrthogonal(3, equip=False)
        self.clusters = [cluster @ so3_lie.random_uniform() for _ in range(num_clusters)]
        self.data = np.concatenate(self.clusters)

    def __str__(self) -> AnyStr:
        return '\n'.join([f'Cluster: {idx+1}\n{str(cluster)}' for idx, cluster in enumerate(self.clusters)])

    def euclidean(self) -> List[KMeansResult]:
        kmeans = KMeans(n_clusters=len(self.clusters))
        kmeans.fit(self.data)
        predicted = kmeans.predict(self.data)
        centers = kmeans.cluster_centers_
        return [KMeansResult(c, y) for y, c in zip(predicted, centers)]
