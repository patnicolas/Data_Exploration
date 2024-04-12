import unittest
import path
import sys
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent)

from Riemannianconnection import RiemannianConnection
from manifoldpoint import ManifoldPoint
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
import numpy as np


class TestRiemannianConnection(unittest.TestCase):

    def test_init(self):
        dim = 2
        coordinates_type = 'extrinsic'
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True, default_coords_type=coordinates_type)
        riemannian_connection = RiemannianConnection(hypersphere, 'HyperSphere')
        print(str(riemannian_connection))

    def test_inner_product_identity(self):
        dim = 2
        coordinates_type = 'extrinsic'
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True, default_coords_type=coordinates_type)
        riemannian_connection = RiemannianConnection(hypersphere, 'HyperSphere')
        vector1 = np.array([0.4, 0.1, 0.8])
        base_point = np.array([0.4, 0.1, 0.0])
        inner_product = riemannian_connection.inner_product(vector1, vector1, base_point)
        self.assertAlmostEqual(first=inner_product, second=0.81, places=None, msg=None, delta=0.0001)
        print(f'Inner product tangent vector: {inner_product}')
        euclidean_inner_product = RiemannianConnection.euclidean_inner_product(vector1, vector1)
        print(f'Euclidean inner product: {euclidean_inner_product}')

    def test_inner_product(self):
        dim = 2
        coordinates_type = 'extrinsic'
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True, default_coords_type=coordinates_type)
        riemannian_connection = RiemannianConnection(hypersphere, 'HyperSphere')
        vector1 = np.array([0.4, 0.1, 0.8])
        vector2 = np.array([-0.4, -0.1, -0.8])
        base_point = np.array([0.4, 0.1, 0.0])
        inner_product = riemannian_connection.inner_product(vector1, vector2, base_point)
        self.assertAlmostEqual(first = inner_product, second=-0.81, places=None, msg=None, delta=0.0001)
        print(f'Inner product: {inner_product}')

    def test_norm(self):
        dim = 2
        coordinates_type = 'extrinsic'
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True, default_coords_type=coordinates_type)
        riemannian_connection = RiemannianConnection(hypersphere, 'HyperSphere')
        vector = np.array([0.4, 0.1, 0.8])
        base_point = np.array([0.4, 0.1, 0.0])
        norm = riemannian_connection.norm(vector, base_point)
        self.assertAlmostEqual(first=norm, second=np.linalg.norm(vector), places=None, msg=None, delta=0.0001)
        print(f'Norm: {norm}')

    def test_parallel_transport(self):
        dim = 2
        coordinates_type = 'extrinsic'
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True, default_coords_type=coordinates_type)
        riemannian_connection = RiemannianConnection(hypersphere, 'HyperSphere')
        base_point = np.array([1.5, 2.0, 1.6])
        manifold_base_point = ManifoldPoint(
            id='base',
            location=base_point,
            tgt_vector=np.array([[0.5, 0.1, 2.1]]))
        manifold_end_point = ManifoldPoint(id='end',location=base_point + 0.4)
        parallel_trans = riemannian_connection.parallel_transport(manifold_base_point, manifold_end_point)
        print(f'Parallel transport: {parallel_trans}')


if __name__ == '__main__':
    unittest.main()