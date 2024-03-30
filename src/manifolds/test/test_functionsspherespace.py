import unittest
import path
import sys
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent)

from manifoldpoint import ManifoldPoint
from functionsspherespace import FunctionsSphereSpace
import numpy as np


class TestFunctionsSphereSpace(unittest.TestCase):
    @unittest.skip('ignore')
    def test_init(self):
        num_samples = 100
        functions_sphere_space = FunctionsSphereSpace(num_samples)
        print(str(functions_sphere_space))

    @unittest.skip('ignore')
    def test_create_manifold_point(self):
        num_samples = 100
        functions_sphere_space = FunctionsSphereSpace(num_samples)
        random_pt = functions_sphere_space.random_point()
        print(random_pt)
        vector = np.array([1.0, 0.5, 1.0])
        manifold_pt = functions_sphere_space.create_manifold_point(vector, random_pt)
        print(str(manifold_pt))

    def test_random_manifold_points(self):
        num_hilbert_samples = 8
        functions_sphere_space = FunctionsSphereSpace(num_hilbert_samples)
        n_points = 5
        random_manifold_pts = functions_sphere_space.random_manifold_points(n_points)
        print(random_manifold_pts[0])

    def test_exp(self):
        num_Hilbert_samples = 8
        functions_sphere_space = FunctionsSphereSpace(num_Hilbert_samples)

        vector = np.array([0.5, 1.0, 0.0, 0.4, 0.7, 0.6, 0.2, 0.9])
        assert num_Hilbert_samples == len(vector)
        exp_map_pt = functions_sphere_space.exp(vector, functions_sphere_space.random_manifold_points(1)[0])
        print(f'Exponential on Hilbert Sphere: {str(exp_map_pt)}')

    def test_log(self):
        num_Hilbert_samples = 8
        functions_sphere_space = FunctionsSphereSpace(num_Hilbert_samples)

        random_points = functions_sphere_space.random_manifold_points(2)
        log_map_pt = functions_sphere_space.log(random_points[0], random_points[1])
        print(f'Logarithm from Hilbert Sphere {str(log_map_pt)}')

    def test_inner_product_same_vector(self):
        import math

        num_Hilbert_samples = 8
        functions_sphere_space = FunctionsSphereSpace(num_Hilbert_samples)
        vector = np.array([0.5, 1.0, 0.0, 0.4, 0.7, 0.6, 0.2, 0.9])
        inner_prd = functions_sphere_space.inner_product(vector, vector)
        print(f'Euclidean norm of vector: {np.linalg.norm(vector)}')
        print(f'Inner product of same vector: {str(inner_prd)}')
        print(f'Norm of vector: {str(math.sqrt(inner_prd))}')

    def test_inner_product_different_vectors(self):
        num_Hilbert_samples = 8
        functions_sphere_space = FunctionsSphereSpace(num_Hilbert_samples)
        vector1 = np.array([0.5, 1.0, 0.0, 0.4, 0.7, 0.6, 0.2, 0.9])
        vector2 = np.array([0.5, 0.5, 0.2, 0.4, 0.6, 0.6, 0.5, 0.5])
        inner_prd = functions_sphere_space.inner_product(vector1, vector2)
        print(f'Inner product of different vectors: {str(inner_prd)}')


if __name__ == '__main__':
    unittest.main()
