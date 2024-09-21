import unittest

from Lie.Lie_SO3_group import LieSO3Group
from Lie.Lie_SO3_group import SO3Point


class LieSO3GroupTest(unittest.TestCase):

    @unittest.skip('Ignored')
    def test_init(self):
        so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
        so3_group = LieSO3Group.build(so3_tangent_vec)
        print(str(so3_group))
        self.assertTrue(so3_group.group_element.shape == (3, 3))

    @unittest.skip('Ignored')
    def test_init_2(self):
        import numpy as np

        so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
        so3_group = LieSO3Group.build(so3_tangent_vec)
        print(str(so3_group))
        so3_point1 = SO3Point(
            group_element=so3_group.group_element,
            base_point=LieSO3Group.identity,
            descriptor='SO3 point elements\n[0.4 0.3 0.8]\n[0.2 0.4 0.1]\n[0.1 0.2 0.6]\n@Identity')

        self.assertTrue(so3_group.group_element.shape == (3, 3))

        base_point = [-0.4, 0.5, -0.3, 0.0, 1.0, -0.6, 0.2, 0.1, 0.0]
        so3_group2 = LieSO3Group.build(so3_tangent_vec, base_point)
        so3_point2 = SO3Point(
            group_element=so3_group2.group_element,
            base_point=np.reshape(base_point, (3, 3)),
            descriptor='Same SO3 point elements\n@ [-0.4  0.5 -0.3]\n[ 0.0  1.0 -0.6]\n[ 0.2  0.1  0.0]')
        LieSO3Group.visualize_all([so3_point1, so3_point2])

    @unittest.skip('Ignored')
    def test_inverse(self):
        # Original SO3 rotation matrix
        so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
        so3_group = LieSO3Group.build(so3_tangent_vec)
        so3_point = SO3Point(
            group_element=so3_group.group_element,
            base_point=LieSO3Group.identity,
            descriptor='SO3 point:\n[0.4 0.3 0.8]\n[0.2 0.4 0.1]\n[0.1 0.2 0.6]\n@ Identity')
        print(str(so3_group))

        # Inverse SO3 rotation matrix
        so3_inv_group = so3_group.inverse()
        inv_so3_point = SO3Point(
            group_element=so3_inv_group.group_element,
            base_point=LieSO3Group.identity,
            descriptor='SO3 inverse point:\n[-0.4 -0.3 -0.8]\n[-0.2 -0.4 -0.1]\n[-0.1 -0.2 -0.6]')
        print(f'SO3 Inverse point:{so3_inv_group}')
        self.assertTrue(so3_inv_group.group_element.shape == (3, 3))

        # Visualization
        LieSO3Group.visualize_all([so3_point, inv_so3_point], 1)


    @unittest.skip('Ignored')
    def test_product_1(self):
        # First SO3 rotation matrix
        so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec)
        so3_point = SO3Point(
            group_element=so3_group.group_element,
            base_point=LieSO3Group.identity,
            descriptor='SO3 point:\n[0.4 0.3 0.8]\n[0.2 0.4 0.1]\n[0.1 0.2 0.6]\n@ Identity')

        # Second SO3 rotation matrix
        so3_tangent_vec2 = [-x for x in so3_tangent_vec]
        so3_group2 = LieSO3Group.build(tgt_vector=so3_tangent_vec2)
        so3_point2 = SO3Point(
            group_element=so3_group2.group_element,
            base_point=LieSO3Group.identity,
            descriptor='SO3 inverse point:\n[-0.4 -0.3 -0.8]\n[-0.2 -0.4 -0.1]\n[-0.1 -0.2 -0.6]\n@ Identity')

        # Composition of two SO3 rotation matrices
        so3_group_product = so3_group.product(so3_group2)
        self.assertTrue(so3_group_product.group_element.shape == (3, 3))
        print(f'\nSO3 Product:{so3_group_product}')

        # Visualization
        LieSO3Group.visualize_all([so3_point, so3_point2], 1)
        so3_group_product.visualize('Composition of 3D rotations', 1)



    def test_product_2(self):
        # First SO3 rotation matrix
        so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec)
        so3_point = SO3Point(
            group_element=so3_group.group_element,
            base_point=LieSO3Group.identity,
            descriptor='SO3 point:\n[0.4 0.3 0.8]\n[0.2 0.4 0.1]\n[0.1 0.2 0.6]\n@ Identity')

        # Second SO3 rotation matrix
        so3_tangent_vec2 = [-0.5]*len(so3_tangent_vec)
        so3_group2 = LieSO3Group.build(tgt_vector=so3_tangent_vec2)
        so3_point2 = SO3Point(
            group_element=so3_group2.group_element,
            base_point=LieSO3Group.identity,
            descriptor='SO3 point:\n[-0.5 -0.5 -0.5]\n[-0.5 -0.5 -0.5]\n[-0.5 -0.5 -0.5]\n@ Identity')

        # Composition of two SO3 rotation matrices
        so3_group_product = so3_group.product(so3_group2)
        self.assertTrue(so3_group_product.group_element.shape == (3, 3))
        print(f'\nSO3 points composition:{so3_group_product}')

        # Visualization
        LieSO3Group.visualize_all([so3_point, so3_point2], 1)
        so3_group_product.visualize('Composition of 3D rotations', 0)

    @unittest.skip('Ignored')
    def test_product_3(self):
        # First SO3 rotation matrix
        so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec)
        so3_point = SO3Point(
            group_element=so3_group.group_element,
            base_point=LieSO3Group.identity,
            descriptor='3D Rotation:\n[0.4 0.3 0.8]\n[0.2 0.4 0.1]\n[0.1 0.2 0.6]')

        # Second SO3 rotation matrix
        so3_tangent_vec2 = [0.5]*len(so3_tangent_vec)
        so3_group2 = LieSO3Group.build(tgt_vector=so3_tangent_vec2)
        so3_point2 = SO3Point(
            group_element=so3_group2.group_element,
            base_point=LieSO3Group.identity,
            descriptor='3D Rotation:\n[0.5 0.5 0.5]\n[0.5 0.5 0.5]\n[0.5 0.5 0.5]')

        # Composition of two SO3 rotation matrices
        so3_group_product = so3_group.product(so3_group2)
        self.assertTrue(so3_group_product.group_element.shape == (3, 3))
        print(f'\nSO3 Product:{so3_group_product}')

        LieSO3Group.visualize_all([so3_point, so3_point2])
        so3_group_product.visualize('Composition of 3D rotations')


    @unittest.skip('Ignored')
    def test_algebra(self):
        import random
        so3_tangent_vec = [random.randint(-10, 10)*0.1 for _ in range(9)]
        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec)
        print(f'SO3 Rotation matrix: {so3_group}')
        lie_algebra = so3_group.lie_algebra()
        assert lie_algebra.size == len(so3_tangent_vec)
        print(f'\nLie algebra:\n{lie_algebra}')

    @unittest.skip('Ignored')
    def test_algebra2(self):
        import random
        so3_tangent_vec = [random.randint(-10, 10) * 0.1 for _ in range(9)]
        base_point = [-0.4, 0.5, -0.3, 0.0, 1.0, -0.6, 0.2, 0.1, 0.0]
        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec, base_point=base_point)
        print(f'SO3 Rotation matrix: {so3_group}')
        lie_algebra = so3_group.lie_algebra()
        assert lie_algebra.size == len(so3_tangent_vec)
        print(f'\nLie algebra:\n{lie_algebra}')

    @unittest.skip('Ignored')
    def test_projection(self):
        import random
        so3_tangent_vec = [random.randint(-10, 10) * 0.1 for _ in range(9)]
        base_point = [-0.4, 0.5, -0.3, 0.0, 1.0, -0.6, 0.2, 0.1, 0.0]
        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec, base_point=base_point)
        projected = so3_group.projection()
        print(f'\nProjected point:\n{projected.group_element}')


    @unittest.skip('Ignored')
    def test_bracket(self):
        import numpy as np

        a = np.eye(3)

        so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec, base_point=LieSO3Group.identity)
        print(f'\n{so3_group.group_element}')
        other_tgt_vec = [0.1, -0.3, 0.5, 0.1, 0.0, -0.1, 0.6, -0.3, 0.7]
        so3_group2 = LieSO3Group.build(tgt_vector=other_tgt_vec, base_point=LieSO3Group.identity)
        print(f'\n{so3_group2.group_element}')
        """
        first_term = Matrices.mul(inverse_base_point, tangent_vector_b)
        first_term = Matrices.mul(tangent_vector_a, first_term)

        second_term = Matrices.mul(inverse_base_point, tangent_vector_a)
        second_term = Matrices.mul(tangent_vector_b, second_term)
        """
        x = np.dot(so3_group.group_element, so3_group2.group_element)
        print(f'A.B\n{x}')
        y = np.dot(so3_group2.group_element, so3_group.group_element)
        print(f'B.A\n{y}')
        print(f'Manual bracket\n{(x - y)}')
        bracket = so3_group.bracket(other_tgt_vec)
        print(f'\nBracket:\n{bracket}')

