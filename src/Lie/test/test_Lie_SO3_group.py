import unittest

from Lie.Lie_SO3_group import LieSO3Group
from Lie.Lie_SO3_group import SO3Point


class LieSO3GroupTest(unittest.TestCase):
    @unittest.skip('Ignored')
    def test_init(self):
        so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
        so3_shape = (3, 3)
        so3_group = LieSO3Group.build(so3_tangent_vec, so3_shape)
        print(str(so3_group))
        self.assertTrue(so3_group.group_point.shape == (3, 3))

    def test_init_2(self):
        so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
        so3_group = LieSO3Group.build(so3_tangent_vec, [0.0, 0.0, 0.0])
        print(str(so3_group))
        so3_point1 = SO3Point(
            group_point=so3_group.group_point,
            base_point=[0, 0, 0],
            descriptor='3D Rotation\n[0.4 0.3 0.8]\n[0.2 0.4 0.1]\n[0.1 0.2 0.6]')

        self.assertTrue(so3_group.group_point.shape == (3, 3))

        base_point = [-0.4, 0.5, -0.3]
        so3_group2 = LieSO3Group.build(so3_tangent_vec, base_point)
        so3_point2 = SO3Point(
            group_point=so3_group2.group_point,
            base_point=base_point,
            descriptor='3D Rotation @ [-0.4, 0.5, -0.3]')
        LieSO3Group.visualize_all([so3_point1, so3_point2])

    @unittest.skip('Ignored')
    def test_inverse(self):
        so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
        so3_shape = (3, 3)
        so3_group = LieSO3Group.build(so3_tangent_vec, so3_shape)
        print(str(so3_group))
        so3_inv_group = so3_group.inverse()
        print(f'Inverted SO3:{so3_inv_group}')
        self.assertTrue(so3_inv_group.group_point.shape == (3, 3))
        titles = [
            '3D Rotation:\n[0.4 0.3 0.8]\n[0.2 0.4 0.1]\n[0.1 0.2 0.6]',
            '3D Inverse Rotation:\n[-0.4 -0.3 -0.8]\n[-0.2 -0.4 -0.1]\n[-0.1 -0.2 -0.6]'
        ]
        LieSO3Group.visualize_all([so3_group, so3_inv_group], titles)

    @unittest.skip('Ignored')
    def test_product_1(self):
        so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
        so3_shape = (3, 3)
        so3_group = LieSO3Group.build(so3_tangent_vec, so3_shape)

        so3_tangent_vec2 = [-x for x in so3_tangent_vec]
        so3_group2 = LieSO3Group.build(so3_tangent_vec2, so3_shape)
        so3_group_product = so3_group.product(so3_group2)
        self.assertTrue(so3_group_product.group_point.shape == (3, 3))
        print(f'\nSO3 Product:{so3_group_product}')
        titles = [
            '3D Rotation:\n[0.4 0.3 0.8]\n[0.2 0.4 0.1]\n[0.1 0.2 0.6]',
            '3D Rotation:\n[-0.4 -0.3 -0.8]\n[-0.2 -0.4 -0.1]\n[-0.1 -0.2 -0.6]'
        ]
        LieSO3Group.visualize_all([so3_group, so3_group2], titles)
        so3_group_product.visualize('Composition of 3D rotations')


    @unittest.skip('Ignored')
    def test_product_2(self):
        so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
        so3_shape = (3, 3)
        so3_group = LieSO3Group.build(so3_tangent_vec, so3_shape)

        so3_tangent_vec2 = [-0.5]*len(so3_tangent_vec)
        so3_group2 = LieSO3Group.build(so3_tangent_vec2, so3_shape)
        so3_group_product = so3_group.product(so3_group2)
        self.assertTrue(so3_group_product.group_point.shape == (3, 3))
        print(f'\nSO3 Product:{so3_group_product}')
        titles = [
            '3D Rotation:\n[0.4 0.3 0.8]\n[0.2 0.4 0.1]\n[0.1 0.2 0.6]',
            '3D Rotation:\n[-0.5 -0.5 -0.5]\n[-0.5 -0.5 -0.5]\n[-0.5 -0.5 -0.5]'
        ]
        LieSO3Group.visualize_all([so3_group, so3_group2], titles)
        so3_group_product.visualize('Composition of 3D rotations')

    @unittest.skip('Ignored')
    def test_product_3(self):
        so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
        so3_shape = (3, 3)
        so3_group = LieSO3Group.build(so3_tangent_vec, so3_shape)

        so3_tangent_vec2 = [0.5]*len(so3_tangent_vec)
        so3_group2 = LieSO3Group.build(so3_tangent_vec2, so3_shape)
        so3_group_product = so3_group.product(so3_group2)
        self.assertTrue(so3_group_product.group_point.shape == (3, 3))
        print(f'\nSO3 Product:{so3_group_product}')
        titles = [
            '3D Rotation:\n[0.4 0.3 0.8]\n[0.2 0.4 0.1]\n[0.1 0.2 0.6]',
            '3D Rotation:\n[0.5 0.5 0.5]\n[0.5 0.5 0.5]\n[0.5 0.5 0.5]'
        ]
        LieSO3Group.visualize_all([so3_group, so3_group2], titles)
        so3_group_product.visualize('Composition of 3D rotations')

    @unittest.skip('Ignored')
    def test_algebra(self):
        import random
        so3_tangent_vec = [random.randint(-10, 10)*0.1  for _ in range(9)]
        so3_shape = (3, 3)
        so3_group = LieSO3Group.build(so3_tangent_vec, so3_shape)
        print(f'SO3 Rotation matrix: {so3_group}')
        lie_algebra = so3_group.lie_algebra()
        assert lie_algebra.size == len(so3_tangent_vec)
        print(f'\nLie algebra:\n{lie_algebra}')

