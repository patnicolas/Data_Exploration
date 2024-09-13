__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.special_euclidean import SpecialEuclidean

import numpy as np
from typing import List, Self, AnyStr, Tuple
from manifolds.geometricexception import GeometricException


class LieGroup3Util(object):
    def __init__(self, np_matrix: np.array) -> None:
        match np_matrix.size:
            case 9:
                self.lie_group = SpecialOrthogonal(n=3)
            case 36:
                self.lie_group = SpecialEuclidean(n=3)
            case _:
                raise GeometricException(f'\nLength of tangent vector {len(np_matrix)} should be 3 or 6')
        self.tangent_vec = gs.array(np_matrix)
        self.group_point = self.lie_group.exp(self.tangent_vec)

    @classmethod
    def build(cls, matrix: List[float], shape: Tuple[int, int]) -> Self:
        np_input = np.reshape(matrix, shape)
        return cls(np_matrix=np_input)

    def __str__(self) -> AnyStr:
        return f'\nTangent vector:\n{str(self.tangent_vec)}\nLie group point:\n{str(self.group_point)}'

    def __eq__(self, _lie_group_3_util: Self) -> bool:
        return self.group_point == _lie_group_3_util.group_point

    def lie_algebra(self) -> np.array:
        return self.lie_group.log(self.group_point)

    def product(self, lie_group_so3: Self) -> Self:
        composed_group_point = self.lie_group.compose(self.group_point, lie_group_so3.group_point)
        return LieGroup3Util(composed_group_point)

    def inverse(self) -> Self:
        inverse_group_point = self.lie_group.inverse(self.group_point)
        return LieGroup3Util(inverse_group_point)


if __name__ == '__main__':
    import random
    so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
    so3_shape = (3, 3)
    so3_group = LieGroup3Util.build(so3_tangent_vec, so3_shape)
    print(str(so3_group))

    lie_algebra = so3_group.lie_algebra()
    assert lie_algebra.size == len(so3_tangent_vec)
    print(f'Lie algebra:\n{lie_algebra}')

    so3_inv_group = so3_group.inverse()
    print(f'Inverted SO3:{so3_inv_group}')

    so3_tangent_vec2 = [x*random.random() for x in so3_tangent_vec]
    so3_group2 = LieGroup3Util.build(so3_tangent_vec2, so3_shape)
    so3_group_product = so3_group.product(so3_group2)
    print(f'SO3 Product:{so3_group_product}')


    """
     Create a vector in the Lie algebra
tangent_vec = gs.array([0.1, 0.2, 0.3])

# Use the exponential map to get a point on the Lie group (SO(3))
point_on_group = SO3.exp(tangent_vec)
print("Point on SO(3):", point_on_group)

# Use the logarithmic map to get back to the Lie algebra
log_map = SO3.log(point_on_group)
print("Log map (Lie algebra):", log_map)


# Create another point on the Lie group
point_on_group2 = SO3.exp(gs.array([0.3, -0.2, 0.5]))

# Perform group multiplication (composition of rotations)
group_product = SO3.compose(point_on_group, point_on_group2)
print("Group multiplication (composition):", group_product)

# Compute the inverse of a point on the Lie group
group_inverse = SO3.inverse(point_on_group)
print("Inverse of the point on SO(3):", group_inverse)


# Create a vector in the Lie algebra of SE(3)
tangent_vec_se3 = gs.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # 3 for rotation, 3 for translation

# Exponential map to get a point on SE(3) (translation and rotation)
point_on_se3 = SE3.exp(tangent_vec_se3)
print("Point on SE(3):", point_on_se3)

# Logarithmic map to return to Lie algebra
log_map_se3 = SE3.log(point_on_se3)
print("Log map (SE(3)): ", log_map_se3)
    
    # Random sampling from SO(3)
random_point_so3 = SO3.random_point()
print("Random point on SO(3):", random_point_so3)

# Compute the Riemannian distance between two points on the group
distance_so3 = SO3.metric.dist(point_on_group, point_on_group2)
print("Distance between two points on SO(3):", distance_so3)
    """