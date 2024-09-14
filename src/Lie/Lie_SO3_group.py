__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from typing import List, Self, AnyStr, Tuple, NoReturn
from dataclasses import dataclass

@dataclass
class SO3Point:
    group_point: np.array
    base_point: np.array
    descriptor: AnyStr




class LieSO3Group(object):
    lie_group =  SpecialOrthogonal(n=3, point_type='vector', equip=False)

    # Exponentiates a left-invariant vector field from a base point.
    def __init__(self, tgt_vector: np.array, base_point: np.array = None) -> None:
        assert tgt_vector.size == 9, f'Rotation matrix size {tgt_vector.size} should be 9'
        self.tangent_vec = gs.array(tgt_vector)
        self.base_point = base_point
        self.group_point = LieSO3Group.lie_group.exp(self.tangent_vec, base_point)

    @classmethod
    def build(cls, tgt_vector: List[float], base_point: List[float] = (0.0, 0.0, 0.0)) -> Self:
        np_input = np.reshape(tgt_vector, (3, 3))
        return cls(tgt_vector=np_input, base_point=np.array(base_point))

    def __str__(self) -> AnyStr:
        return f'\nTangent vector:\n{str(self.tangent_vec)}\nLie group point:\n{str(self.group_point)}'

    def __eq__(self, _lie_group_3_util: Self) -> bool:
        return self.group_point == _lie_group_3_util.group_point

    def lie_algebra(self) -> np.array:
        return LieSO3Group.lie_group.log(self.group_point)

    def product(self, lie_group_so3: Self) -> Self:
        composed_group_point = LieSO3Group.lie_group.compose(self.group_point, lie_group_so3.group_point)
        return LieSO3Group(composed_group_point)

    def inverse(self) -> Self:
        inverse_group_point = LieSO3Group.lie_group.inverse(self.group_point)
        return LieSO3Group(inverse_group_point)

    def projection(self) -> Self:
        projected = LieSO3Group.lie_group.projection(self.group_point)
        return LieSO3Group(projected)

    def visualize(self, title: AnyStr) -> NoReturn:
        so3_point = SO3Point(self.group_point, self.base_point, title)
        LieSO3Group.visualize_all([so3_point])

    @staticmethod
    def visualize_all(so3_points: List[SO3Point]) -> NoReturn:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 12))
        if len(so3_points) == 1:
            ax = fig.add_subplot(111, projection="3d")
            LieSO3Group.__visualize_one(so3_points[0], ax)
        else:
            ax1 = fig.add_subplot(121, projection="3d")
            LieSO3Group.__visualize_one(so3_points[0], ax1)
            ax2 = fig.add_subplot(122, projection="3d")
            LieSO3Group.__visualize_one(so3_points[1], ax2)

        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def __visualize_one(so3_point: SO3Point, ax: Axes3D) -> NoReturn:
        import geomstats.visualization as visualization

        ax.text(x=-0.3, y=-0.4, z=-0.1, s=str(so3_point.base_point), fontdict={'size': 14})

        # cmap: cool, copper
        visualization.plot(so3_point.group_point, ax=ax, space="SO3_GROUP")
        ax.set_title(so3_point.descriptor, fontsize=15)
        LieSO3Group.__set_axes(ax)

    @staticmethod
    def __set_axes(ax: Axes3D) -> NoReturn:
        label_size = 13
        ax.set_xlabel('X values', fontsize=label_size)
        ax.set_ylabel('Y values', fontsize=label_size)
        ax.set_zlabel('Z values', fontsize=label_size)

        tick_size = 11
        for tick in ax.get_xticklabels():
            tick.set_fontsize(tick_size)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(tick_size)
        for tick in ax.get_zticklabels():
            tick.set_fontsize(tick_size)



if __name__ == '__main__':
    import random
    so3_tangent_vec = [0.4, 0.3, 0.8, 0.2, 0.4, 0.1, 0.1, 0.2, 0.6]
    so3_tangent_veca = [0.0, 0.1, 0.0, 0.6, 0.4, 0.6, 0.9, 0.2, 0.1]
    so3_shape = (3, 3)
    so3_group = LieSO3Group.build(so3_tangent_vec, so3_shape)
    so3_group_a = LieSO3Group.build(so3_tangent_veca, so3_shape)
    LieSO3Group.visualize_all([so3_group, so3_group_a])
    print(str(so3_group))
    so3_group.visualize()

    lie_algebra = so3_group.lie_algebra()
    assert lie_algebra.size == len(so3_tangent_vec)
    print(f'Lie algebra:\n{lie_algebra}')

    so3_inv_group = so3_group.inverse()
    print(f'Inverted SO3:{so3_inv_group}')

    so3_tangent_vec2 = [x*random.random() for x in so3_tangent_vec]
    so3_group2 = LieSO3Group.build(so3_tangent_vec2, so3_shape)
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


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

visualization.plot(data[:2], ax=ax, space="SO3_GROUP")

ax.set_title("3D orientations of the beds.");
    """