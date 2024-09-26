__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import numpy as np
from typing import AnyStr, List, Self
import geomstats.backend as gs
from geomstats.geometry.special_euclidean import SpecialEuclidean


class LieSE3Group(object):
    dim = 3
    # Lie group as defined in Geomstats library
    lie_group = SpecialEuclidean(n=dim, point_type='matrix', epsilon=0.15, equip=False)
    identity = gs.eye(dim)

    def __init__(self, rot_matrix: np.array, trans_matrix: np.array, base_point: np.array = identity) -> None:
        """
        Constructor for the wrapper for key operations on SE3 Special Euclidean Lie manifold
        @param rot_matrix: 3x3 Rotation matrix (see. LieSO3Group)
        @type rot_matrix: Numpy array
        @param trans_matrix: 3x3 matrix for translation
        @type trans_matrix: Numpy array
        @param base_point: Base point vector on the manifold (Identity if not defined)
        @type base_point: Numpy array
        """
        assert rot_matrix.size == 9, f'Rotation matrix size {rot_matrix.size} should be 9'
        assert trans_matrix.size == 9, f'Translation matrix size {trans_matrix.size} should be 9'
        assert base_point.size == 9, f'Base point size {base_point.size} should be 9'

        rotation_matrix = gs.array(rot_matrix)
        translation_matrix = gs.array(trans_matrix)

        self.se3_element = LieSE3Group.lie_group.compose(rotation_matrix,  translation_matrix)
        self.group_element = LieSE3Group.lie_group.exp(self.se3_element )
        self.base_point = base_point

    @classmethod
    def build(cls, rotation_matrix: List[float], trans_matrix: List[float], base_point: List[float] = None) -> Self:
        """
        Build an instance of LieSE3Group given a rotation matrix, a tangent vector and a base point if defined
        @param rotation_matrix: 3x3 Rotation matrix (see. LieSO3Group)
        @type rotation_matrix: List[float]
        @param tgt_vector: 3 length tangent vector for translation
        @type tgt_vector: List[float]
        @param base_point: Base point vector on the manifold (Identity if not defined)
        @type base_point: List[float]
        @return: Instance of LieSE3Group
        @rtype: LieSE3Group
        """
        np_rotation_matrix = np.reshape(rotation_matrix, (3, 3))
        np_translation_vector = np.reshape(trans_matrix, (3, 3))
        np_base_point = np.reshape(base_point, (3, 3)) if base_point is not None else LieSE3Group.identity
        return cls(np_rotation_matrix, np_translation_vector, np_base_point)

    def __str__(self) -> AnyStr:
        return f'\nSE3 element:\n{str(self.se3_element)}\nLie group point:\n{str(self.group_element)}'


"""
Certainly! Here's a simple example of how to work with the SE(3) Lie group and its associated Lie algebra using the Geomstats library in Python. SE(3) represents the special Euclidean group in 3D, which combines both rotations (SO(3)) and translations.

First, make sure you have installed the Geomstats library:

```bash
pip install geomstats
```

Now, here's the code that demonstrates basic operations with SE(3):

```python
import numpy as np
import geomstats.backend as gs
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.se3 import SE3

# Define the SE(3) group
se3_group = SE3()

# Example of an element of SE(3): combination of a rotation and a translation
rotation_matrix = gs.array([[0, -1, 0],
                            [1, 0, 0],
                            [0, 0, 1]])  # 90 degrees around z-axis
translation_vector = gs.array([1.0, 2.0, 3.0])  # Translation by (1, 2, 3)

se3_element = se3_group.compose(rotation_matrix, translation_vector)

# Print the SE(3) element
print("SE(3) element (rotation + translation):", se3_element)

# Lie algebra element (vector representation)
lie_algebra_element = gs.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
print("Lie algebra element (se3):", lie_algebra_element)

# Map the Lie algebra element to the Lie group (exponential map)
se3_from_algebra = se3_group.exp(lie_algebra_element)
print("SE(3) element from Lie algebra:", se3_from_algebra)

# Map back to the Lie algebra (logarithmic map)
lie_algebra_from_se3 = se3_group.log(se3_from_algebra)
print("Lie algebra element from SE(3):", lie_algebra_from_se3)

# Compose two elements of SE(3)
se3_element_2 = se3_group.compose(gs.eye(3), gs.array([0.5, 0.5, 0.5]))
se3_composed = se3_group.compose(se3_element, se3_element_2)
print("Composed SE(3) element:", se3_composed)

# Inverse of an SE(3) element
se3_inverse = se3_group.inverse(se3_element)
print("Inverse of SE(3) element:", se3_inverse)
```

### Explanation:
1. **SE(3) Group Definition:**
   We define the SE(3) group using the `SpecialEuclidean` class for dimension 3.

2. **Creating SE(3) Elements:**
   We create an SE(3) element by combining a rotation matrix and a translation vector.

3. **Lie Algebra:**
   We represent a Lie algebra element in vector form and map it to the SE(3) group using the exponential map (`exp`). The logarithmic map (`log`) takes us back to the algebra.

4. **Composing and Inverting Elements:**
   We demonstrate how to compose two SE(3) elements and how to compute the inverse of an SE(3) element.

This code provides a basic framework for working with SE(3) using Geomstats.



"""