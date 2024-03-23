__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import numpy as np
from typing import List, AnyStr
from geomstats.geometry.base import LevelSet
from dataclasses import dataclass

"""
    Data class for defining a Manifold point as a pair of data point on the manifold and 
    a vector.
    :param id: Identifier for the point on a manifold
    :param data_point: Point on the manifold
    :param tgt_vector: Optional reference Tangent vector
    :param geodesic: Enable computation and display of geodesic if True, none otherwise
    :param intrinsic: Flag if the coordinate for this Manifold point is intrinsic
"""


@dataclass
class ManifoldPoint:
    id: AnyStr
    location: np.array
    tgt_vector: List[float] = None
    geodesic: bool = False
    intrinsic: bool = False

    def to_intrinsic(self, space: LevelSet) -> np.array:
        """
        Convert the location as numpy array  from extrinsic to intrinsic coordinates if extrinsic
        coordinates have been used.
        The same coordinates are returned if the manifold location is intrinsic
        :param space Generic manifold
        :return intrinsic coordinates as Numpy array
        """
        return space.extrinsic_to_intrinsic_coords(self.location) if not self.intrinsic else self.location

    def to_intrinsic_polar(self, space: LevelSet) -> np.array:
        """
        Convert current extrinsic or intrinsic cartesian coordinates into intrinsic polar coordinates
        :param space Generic manifold
        :return intrinsic polar coordinates as Numpy array
        """
        intrinsic_coordinates = self.to_intrinsic(space)
        return ManifoldPoint.__cartesian_to_polar(intrinsic_coordinates)

    def to_extrinsic(self, space: LevelSet) -> np.array:
        """
        Convert the location as numpy array from intrinsic to extrinsic coordinates if intrinsic
        coordinates have been used.
        The same coordinates are returned if the manifold location is extrinsic
        :param space Generic manifold
        :return extrinsic coordinates as Numpy array
        """
        return space.intrinsic_to_extrinsic_coords(self.location) if self.intrinsic else self.location

    # -------------- Helper private methods ----------------------

    @staticmethod
    def __cartesian_to_polar(cartesian_coordinates: np.array) -> np.array:
        import math
        # Make sure that the cartesian coordinates are defined in 2-dimension space
        if len(cartesian_coordinates) != 2:
            raise GeometricException(f'Number of coordinates {len(cartesian_coordinates)} should be 2')

        x = cartesian_coordinates[0]
        y = cartesian_coordinates[1]

        # The intrinsic cartesian point cannot be the origin
        if x == 0.0 and y == 0.0:
            raise GeometricException(f'x {x} and y {y} should not be 0')

        r = math.sqrt(x ** 2 + y ** 2)
        theta = math.acos(x / r) if y >= 0.0 else -math.acos(x / r)
        return np.array([r, theta])
