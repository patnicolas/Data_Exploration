__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from sympy.vector import CoordSys3D
from typing import AnyStr, List
from dataclasses import dataclass

@dataclass
class Vector3D:
    x: float
    y: float
    z: float

    def __call__(self, ref_frame: AnyStr):
        frame = CoordSys3D(ref_frame)
        return self.x*frame.i + self.y*frame.j + self.z*frame.k

    def mul(self, alpha: float) -> Vector3D:
        return Vector3D(self.x*alpha, self.y*alpha, self.z*alpha)

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def dot(self, other: Vector3D) -> float:
        return self.x*other.x + self.y*other.y + self.z.otherz

    def __str__(self):
        return f'{self.x}.i + {self.y}.j + {self.z}.k'


class VectorAlgebra3D(object):
    def __init__(self, vectors: List[Vector3D]):
        self.vectors = vectors

    def sum(self) -> Vector3D:
       sum_x = sum([v.x for v in self.vectors])
       sum_y = sum([v.y for v in self.vectors])
       sum_z = sum([v.z for v in self.vectors])
       return Vector3D(sum_x, sum_y, sum_z)

    def mul(self, alpha: float) -> List[Vector3D]:
        return [v.mul(alpha) for v in self.vectors]


if __name__ == '__main__':
    v = Vector3D(3.0, 6.2, 1.0)
    print(v.mul(2.0))
    vector_algebra = VectorAlgebra3D([Vector3D(3, 6, 1), Vector3D(4, 10, 1)])
    print(str(vector_algebra.sum()))
