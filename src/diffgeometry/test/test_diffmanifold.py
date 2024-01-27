import unittest
import sys
import path
from sympy import symbols, Matrix, sqrt, cos, sin, atan2

directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
print(sys.path)
from diffmanifold import DiffManifold, CoordModel


class TestDiffManifold(unittest.TestCase):

    @staticmethod
    def __create_manifold():
        x, y = symbols('x y', real=True)
        X, Y = symbols('X Y', real=True)
        this_coord_sys = CoordModel('cartesian', (x, y), Matrix([sqrt(x ** 2 + y ** 2), atan2(x, y)]))
        this_inv_coord_sys = CoordModel('polar', (X, Y), Matrix([X * cos(Y), X * sin(Y)]))
        return DiffManifold('M', 2, 'P', this_coord_sys, this_inv_coord_sys)

    def test_coord_model(self):
        x, y = symbols('x y', real=True)
        this_coord_sys = CoordModel('cartesian', (x, y), Matrix([sqrt(x ** 2 + y ** 2), atan2(x, y)]))
        print(f'Coordinate model: {str(this_coord_sys)}')
        print(f'Coordinate lambda: {str(this_coord_sys.get_lambda())}')

    def test_init(self):
        diff_manifold = TestDiffManifold.__create_manifold()
        print(f'Manifold: --------\n{str(diff_manifold)}')

    def test_coord_systems(self):
        diff_manifold = TestDiffManifold.__create_manifold()
        cartesian, polar = diff_manifold.get_coord_systems()
        print(f'Cartesian coordinates:\n{str(cartesian)}\nPolar:\n{str(polar)}')

    def test_get_src_base_scalar_field(self):
        diff_manifold = TestDiffManifold.__create_manifold()
        base_scalar_field = diff_manifold.get_base_scalar_field(True, DiffManifold.norm, (2, 0))
        print(f'Base Scalar Field: {base_scalar_field}')

    def test_get_inv_base_scalar_field(self):
        diff_manifold = TestDiffManifold.__create_manifold()
        base_scalar_field = diff_manifold.get_base_scalar_field(False, DiffManifold.norm, (2, 3))
        print(f'Base Scalar Inv Field: {base_scalar_field}')


if __name__ == '__main__':
    unittest.main()