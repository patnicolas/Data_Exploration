from unittest import TestCase


class DiffManifoldTest(TestCase):
    def test_init(self):
        x, y = symbols('x y', real=True)
        X, Y = symbols('X Y', real=True)
        this_coord_sys = CoordModel('cartesian', (x, y), Matrix([sqrt(x ** 2 + y ** 2), atan2(x, y)]))
        this_inv_coord_sys = CoordModel('polar', (X, Y), Matrix([X * cos(Y), X * sin(Y)]))

        diff_manifold2 = DiffManifold('M', 2, 'P', this_coord_sys, this_inv_coord_sys)
        print(str(diff_manfiold2))