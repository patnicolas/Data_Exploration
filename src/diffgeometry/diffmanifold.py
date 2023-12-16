
from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField
from sympy import Lambda, symbols, Matrix, atan2, cos, sin, sqrt
from typing import AnyStr, Optional, Callable, TypedDict, Tuple, List
from dataclasses import dataclass



@dataclass
class CoordModel:
    name: AnyStr
    in_symbols: symbols
    transform: Matrix

    def get_lambda(self):
        return Lambda(self.in_symbols, self.transform)

class DiffManifold(object):
    def __init__(self,
                 name: AnyStr,
                 dimension: int,
                 patch: AnyStr,
                 coord_model: CoordModel,
                 inv_coord_model: CoordModel):

        self.manifold = Manifold(name, dimension)
        self.patch = Patch(patch, self.manifold)
        _lambda = coord_sys.get_lambda()
        inv_lambda = inv_coord_sys.get_lambda()

        self.relation = {
            (coord_sys.coord_name, inv_coord_sys.coord_name): _lambda,
            (inv_coord_sys.coord_name, coord_sys.coord_name): inv_lambda
        }

    def get_coord_systems(self) -> (CoordSystem, CoordSystem):
        coord_sys = CoordSystem(coord_name, self.patch, [x, y], this_relation)
        inv_coord_sys = CoordSystem(inv_coord_name, self.patch, [X, Y], this_relation)
        return coord_sys, inv_coord_sys

    def base_scalar_field(self,  coord_name: AnyStr, inv_coord_name: AnyStr) -> (CoordSystem, CoordSystem):
        this_relation = self.get_relation(coord_name, inv_coord_name)
        coord_sys = CoordSystem(coord_name, self.patch, [x, y], this_relation)
        inv_coord_sys = CoordSystem(inv_coord_name, self.patch, [X, Y], this_relation)
        return coord_sys, inv_coord_sys


def wedge_product_c():
    from sympy.diffgeom.rn import R3_c
    from sympy.diffgeom import WedgeProduct

    fx, fy, fz = R3_c.base_scalars()
    e_x, e_y, e_z = R3_c.base_vectors()
    dx, dy, dz = R3_c.base_oneforms()
    wp_xyz = WedgeProduct(dx, dy, dz)(e_x, e_y, e_z)
    print(wp_xyz)
    wp_yxz = WedgeProduct(dx, dy, dz)(e_y, e_x, e_z)
    print(wp_yxz)
    wp_xfyfz = WedgeProduct(dx, fx*dy, fx*dz)(fx*e_x, e_y, e_z)
    print(f'WedgeProduct(dx, fx*dy, fx*dz)(fx*e_x, e_y, e_z): {wp_xfyfz}')
    wp_xfyfz = WedgeProduct(dx, fx*dy, fx*dz)(4*e_x, e_y, e_z)
    print(f'WedgeProduct(dx, fx*dy, fx*dz)(4*e_x, e_y, e_z): {wp_xfyfz}')



def wedge_product_r3():
    from sympy.diffgeom.rn import R3_r
    from sympy.diffgeom import WedgeProduct

    x, y, z = symbols('x, y, z', real=True)
    fx, fy, fz = R3_r.base_scalars()
    e_x, e_y, e_z = R3_r.base_vectors()
    dx, dy, dz = R3_r.base_oneforms()
    wp_xyz = WedgeProduct(dx, dy, dz)(e_x, e_y, e_z)
    print(wp_xyz)
    wp_yxz = WedgeProduct(dx, dy, dz)(e_y, e_x, e_z)
    print(wp_yxz)
    wp_xfyfz = WedgeProduct(x*x*dx, dy, dz)(e_x, e_y, e_z)
    print(f'WedgeProduct(x*x*dx, dy, dz)(e_x, e_y, e_z): {wp_xfyfz}')
    wp_xfyfz = WedgeProduct(x*fx*dx, dy, dz)(e_x, e_y, e_z)
    print(f'WedgeProduct(x*fx*dx, dy, dz)(e_x, e_y, e_z): {wp_xfyfz}')
    wp_xfyfz = WedgeProduct(dx, dy, dz)(fx*e_x, e_y, e_z)
    print(f'WedgeProduct(dx, dy, dz)(fx*e_x, e_y, e_z): {wp_xfyfz}')
    wp_xfyfz = WedgeProduct(fx*dx, dy, dz)(fx*e_x, e_y, e_z)
    print(f'WedgeProduct(fx*dx, dy, dz)(fx*e_x, e_y, e_z): {wp_xfyfz}')
    wp_xfyfz = WedgeProduct(dx, fx*dy, fx*dz)(4*e_x, e_y, e_z)
    print(f'WedgeProduct(dx, fx*dy, fx*dz)(4*e_x, e_y, e_z): {wp_xfyfz}')

    wp_xfyfz = WedgeProduct(dx, fx*dy, fx*dz)(4*fx*e_x, e_y, e_z)
    print(f'WedgeProduct(dx, fx*dy, fx*dz)(4*fx*e_x, e_y, e_z): {wp_xfyfz}')
    wp_xfyfz = WedgeProduct(dx, fx*dy, fy*dz)(fx*e_x, e_y, e_z)
    print(f'WedgeProduct(dx, fx*dy, fx*dz)(fx*e_x, e_y, e_z): {wp_xfyfz}')
    wp_xfyfz = WedgeProduct(dx, dy, dz)(fx*e_x, e_y, e_z)
    print(f'WedgeProduct(dx, dy, dz)(fx*e_x, e_y, e_z): {wp_xfyfz}')
    wp_xfyfz = WedgeProduct(fx*dx, fy*dy, fz*dz)(e_x, e_y, e_z)
    print(f'WedgeProduct(fx*dx, fy*dy, fz*dz)(e_x, e_y, e_z): {wp_xfyfz}')
    wp_xfyfz = WedgeProduct(dx, dy, dz)(e_x, e_y, e_z)
    print(f'WedgeProduct(dx, dy, dz)(e_x, e_y, e_z): {wp_xfyfz}')
    wp_xfyfz = WedgeProduct(fx*dx, fy*dy, fz*dz)(fx*e_x, fy*e_y, fz*e_z)
    print(f'WedgeProduct(fx*dx, fy*dy, fz*dz)(fx*e_x, fy*e_y, fz*e_z): {wp_xfyfz}')
    wp_xfyfz = WedgeProduct(fx*dx, fx*dy, fx*dz)(fx*e_x, fx*e_y, fx*e_z)
    print(f'WedgeProduct(fx*dx, fx*dy, fx*dz)(fx*e_x, fx*e_y, fx*e_z): {wp_xfyfz}')
    wp_xfyfz = WedgeProduct(fy*dx, fy*dy, fy*dz)(fy*e_x, fy*e_y, fy*e_z)
    print(f'WedgeProduct(fy*dx, fy*dy, fy*dz)(fy*e_x, fy*e_y, fy*e_z): {wp_xfyfz}')
    wp_xfyfz = WedgeProduct(fx*fx*dx, fx*fx*dy, fx*fx*dz)(fx*fx*e_x, fx*fx*e_y, fx*fx*e_z)
    print(f'WedgeProduct(fx*fx*dx, fx*fx*dy, fx*fx*dz)(fx*fx*e_x, fx*fx*e_y, fx*fx*e_z): {wp_xfyfz}')



def differential():
    from sympy import Function
    from sympy.diffgeom.rn import R3_r, R3_c
    from sympy.diffgeom import Differential
    from sympy import pprint

    fx, fy, fz = R3_r.base_scalars()
    e_x, e_y, e_z = R3_r.base_vectors()
    g = Function('g')
    s_field = g(fx, fy, fz)
    dg = Differential(s_field)
    pprint(dg(e_x))
    pprint(dg(e_y))
    pprint(dg(e_z))

    fx, fy, fz = R3_c.base_scalars()
    e_x, e_y, e_z = R3_c.base_vectors()
    s_field = g(fx, fy, fz)
    dg = Differential(s_field)
    pprint(dg(e_x))
    pprint(dg(e_y))
    pprint(dg(e_z))



def test_rn_scalar():
    from sympy.diffgeom.rn import R2_r, R2_p
    from sympy import Function, pi

    rho, theta = R2_p.symbols
    fx, fy = R2_r.base_scalars()
    ftheta = BaseScalarField(R2_r, 1)
    print(f'fx: {fx}, fy: {fy} ftheta: {ftheta}')
    print((fx**2+fy**2).rcall(R2_p.point([rho, theta])))
    g = Function('g')
    fg = g(ftheta+pi)
    print(fg.rcall(R2_p.point([rho, theta])))
    print(g(-pi))

def test_rn_vector():
    from sympy import Function
    from sympy.diffgeom.rn import R2_p, R2_r
    from sympy.diffgeom import BaseVectorField
    from sympy import pprint

    x, y = R2_r.symbols
    fx, fy = R2_r.base_scalars()
    r_pt = R2_r.point([x, y])
    g = Function('g')
    s_field = g(fx, fy)
    pprint(f's_field: {s_field}')
    v0 = BaseVectorField(R2_r, 0)
    v1 = BaseVectorField(R2_r, 1)
    print('\nv0(s_field).rcall(r_pt).doit')
    pprint(v0(s_field).rcall(r_pt).doit())
    print('\nv1(s_field).rcall(r_pt).doit')
    pprint(v1(s_field).rcall(r_pt).doit())

    rho, theta = R2_p.symbols
    p_pt = R2_p.point([rho, theta])
    w0 = BaseVectorField(R2_p, 0)
    w1 = BaseVectorField(R2_p, 1)
    print('\nw0(s_field).rcall(p_pt).doit')
    pprint(w0(s_field).rcall(p_pt).doit())
    print('\nw1(s_field).rcall(p_pt).doit')
    pprint(w1(s_field).rcall(p_pt).doit())


if __name__ == '__main__':
    wedge_product_r3()
    wedge_product_c()
    """
    (x, y) = symbols('x y', real=True)
    (X, Y) = symbols('X Y', real=True)
    this_coord_sys = CoordDef((x, y), Matrix([sqrt(x**2 + y**2), atan2(x, y)]))
    this_inv_coord_sys = CoordDef((X, Y), Matrix([X*cos(Y), X*sin(Y)]))

    diff_manifold = DiffManifold('M', 2, 'P', this_coord_sys, this_inv_coord_sys)
    for relation in diff_manifold.get_relation('Cartesian', 'Polar').items():
        print(relation)

    coord, inv_coord = diff_manifold.base_scalar_field('Cartesian', 'Polar')
    """

