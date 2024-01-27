__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2024  All rights reserved."

from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField
from sympy import Lambda, symbols, Matrix, atan2, cos, sin, sqrt
from typing import AnyStr, Optional, Callable, TypedDict, Tuple, List, Any
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
        _lambda = coord_model.get_lambda()
        inv_lambda = inv_coord_model.get_lambda()

        self.relation: dict[tuple[AnyStr, AnyStr], Lambda] = {
            (coord_model.name, inv_coord_model.name): _lambda,
            (inv_coord_model.name, coord_model.name): inv_lambda
        }

    def get_lambdas(self) -> List[AnyStr]:
        return [str(rel[1].expr.args[2]) for rel in self.relation.items()]

    def get_coord_names(self) -> (AnyStr, AnyStr):
        return list(self.relation.keys())[0]

    def __str__(self) -> AnyStr:
        return "\n".join([f'{str(rel[0])}: {str(rel[1].expr.args[2])}' for rel in self.relation.items()])

    def get_coord_systems(self) -> (CoordSystem, CoordSystem):
        x, y = symbols('x y', real=True)
        coord_name, inv_coord_name = self.get_coord_names()
        coord_sys = CoordSystem(coord_name, self.patch, [x, y], self.relation)

        X, Y = symbols('X Y', real=True)
        inv_coord_sys = CoordSystem(inv_coord_name, self.patch, [X, Y], self.relation)
        return coord_sys, inv_coord_sys

    def get_base_scalar_field(self,
                              first_coord_system: bool,
                              func: Callable[[[BaseScalarField]], BaseScalarField],
                              var_symbols: tuple) -> Any:
        coord_system_index = 0 if first_coord_system else 1
        target_coord_system = CoordSystem(self.get_coord_names()[coord_system_index], self.patch)
        first_field = BaseScalarField(target_coord_system, 0)
        second_field = BaseScalarField(target_coord_system, 1)
        target = target_coord_system.point(list(var_symbols))
        return func([first_field, second_field]).rcall(target)

    def get_base_vector_field(self, index: int):
        coord
        from sympy.diffgeom.rn import R2, R2_p, R2_r
        from sympy.diffgeom import BaseVectorField


    @staticmethod
    def norm(base_scalar_fields: List[BaseScalarField]) -> BaseScalarField:
        return sqrt(base_scalar_fields[0] ** 2 + base_scalar_fields[1] ** 2)


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
    v = fx*R2_r.i + fy*R2_r.j
    pprint(v)

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
    """
    test_rn_vector()
    
    wedge_product_r3()
    wedge_product_c()
       """
    x, y = symbols('x y', real=True)
    X, Y = symbols('X Y', real=True)
    this_coord_sys = CoordModel('cartesian', (x, y), Matrix([sqrt(x**2 + y**2), atan2(x, y)]))
    this_inv_coord_sys = CoordModel('polar', (X, Y), Matrix([X*cos(Y), X*sin(Y)]))

    diff_manifold = DiffManifold('M', 2, 'P', this_coord_sys, this_inv_coord_sys)
    print(diff_manifold.get_coord_names())

    coord_system_1, coord_system_2 = diff_manifold.get_coord_names()
    print(str(coord_system_1))

    base_scalar_field = diff_manifold.get_base_scalar_field(True, DiffManifold.norm, (2, 0))
    print(base_scalar_field)

#    coord, inv_coord = diff_manifold.base_scalar_field('Cartesian', 'Polar')


