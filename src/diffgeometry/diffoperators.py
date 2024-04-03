__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from sympy.vector import CoordSys3D, VectorZero
from sympy import Expr
from typing import Any


class DiffOperator(object):
    def __init__(self, expr: Expr = None):
        self.expr = expr

    def gradient(self) -> VectorZero:
        from sympy.vector import gradient
        return gradient(self.expr, doit=True)

    def divergence(self, base_vectors: Expr) -> VectorZero:
        from sympy.vector import divergence
        div_vec = self.expr * base_vectors
        return divergence(div_vec, doit=True)

    def curl(self, base_vectors: Expr) -> VectorZero:
        from sympy.vector import curl
        curl_vec = self.expr * base_vectors
        return curl(curl_vec, doit=True)

    def laplace(self):
        from sympy import laplace_transform
        return laplace_transform(self.expr, t, s, noconds=True)

    def fourier(self):
        from sympy import fourier_transform
        return fourier_transform(self.expr, x, k)


if __name__ == '__main__':
    def dim2_test():
        import sympy
        import math
        from sympy import symbols
        from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField
        import matplotlib.pyplot as plt
        import numpy as np

        x, y = symbols('x, y', real=True)
        norm = sympy.sqrt(x * x + y + y)
        coord_system = CoordSystem('x y', Patch('P', Manifold('square', 2)))
        first_field = BaseScalarField(coord_system, 0)
        second_field = BaseScalarField(coord_system, 1)
        v = -second_field / norm + first_field / norm
        w = v.evalf(subs={x: 1.0, y: 2.0})
        print(w)
        x_values = np.linspace(-8, 8, 16)
        n_x, n_y = np.meshgrid(x_values, x_values)
        np_norm = np.sqrt(n_x ** 2 + n_y ** 2)
        X = -n_y / np_norm
        Y = n_x / np_norm
        plt.quiver(n_x, n_y, X, Y, color='red')
        plt.show()


    # dim2_test()

    def dim3_test():
        import sympy
        import math
        from sympy import exp, sin, symbols, pprint
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        import numpy as np

        r = CoordSys3D('r')
        f = (2 * r.x + r.z ** 3) * r.i + (r.x * r.y + sympy.exp(-r.y) - r.z ** 2) * r.j + (r.x ** 3 / r.y) * r.k
        w = f.evalf(subs={r.x: 1.0, r.y: 2.0, r.z: 0.2})
        print(w)
        grid_values = np.linspace(0.1, 0.9, 8)
        n_x, n_y, n_z = np.meshgrid(grid_values, grid_values, grid_values)
        X = 2 * n_x + n_z ** 3
        Y = n_x * n_y + np.exp(-n_y) - n_z ** 2
        Z = n_x ** 3 / n_y

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.quiver(n_x, n_y, n_z, X, Y, Z, length=0.2, color='blue', normalize=True)
        plt.show()


    # dim3_test()

    def covector_test():
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import numpy as np

        # Limit values for vector and covector fields
        _min = -8
        _max = 8

        fig, ax = plt.subplots()
        # Generate co-vector fields
        co_vectors = [{'center': (0.0, 0.0), 'radius': _max * (1 - 1 / n), 'color': 'blue', 'fill': False}
                      for n in range(1, 16)]
        [plt.gca().add_patch(Circle(circle['center'], circle['radius'], color=circle['color'], fill=circle['fill']))
         for circle in co_vectors]

        # Set limits for x and y axes
        ax.set_aspect('equal')
        ax.set_xlim(_min, _max)
        ax.set_ylim(_min, _max)

        # Set up the grid
        grid_values = np.linspace(_min, _max, _max - _min)
        x, y = np.meshgrid(grid_values, grid_values)

        # Display vector and co-vector fields
        plt.quiver(x, y, x, y, color='red')
        plt.show()


    covector_test()


    def co_field():
        import numpy as np
        import matplotlib.pyplot as plt
        # Define vector field function
        def vector_field(x, y):
            return x, y

        # Define covector field function
        def covector_field(x, y):
            # In this example, let's choose g(x, y) = x - y
            return 2.0 * (1.0 - 1.0 / np.sqrt(x ** 2 + y ** 2))

        # Generate grid of points
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)

        # Compute covector field values
        G = covector_field(X, Y)

        # Plot vector field
        plt.figure(figsize=(6, 6))
        plt.quiver(X, Y, vector_field(X, Y)[0], vector_field(X, Y)[1], color='blue', angles='xy', scale_units='xy',
                   scale=1)

        # Plot level sets of covector field
        plt.contour(X, Y, G, levels=np.linspace(-2, 2, 10), colors='red')

        # Set plot limits
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        # Set labels and title
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Level Sets of Covector Field')

        # Show plot
        plt.grid(True)
        plt.show()


    # co_field()
    """
    x, k = sympy.var('x, k', real=True)

    diff_operator = DiffOperator(sympy.exp(-x**2))
    func = diff_operator.fourier()
    print(func.evalf(subs={k: 0.4}))   # 0.3654
    diff_operator = DiffOperator(sympy.sin(x))
    func = diff_operator.fourier()
    diff_operator = DiffOperator(sympy.cos(x))
    print(diff_operator.fourier())

    t, s = sympy.symbols('t, s', real=True)
    a = sympy.symbols('a', real=True)
    diff_operator = DiffOperator(sympy.exp(-a*t))
    func = diff_operator.laplace()
    print(func)
    print(f'Laplace (0.5, 0.2): {func.evalf(subs={s:0.5, a:0.2})}')
    diff_operator = DiffOperator(sympy.sqrt(a*t))
    func = diff_operator.laplace()
    print(func)
    print(f'Laplace (0.5, 0.2): {func.evalf(subs={s:0.5, a:0.2})}')
    r = CoordSys3D('r')
    this_expr = r.x*r.x*r.y*r.z
    diff_operator = DiffOperator(this_expr)
    grad_res = diff_operator.gradient()
    print(grad_res)
    print(diff_operator.curl(grad_res))
    div_res = diff_operator.divergence(r.i + r.j + r.k)
    print(div_res)
    curl_res = diff_operator.curl(r.i + r.j + r.k)
    print(curl_res)
    print(diff_operator.divergence(curl_res))
    """
