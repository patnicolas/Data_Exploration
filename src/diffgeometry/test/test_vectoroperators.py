import unittest
import sys
import path
from sympy import symbols, Matrix, sqrt, cos, sin, atan2
from sympy.vector import CoordSys3D
from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField

directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
print(sys.path)
from vectoroperators import VectorOperators


class TestVectorOperators(unittest.TestCase):

    def test_2D_gradient_field(self):
        import sympy
        import math
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

    def test_3D_gradient_field(self):
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


    def test_covector(self):
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

    def test_covector_field(self):
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

    def test_laplacian(self):
        r = CoordSys3D('r')
        this_expr = r.x * r.x + r.y * r.y + r.z * r.z
        vector_operators = VectorOperators(this_expr)
        laplace_op = vector_operators.laplacian()
        print(laplace_op)  # 6

        this_expr = r.x * r.x * r.y * r.y * r.z * r.z
        vector_operators = VectorOperators(this_expr)
        laplace_op = vector_operators.laplacian()
        print(laplace_op)  # 2*r.x**2*r.y**2 + 2*r.x**2*r.z**2 + 2*r.y**2*r.z**2

    def test_divergence(self):
        r = CoordSys3D('r')
        func = r.x * r.x * r.y * r.z
        vector_operators = VectorOperators(func)

        divergence = vector_operators.divergence(r.i + r.j + r.k)
        print(divergence)

    def test_curl(self):
        r = CoordSys3D('r')
        func = r.x * r.x * r.y * r.z
        vector_operators = VectorOperators(func)

        curl = vector_operators.curl(r.i + r.j + r.k)
        print(curl)
        print(vector_operators.divergence(curl))


if __name__ == '__main__':
    unittest.main()