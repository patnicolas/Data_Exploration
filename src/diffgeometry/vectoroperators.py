__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from sympy.vector import CoordSys3D, VectorZero
from sympy import Expr
from typing import Any
from typing import Callable, Tuple, NoReturn, AnyStr, List
import numpy as np
from matplotlib.axes import Axes


class VectorOperators(object):
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

    def laplacian(self) -> VectorZero:
        from sympy.vector import divergence, gradient
        # Step 1 Compute the Gradient vector
        grad_f = self.gradient()

        # Step 2 Apply the divergence to the gradient
        return divergence(grad_f)

    def laplace(self):
        from sympy import laplace_transform
        return laplace_transform(self.expr, t, s, noconds=True)

    def fourier(self):
        from sympy import fourier_transform
        return fourier_transform(self.expr, x, k)

    @staticmethod
    def show_3D_function(f: Callable[[float, float, float], float], grid_units: np.array) -> NoReturn:
        """
            Method to display a function 
                 :param grad_f List of gradient component as functions
                 :param axis_values Values assigned to each axis
             """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d

        # Setup the grid with the appropriate units and boundary
        x, y, z = np.meshgrid(grid_units, grid_units, grid_units)
        # Apply the function f
        data = f(x, y, z)
        # Set up plot (labels, legend,..)
        ax: Axes = DiffOperator.__setup_3D_plot('3D Plot f(x,y,z) = x^2 + y^2 + z^2')

        # Display the data along x, y and z using scatter plot
        ax.scatter(x, y, z, c=data)
        plt.show()

    @staticmethod
    def show_3D_gradient(grad_f: List[Callable[[float], float]], axis_values: np.array) -> NoReturn:
        """
            Method to display the gradient, given its components as vector fields.
            :param grad_f List of gradient component as functions
            :param axis_values Values assigned to each axis
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        assert(len(grad_f) == 3, f'Gradient has {len(grad_f)} components instead of 3')

        # Setup the grid with the appropriate units and boundary
        x, y, z = np.meshgrid(axis_values, axis_values, axis_values)
        ax = DiffOperator.__setup_3D_plot('3D Plot Gradient 2x.i + 2y.j + 2z.k')
        # Extract the gradient df/dx, df/dy and df/dz
        X = grad_f[0](x)
        Y = grad_f[1](y)
        Z = grad_f[2](z)
        # Display the gradient vectors as vector fields
        ax.quiver(x, y, z, X, Y, Z, length=1.5, color='grey', normalize=True)
        plt.show()

    @staticmethod
    def __setup_3D_plot(title: AnyStr) -> Axes:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        return ax
