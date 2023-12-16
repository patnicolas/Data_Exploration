
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
        div_vec = self.expr*base_vectors
        return divergence(div_vec, doit=True)

    def curl(self, base_vectors: Expr) -> VectorZero:
        from sympy.vector import curl
        curl_vec = self.expr*base_vectors
        return curl(curl_vec, doit=True)

    def laplace(self):
        from sympy import laplace_transform
        return laplace_transform(self.expr, t, s, noconds=True)

    def fourier(self):
        from sympy import fourier_transform
        return fourier_transform(self.expr, x, k)

if __name__ == '__main__':
    import sympy
    from sympy import exp, sin
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
