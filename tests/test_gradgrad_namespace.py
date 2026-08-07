"""Grad-of-grad must resolve names in the generated function's own namespace.

Differentiating a gradient function a second time failed with 'TypeError: None'
(unresolved call) and then 'NameError: name np is not defined'. The generated
gradient function is wrapped by the caching layer with functools.wraps; the
wrapper is a plain function carrying Tangent's module globals, not the
numpy/tangent names the generated code references. `inspect` unwraps such
wrappers when reading source, but name resolution and namespace assembly did
not, so the source and the namespace disagreed.

Following the __wrapped__ chain for both resolution and namespace assembly fixes
grad-of-grad for scalar functions. (Second derivatives of array-valued
functions remain limited by a separate, pre-existing issue.)
"""
import math

import numpy as np
import pytest

import tangent


def _gradgrad(f):
    return tangent.grad(tangent.grad(f))


class TestScalarSecondDerivatives:
    def test_cubic(self):
        def f(x):
            return x ** 3

        # d2/dx2 x^3 = 6x = 12 at x = 2
        assert _gradgrad(f)(2.0) == pytest.approx(12.0)

    def test_quadratic(self):
        def f(x):
            return x * x

        assert _gradgrad(f)(5.0) == pytest.approx(2.0)

    def test_polynomial(self):
        def f(x):
            return x ** 4 + 2.0 * x ** 2

        # 12x^2 + 4 = 52 at x = 2
        assert _gradgrad(f)(2.0) == pytest.approx(52.0)

    def test_tanh(self):
        def f(x):
            return np.tanh(x)

        t = math.tanh(2.0)
        assert _gradgrad(f)(2.0) == pytest.approx(-2 * t * (1 - t * t))

    def test_exp(self):
        def f(x):
            return np.exp(x)

        assert _gradgrad(f)(1.0) == pytest.approx(math.exp(1.0))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
