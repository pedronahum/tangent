"""Class method differentiation, in both notebook styles.

The notebooks define a small Polynomial class and differentiate a loss that
instantiates it; this pins that both a class defined locally (test style) and
one defined at module level (notebook style) differentiate correctly.
"""
import pytest

import tangent


class Polynomial:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x):
        return self.a * x ** 2 + self.b * x + self.c


def loss_with_module_class(x):
    poly = Polynomial(2.0, 3.0, 1.0)
    return poly.evaluate(x)


def test_local_class():
    """Class defined inside the test function (test style)."""

    class LocalPolynomial:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

        def evaluate(self, x):
            return self.a * x ** 2 + self.b * x + self.c

    def loss_with_class(x):
        poly = LocalPolynomial(2.0, 3.0, 1.0)
        return poly.evaluate(x)

    dloss = tangent.grad(loss_with_class)
    # d/dx(2x^2 + 3x + 1) = 4x + 3 = 23 at x = 5.
    assert dloss(5.0) == pytest.approx(23.0)


def test_module_level_class():
    """Class defined at module level (notebook style)."""
    dloss = tangent.grad(loss_with_module_class)
    assert dloss(5.0) == pytest.approx(4 * 5.0 + 3)
