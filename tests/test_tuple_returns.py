"""Tuple returns and multi-argument gradients.

`grad` of a function returning a tuple treats the outputs as jointly seeded
(each output's adjoint is seeded with 1), so the result equals the gradient of
the *sum* of the outputs. These tests pin that behavior alongside multi-wrt
gradients and unpacking of tuple-returning callees.
"""
import numpy as np
import pytest
from tangent import grad


def test_return_tuple():
    """Gradient of a function that returns a tuple built from locals."""

    def f(x):
        a = x ** 2
        b = x * 3
        return a, b  # Return tuple

    # Jointly seeded outputs: d/dx(x^2 + 3x) = 2x + 3 = 7 at x = 2.
    assert grad(f)(2.0) == pytest.approx(7.0)


def test_multi_argument_gradient():
    """Gradient with respect to multiple arguments returns a tuple."""

    def f(x, y):
        return x ** 2 + y ** 2

    df = grad(f, wrt=(0, 1))
    result = df(2.0, 3.0)
    assert isinstance(result, tuple)
    assert result == pytest.approx((4.0, 6.0))


def test_unpack_function_return():
    """Unpacking the tuple return value of another function."""

    def g(x):
        return x ** 2, x * 3

    def f(x):
        a, b = g(x)
        return a + b

    # d/dx(x^2 + 3x) = 2x + 3 = 7 at x = 2.
    assert grad(f)(2.0) == pytest.approx(7.0)


def test_tuple_in_tuple_unpacking():
    """Unpacking where the RHS is an explicit tuple expression."""

    def f(x):
        a, b = (x ** 2, x * 3)  # Explicit tuple on RHS
        return a + b

    assert grad(f)(2.0) == pytest.approx(7.0)


def test_multiple_return_gradient():
    """Gradient of a direct multi-value return sums the seeded outputs."""

    def f(x):
        return x ** 2, x * 3  # Return two values

    result = grad(f)(2.0)
    assert np.isscalar(result) or np.shape(result) == ()
    assert result == pytest.approx(7.0)
