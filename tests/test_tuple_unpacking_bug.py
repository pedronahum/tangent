"""Regression tests for the historical tuple unpacking gradient bug.

Tuple unpacking (`a, b = x ** 2, x * 3`) must produce the same gradients as
the equivalent separate assignments.
"""
import numpy as np
import pytest
from tangent import grad


def test_tuple_unpacking_current_behavior():
    """Tuple unpacking with both values used."""

    def f(x):
        a, b = x ** 2, x * 3
        return a + b

    # d/dx(x^2 + 3x) = 2x + 3 = 7 at x = 2.
    assert grad(f)(2.0) == pytest.approx(7.0)


def test_separate_assignments_correct_behavior():
    """The equivalent separate assignments give the same gradient."""

    def f(x):
        a = x ** 2
        b = x * 3
        return a + b

    assert grad(f)(2.0) == pytest.approx(7.0)


def test_tuple_unpacking_multiple_uses():
    """Tuple unpacking where both values are used with different weights."""

    def f(x):
        a, b = x ** 2, x * 3
        return a * 2 + b * 5

    # d/dx(2x^2 + 15x) = 4x + 15 = 23 at x = 2.
    assert grad(f)(2.0) == pytest.approx(23.0)


def test_tuple_unpacking_with_array():
    """Tuple unpacking with array-valued expressions."""

    def f(x):
        a, b = x ** 2, x * 3
        return np.sum(a) + np.sum(b)

    result = grad(f)(np.array([1.0, 2.0]))
    # d/dx_i (sum(x^2) + sum(3x)) = 2*x_i + 3.
    assert np.allclose(result, np.array([5.0, 7.0]))
