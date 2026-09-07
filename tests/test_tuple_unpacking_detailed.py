"""Gradient correctness for tuple-unpacking assignments.

Each case pins the analytically expected gradient; historically these were
investigation scripts that printed results, so a summing bug in the unpacking
adjoint would have gone unnoticed.
"""
import numpy as np
import pytest
from tangent import grad


def test_case_1_only_use_first():
    """Use only the first element of tuple unpacking."""

    def f(x):
        a, b = x ** 2, x * 3
        return a  # Only use 'a', not 'b'

    # d/dx(x^2) = 2x = 4; a summing bug would give 2x + 3 = 7.
    assert grad(f)(2.0) == pytest.approx(4.0)


def test_case_2_only_use_second():
    """Use only the second element of tuple unpacking."""

    def f(x):
        a, b = x ** 2, x * 3
        return b  # Only use 'b', not 'a'

    # d/dx(3x) = 3; a summing bug would give 2x + 3 = 7.
    assert grad(f)(2.0) == pytest.approx(3.0)


def test_case_3_weighted_sum():
    """Use both elements but with different weights."""

    def f(x):
        a, b = x ** 2, x * 3
        return 10 * a + 1 * b  # Weight 'a' much more

    # d/dx(10*x^2 + 3x) = 20x + 3 = 43 at x = 2.
    assert grad(f)(2.0) == pytest.approx(43.0)


def test_case_4_swap_order():
    """Swapping the target order must not change the gradient."""

    def f(x):
        b, a = x * 3, x ** 2  # Swapped order
        return a + b

    # d/dx(x^2 + 3x) = 2x + 3 = 7 at x = 2.
    assert grad(f)(2.0) == pytest.approx(7.0)


def test_case_5_triple_unpacking():
    """Three-way unpacking, using only the first element."""

    def f(x):
        a, b, c = x ** 2, x * 3, x + 1
        return a  # Only use first one

    # d/dx(x^2) = 4; summing all gradients would give 2x + 3 + 1 = 9.
    assert grad(f)(2.0) == pytest.approx(4.0)


def test_case_6_nested_tuple():
    """Nested tuple unpacking."""

    def f(x):
        (a, b), c = (x ** 2, x * 3), x + 1
        return a + c

    # d/dx(x^2 + x + 1) = 2x + 1 = 5 at x = 2.
    assert grad(f)(2.0) == pytest.approx(5.0)


def test_tuple_unpacking_matches_separate_assignments():
    """Tuple unpacking and separate assignments must agree."""

    def f_tuple(x):
        a, b = x ** 2, x * 3
        return a

    def f_separate(x):
        a = x ** 2
        b = x * 3
        return a

    assert np.isclose(grad(f_tuple)(2.0), grad(f_separate)(2.0))
