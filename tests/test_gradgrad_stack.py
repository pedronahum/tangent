"""Second derivatives must not leak the gradient sentinel through the tape.

The adjoints of the tape operations (push/pop/push_stack/pop_stack) wrapped the
op_id tape marker in the gradient operator (``d[op_id]``). The op_id is a
non-differentiable string marker, so this was meaningless; in higher-order AD -
where these adjoints actually fire - it leaked an undefined ``d`` into the
generated code (``NameError: name 'd' is not defined``). It also broke
first-order gradients of functions that use explicit stack operations.

These tests differentiate twice in both optimized and unoptimized modes (the
``optimized`` argument is parametrized by conftest; the unoptimized path keeps
the push/pop operations, which is what exercises the tape adjoints) and check
the result against analytic second derivatives.
"""
import math

import numpy as np
import pytest

import tangent


def _gradgrad(f, optimized):
    return tangent.grad(tangent.grad(f, optimized=optimized), optimized=optimized)


def test_second_derivative_cubic(optimized):
    def f(x):
        return x ** 3

    assert _gradgrad(f, optimized)(2.0) == pytest.approx(12.0)


def test_second_derivative_quartic(optimized):
    def f(x):
        return x ** 4

    # d2/dx2 x^4 = 12 x^2 = 48 at x = 2
    assert _gradgrad(f, optimized)(2.0) == pytest.approx(48.0)


def test_second_derivative_tanh(optimized):
    def f(x):
        return np.tanh(x)

    t = math.tanh(2.0)
    assert _gradgrad(f, optimized)(2.0) == pytest.approx(-2 * t * (1 - t * t))


def test_second_derivative_product_of_powers(optimized):
    def f(x):
        return x ** 2 * x  # x^3

    assert _gradgrad(f, optimized)(3.0) == pytest.approx(18.0)  # 6x


def test_array_second_derivative():
    def f(x):
        return np.sum(np.tanh(x))

    # The grad-of-grad of sum(tanh(x)) gives the elementwise tanh''(x_i) (the
    # Hessian diagonal). Only the optimized path is exercised here; the
    # unoptimized path additionally hits a separate, pre-existing default-seed
    # shape limitation for array-valued second derivatives.
    ddf = tangent.grad(tangent.grad(f))
    x = np.array([0.5, 1.0, 1.5])
    got = ddf(x)
    t = np.tanh(x)
    expected = -2 * t * (1 - t * t)
    assert np.allclose(got, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
