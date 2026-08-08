"""Test suite for the built-in ``min`` and ``max`` (two-argument form).

``max(x, y)`` / ``min(x, y)`` are piecewise-linear: the incoming gradient flows
to whichever argument is selected, and 0 to the other (ties go to the first
argument). This covers common patterns like clamping (``min(x, cap)``) and ReLU
(``max(x, 0.0)``). These are reverse-mode gradients; the low-level three-or-more
argument form is not supported.
"""
import numpy as np
import pytest

import tangent


def _fd(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


class TestMaxBuiltin:
    def test_first_arg_selected(self):
        def f(x):
            return max(x, 1.0)

        assert tangent.grad(f)(2.0) == pytest.approx(1.0)   # x > 1

    def test_second_arg_selected(self):
        def f(x):
            return max(x, 5.0)

        assert tangent.grad(f)(2.0) == pytest.approx(0.0)   # 5 > x

    def test_relu(self):
        def relu(x):
            return max(x, 0.0)

        df = tangent.grad(relu)
        assert df(3.0) == pytest.approx(1.0)
        assert df(-2.0) == pytest.approx(0.0)

    def test_two_active_arguments(self):
        def f(x):
            return max(x, x * 0.5)

        # x > x*0.5 for x > 0 -> gradient 1; x < x*0.5 for x < 0 -> gradient 0.5
        assert tangent.grad(f)(2.0) == pytest.approx(1.0)
        assert tangent.grad(f)(-2.0) == pytest.approx(0.5)

    def test_in_expression(self):
        def f(x):
            return max(x, 1.0) * x

        # x > 1 -> x*x -> gradient 2x = 4
        assert tangent.grad(f)(2.0) == pytest.approx(4.0)


class TestMinBuiltin:
    def test_first_arg_selected(self):
        def f(x):
            return min(x, 5.0)

        assert tangent.grad(f)(2.0) == pytest.approx(1.0)   # x < 5

    def test_second_arg_selected(self):
        def f(x):
            return min(x, 1.0)

        assert tangent.grad(f)(2.0) == pytest.approx(0.0)   # 1 < x

    def test_clamp_on_expression(self):
        def f(x):
            return min(x * x, 4.0)

        df = tangent.grad(f)
        assert df(1.0) == pytest.approx(2.0)   # x^2 < 4 -> d(x^2) = 2x
        assert df(3.0) == pytest.approx(0.0)   # x^2 > 4 -> cap selected


class TestAgainstFiniteDifferences:
    @pytest.mark.parametrize('xval', [-2.0, -0.5, 0.5, 3.0])
    def test_relu_fd(self, xval):
        def f(x):
            return max(x, 0.0) * x

        assert tangent.grad(f)(xval) == pytest.approx(_fd(f, xval), abs=1e-4)


class TestHigherOrder:
    def test_second_derivative(self):
        def f(x):
            return max(x, 0.0) * x  # x^2 for x > 0

        ddf = tangent.grad(tangent.grad(f))
        assert ddf(3.0) == pytest.approx(2.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
