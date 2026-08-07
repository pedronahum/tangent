"""Test suite for membership operators (``in`` / ``not in``).

Membership tests produce a non-differentiable boolean, exactly like the other
comparison operators (``<``, ``==``, ``is`` ...), so they are safe to use in
conditions that guide control flow. This suite exercises them in both reverse
and forward mode.
"""
import numpy as np
import pytest

import tangent


class TestMembershipReverseMode:
    """``in`` / ``not in`` used as branch guards under reverse-mode grad."""

    def test_in_tuple_member(self):
        def f(x):
            if x in (1.0, 2.0, 3.0):
                y = x * x
            else:
                y = x * 3.0
            return y

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(4.0)  # d(x*x)/dx = 2x

    def test_in_tuple_non_member(self):
        def f(x):
            if x in (1.0, 2.0, 3.0):
                y = x * x
            else:
                y = x * 3.0
            return y

        df = tangent.grad(f)
        assert df(5.0) == pytest.approx(3.0)  # d(3x)/dx = 3

    def test_not_in_tuple_non_member(self):
        def f(x):
            if x not in (1.0, 2.0):
                y = x ** 2
            else:
                y = x * 5.0
            return y

        df = tangent.grad(f)
        assert df(4.0) == pytest.approx(8.0)  # d(x^2)/dx = 2x

    def test_not_in_tuple_member(self):
        def f(x):
            if x not in (1.0, 2.0):
                y = x ** 2
            else:
                y = x * 5.0
            return y

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(5.0)  # d(5x)/dx = 5

    def test_in_list_literal(self):
        def f(x):
            valid = [10.0, 20.0]
            if x in valid:
                y = x * x
            else:
                y = x
            return y

        df = tangent.grad(f)
        assert df(10.0) == pytest.approx(20.0)

    def test_in_as_loop_guard(self):
        def f(x):
            result = 0.0
            for i in range(1, 4):
                if i in (2, 3):
                    result = result + x ** i
            return result

        # d/dx (x^2 + x^3) at x=2 = 2x + 3x^2 = 4 + 12 = 16
        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(16.0)


class TestMembershipForwardMode:
    """``in`` / ``not in`` guards under forward-mode grad."""

    def test_in_forward(self):
        def f(x):
            if x in (1.0, 2.0, 3.0):
                y = x * x
            else:
                y = x * 3.0
            return y

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(2.0, 1.0) == pytest.approx(4.0)  # seed derivative = 1.0

    def test_not_in_forward(self):
        def f(x):
            if x not in (1.0, 2.0):
                y = x ** 2
            else:
                y = x * 5.0
            return y

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(4.0, 1.0) == pytest.approx(8.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
