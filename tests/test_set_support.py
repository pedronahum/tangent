"""Test suite for set literal support.

Sets are unordered, non-differentiable collections. Their most common use in
numeric code is as a membership guard (``if x in {1, 2, 3}``), which steers
control flow without contributing to the gradient - exactly like a tuple or list
used the same way. Set literals are therefore allowed and carried through the
transformation with no adjoint.
"""
import pytest

import tangent


class TestSetMembershipReverseMode:
    def test_in_set_member(self):
        def f(x):
            if x in {1.0, 2.0, 3.0}:
                y = x * x
            else:
                y = x * 3.0
            return y

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(4.0)

    def test_in_set_non_member(self):
        def f(x):
            if x in {1.0, 2.0, 3.0}:
                y = x * x
            else:
                y = x * 3.0
            return y

        df = tangent.grad(f)
        assert df(5.0) == pytest.approx(3.0)

    def test_not_in_set(self):
        def f(x):
            if x not in {2.0, 4.0}:
                y = x ** 2
            else:
                y = x * 7.0
            return y

        df = tangent.grad(f)
        assert df(3.0) == pytest.approx(6.0)   # 3 not in set -> 2x
        assert df(2.0) == pytest.approx(7.0)   # 2 in set -> 7

    def test_set_bound_to_variable(self):
        def f(x):
            valid = {10.0, 20.0}
            if x in valid:
                y = x * x
            else:
                y = x
            return y

        df = tangent.grad(f)
        assert df(10.0) == pytest.approx(20.0)

    def test_set_as_loop_guard(self):
        def f(x):
            total = 0.0
            for i in range(5):
                if i in {1, 3}:
                    total = total + x ** 2
            return total

        # 2 active iterations -> 2 * x^2 -> gradient 4x = 8 at x = 2
        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(8.0)

    def test_unused_set_literal(self):
        def f(x):
            s = {1, 2, 3}
            return x * x

        df = tangent.grad(f)
        assert df(3.0) == pytest.approx(6.0)

    def test_set_of_active_values_is_non_differentiable(self):
        # A set built from active variables is still just a non-differentiable
        # collection; it must not crash and the gradient flows through the
        # selected branch only.
        def f(x):
            markers = {x, x + 1}
            if 2.0 in markers:
                y = x * x
            else:
                y = x * 3.0
            return y

        df = tangent.grad(f)
        assert df(3.0) == pytest.approx(3.0)  # 2 not in {3,4} -> d(3x) = 3


class TestSetForwardMode:
    def test_in_set_forward(self):
        def f(x):
            if x in {1.0, 2.0, 3.0}:
                y = x * x
            else:
                y = x * 3.0
            return y

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(2.0, 1.0) == pytest.approx(4.0)
        assert df(5.0, 1.0) == pytest.approx(3.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
