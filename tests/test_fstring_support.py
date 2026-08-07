"""Test suite for f-string (formatted string literal) support.

f-strings produce a non-differentiable string - typically a debug or assert
message - so they are safe to allow in differentiated functions, just like plain
string constants and comparison operators. The interpolated string is stored and
carried through the transformation but never contributes to the gradient.
"""
import pytest

import tangent


class TestFStringReverseMode:
    def test_unused_debug_string(self):
        def f(x):
            label = f"x is {x}"
            return x * x

        df = tangent.grad(f)
        assert df(3.0) == pytest.approx(6.0)

    def test_assert_message(self):
        def f(x):
            assert x > 0, f"need positive, got {x}"
            return x ** 2

        df = tangent.grad(f)
        assert df(4.0) == pytest.approx(8.0)

    def test_multiple_interpolations(self):
        def f(x):
            msg = f"value={x}, squared={x * x}, half={x / 2}"
            return x * 3.0

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(3.0)

    def test_format_spec(self):
        def f(x):
            msg = f"{x:.2f}"
            return x * 5.0

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(5.0)

    def test_fstring_inside_loop(self):
        def f(x):
            total = 0.0
            for i in range(3):
                dbg = f"iter {i}: {x}"
                total = total + x
            return total

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(3.0)

    def test_fstring_inside_branch(self):
        def f(x):
            if x > 0:
                msg = f"positive {x}"
                y = x * x
            else:
                y = x
            return y

        df = tangent.grad(f)
        assert df(3.0) == pytest.approx(6.0)

    def test_fstring_as_subexpression(self):
        def f(x):
            n = len(f"{x}")
            return x * x

        df = tangent.grad(f)
        assert df(3.0) == pytest.approx(6.0)


class TestFStringForwardMode:
    def test_unused_debug_string(self):
        def f(x):
            label = f"x is {x}"
            return x * x

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(3.0, 1.0) == pytest.approx(6.0)

    def test_assert_message(self):
        def f(x):
            assert x > 0, f"need positive, got {x}"
            return x ** 2

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(4.0, 1.0) == pytest.approx(8.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
