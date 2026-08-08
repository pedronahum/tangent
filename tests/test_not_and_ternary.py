"""Gradient correctness for the `not` operator and ternary expressions.

`not` yields a non-differentiable boolean: it is valid in conditions (its result
steers control flow) and, when it appears as a value, simply has no adjoint -
previously that value form raised "unknown unary operator" in reverse mode.

Conditional (ternary) expressions ``a if cond else b`` differentiate through the
branch selected at runtime.
"""
import pytest

import tangent


class TestNotOperator:
    def test_not_as_value_is_non_differentiable(self):
        def f(x):
            return not x

        # `not x` is a bool; its gradient is zero, and it must not crash.
        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(0.0)

    def test_not_in_condition(self):
        def f(x):
            if not (x > 5):
                y = x * x
            else:
                y = x
            return y

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(4.0)   # not(2>5) True -> 2x = 4
        assert df(8.0) == pytest.approx(1.0)   # not(8>5) False -> 1

    def test_not_as_gating_flag(self):
        def f(x):
            flag = not (x > 0)
            if flag:
                y = x * 2
            else:
                y = x * x
            return y

        df = tangent.grad(f)
        assert df(-3.0) == pytest.approx(2.0)  # flag True -> d(2x) = 2
        assert df(3.0) == pytest.approx(6.0)   # flag False -> d(x^2) = 2x = 6

    def test_not_in_condition_forward_mode(self):
        def f(x):
            if not (x > 5):
                y = x * x
            else:
                y = x
            return y

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(2.0, 1.0) == pytest.approx(4.0)
        assert df(8.0, 1.0) == pytest.approx(1.0)


class TestTernary:
    def test_ternary_selects_branch(self):
        def f(x):
            return x ** 2 if x > 1 else x ** 3

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(4.0)    # x>1 -> d(x^2) = 2x = 4
        assert df(0.5) == pytest.approx(0.75)   # else -> d(x^3) = 3x^2 = 0.75

    def test_ternary_in_assignment(self):
        def f(x):
            y = (x * x) if x > 0 else (x * 3.0)
            return y

        df = tangent.grad(f)
        assert df(3.0) == pytest.approx(6.0)
        assert df(-3.0) == pytest.approx(3.0)

    def test_ternary_constant_branches(self):
        def f(x):
            return 1.0 if x else 2.0

        # Constant branches -> zero gradient (must not crash).
        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(0.0)

    def test_ternary_forward_mode(self):
        # Forward mode lowers ternaries to if-statements, so they differentiate.
        def f(x):
            y = (x * x) if x > 0 else (x * 3.0)
            return y

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(2.0, 1.0) == pytest.approx(4.0)   # x>0 -> d(x^2) = 2x
        assert df(-2.0, 1.0) == pytest.approx(3.0)  # else -> d(3x) = 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
