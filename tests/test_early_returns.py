"""Test suite for early returns (multiple / non-tail return statements).

Tangent requires single-exit form, so early returns are lifted into a result
variable with one trailing return:

    if x > 0:
        return x * x
    return x * 3.0
    # becomes
    if x > 0:
        __tangent_retval = x * x
    else:
        __tangent_retval = x * 3.0
    return __tangent_retval

Returns at the tail of if/else branches, fall-through returns, elif chains, and
nested conditionals are supported. A return inside a plain loop lowers into an
assignment plus a flag plus ``break`` (which loop_exit_desugar then lowers into
guard flags), with ``if __tangent_returning: return`` propagating the exit past
each enclosing loop. Only loops with an ``else`` clause reject returns (their
semantics depend on how the loop exited).
"""

import pytest

import tangent
from tangent.errors import TangentParseError


class TestEarlyReturns:
    def test_if_else_both_return(self):
        def f(x):
            if x > 0:
                return x * x
            else:
                return x * 3.0

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(4.0)  # x>0 -> 2x
        assert df(-2.0) == pytest.approx(3.0)  # else -> 3

    def test_if_return_fallthrough(self):
        def f(x):
            if x > 0:
                return x * x
            return x * 3.0

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(4.0)
        assert df(-2.0) == pytest.approx(3.0)

    def test_code_before_return(self):
        def f(x):
            y = x * 2.0
            if x > 5:
                return y
            return y * x

        df = tangent.grad(f)
        assert df(3.0) == pytest.approx(12.0)  # y*x = 2x^2 -> 4x
        assert df(6.0) == pytest.approx(2.0)  # y = 2x -> 2

    def test_elif_chain(self):
        def f(x):
            if x < 0:
                return -x
            elif x < 1:
                return x * x
            else:
                return x * 3.0

        df = tangent.grad(f)
        assert df(-2.0) == pytest.approx(-1.0)
        assert df(0.5) == pytest.approx(1.0)
        assert df(2.0) == pytest.approx(3.0)

    def test_nested_conditionals(self):
        def f(x):
            if x > 0:
                if x > 10:
                    return x
                return x * x
            return x * 3.0

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(4.0)
        assert df(20.0) == pytest.approx(1.0)
        assert df(-2.0) == pytest.approx(3.0)

    def test_sequential_ifs(self):
        def f(x):
            if x > 10:
                return x
            if x > 5:
                return x * x
            return x * 3.0

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(3.0)
        assert df(7.0) == pytest.approx(14.0)
        assert df(20.0) == pytest.approx(1.0)

    def test_forward_mode(self):
        def f(x):
            if x > 0:
                return x * x
            return x * 3.0

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(2.0, 1.0) == pytest.approx(4.0)
        assert df(-2.0, 1.0) == pytest.approx(3.0)


class TestSingleReturnUnaffected:
    def test_plain_single_return(self):
        def f(x):
            y = x * x
            return y

        assert tangent.grad(f)(3.0) == pytest.approx(6.0)


class TestReturnInLoop:
    def test_return_inside_for_loop(self):
        # Formerly rejected; now lowered into flag + break. At x=1.5 the loop
        # returns x when i=2 > x (gradient 1); at x=5 it falls through to x*x.
        def f(x):
            for i in range(3):
                if i > x:
                    return x
            return x * x

        assert tangent.grad(f)(1.5) == pytest.approx(1.0)
        assert tangent.grad(f)(5.0) == pytest.approx(10.0)

    def test_return_inside_while_loop(self):
        def f(x):
            t = 0.0
            i = 0
            while i < 100:
                t = t + x
                if t > 5.0:
                    return t * t
                i = i + 1
            return t

        # At x=2: t reaches 6 after 3 iterations -> f = (3x)**2 -> f' = 18x.
        assert tangent.grad(f)(2.0) == pytest.approx(36.0)

    def test_return_in_nested_loop_exits_all_loops(self):
        def f(x):
            s = 0.0
            for i in range(5):
                for j in range(5):
                    s = s + x * x
                    if s > 20.0:
                        return s
            return s + x

        # At x=1.5: s accumulates 2.25 per step, exceeds 20 on step 9 -> s =
        # 9 * x**2 -> gradient 18x = 27.
        assert tangent.grad(f)(1.5) == pytest.approx(27.0)

    def test_forward_and_second_order(self):
        def f(x):
            t = 0.0
            i = 0
            while i < 100:
                t = t + x
                if t > 5.0:
                    return t * t
                i = i + 1
            return t

        assert tangent.autodiff(f, mode='forward')(2.0, 1.0) == pytest.approx(36.0)
        assert tangent.grad(tangent.grad(f))(2.0) == pytest.approx(18.0)

    def test_return_in_loop_with_else_rejected(self):
        def f(x):
            for i in range(3):
                if i > x:
                    return x
            else:
                x = x * 2.0
            return x * x

        with pytest.raises(TangentParseError):
            tangent.grad(f)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
