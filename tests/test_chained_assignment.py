"""Test suite for chained assignment (``a = b = expr``).

Tangent's core transforms handle only single-target assignments, so a chained
assignment previously raised "no support for chained assignment". Since every
target is bound to the same value, it is desugared into a first assignment plus
copies:

    a = b = c = expr   ->   a = expr; b = a; c = a

which the existing machinery differentiates correctly (gradients accumulate
across the copies).
"""
import pytest

import tangent


class TestChainedAssignment:
    def test_two_targets(self):
        def f(x):
            a = b = x * x
            return a + b

        # 2 * x^2 -> gradient 4x = 8 at x = 2
        assert tangent.grad(f)(2.0) == pytest.approx(8.0)

    def test_three_targets(self):
        def f(x):
            a = b = c = x ** 2
            return a + b + c

        # 3 * x^2 -> gradient 6x = 12 at x = 2
        assert tangent.grad(f)(2.0) == pytest.approx(12.0)

    def test_targets_used_differently(self):
        def f(x):
            p = q = x
            return p * 2 + q * 3

        # 2x + 3x = 5x -> gradient 5
        assert tangent.grad(f)(2.0) == pytest.approx(5.0)

    def test_expression_referencing_input(self):
        def f(x):
            a = b = x ** 3 + x
            return a - b

        # a - b = 0 -> gradient 0
        assert tangent.grad(f)(2.0) == pytest.approx(0.0)

    def test_forward_mode(self):
        def f(x):
            a = b = x * x
            return a + b

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(2.0, 1.0) == pytest.approx(8.0)

    def test_equivalent_to_separate_assignments(self):
        def chained(x):
            a = b = x ** 2
            return a + b

        def separate(x):
            a = x ** 2
            b = a
            return a + b

        assert tangent.grad(chained)(3.0) == pytest.approx(
            tangent.grad(separate)(3.0))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
