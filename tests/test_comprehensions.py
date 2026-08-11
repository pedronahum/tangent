"""Test suite for list, set and dict comprehensions.

Building a collection element-by-element cannot be differentiated correctly, but
a comprehension over a compile-time-constant iterable can be fully unrolled into
a plain literal - which the normal machinery handles. So list/set/dict
comprehensions over a constant ``range(...)`` or a list/tuple literal are
supported:

    {i: x ** i for i in range(1, 3)}   ->   {1: x ** 1, 2: x ** 2}
    {x * i for i in range(3)}          ->   {x * 0, x * 1, x * 2}
    [x * i for i in range(4)]          ->   [x * 0, x * 1, x * 2, x * 3]

List comprehensions additionally support ``if`` filters, evaluated at compile
time once the loop variable is substituted:

    [x * i for i in range(4) if i > 1] ->   [x * 2, x * 3]

Set/dict comprehensions do not support filters, and any comprehension that
cannot be unrolled (dynamic iterable, undecidable filter, etc.) either falls
through to the language fence and is rejected with a clear error, or is lowered
to an explicit loop - never silently mis-differentiated on the supported paths.
"""
import numpy as np
import pytest

import tangent
from tangent.errors import TangentParseError


class TestDictComprehension:
    def test_int_keys_subscript(self):
        def f(x):
            powers = {i: x ** i for i in range(1, 3)}
            return powers[1] + powers[2]

        # d/dx (x + x^2) = 1 + 2x = 5 at x = 2
        assert tangent.grad(f)(2.0) == pytest.approx(5.0)

    def test_string_keys(self):
        def f(x):
            table = {k: x for k in ['a', 'b']}
            return table['a'] + table['b']

        assert tangent.grad(f)(2.0) == pytest.approx(2.0)

    def test_with_sum_values(self):
        def f(x):
            terms = {i: x * i for i in range(1, 4)}
            return sum(terms.values())

        # x * (1 + 2 + 3) = 6x -> gradient 6
        assert tangent.grad(f)(2.0) == pytest.approx(6.0)

    def test_forward_mode(self):
        def f(x):
            powers = {i: x ** i for i in range(1, 3)}
            return powers[1] + powers[2]

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(2.0, 1.0) == pytest.approx(5.0)


class TestSetComprehension:
    def test_membership_guard_constant(self):
        def f(x):
            if x in {i for i in range(5)}:
                y = x * x
            else:
                y = x
            return y

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(4.0)   # 2 in {0..4} -> 2x
        assert df(9.0) == pytest.approx(1.0)   # 9 not in set -> 1

    def test_membership_guard_active_elements(self):
        def f(x):
            if 4.0 in {x * i for i in range(3)}:
                y = x * x
            else:
                y = x * 3.0
            return y

        # At x = 2, {0, 2, 4} contains 4 -> gradient of x*x is 2x = 4
        assert tangent.grad(f)(2.0) == pytest.approx(4.0)

    def test_set_comp_over_list_literal(self):
        def f(x):
            if 2.0 in {c * 2 for c in [1.0, 2.0]}:
                y = x * x
            else:
                y = x
            return y

        # {2, 4} contains 2 -> gradient of x*x is 2x = 6 at x = 3
        assert tangent.grad(f)(3.0) == pytest.approx(6.0)


class TestListComprehension:
    def test_no_filter_sum(self):
        def f(x):
            vals = [x * i for i in range(4)]
            return np.sum(vals)

        # x * (0 + 1 + 2 + 3) = 6x -> gradient 6
        assert tangent.grad(f)(2.0) == pytest.approx(6.0)

    def test_filtered_index(self):
        def f(x):
            vals = [x * i for i in range(4) if i > 1]
            return vals[0]

        # vals = [2x, 3x]; vals[0] = 2x -> gradient 2
        assert tangent.grad(f)(2.0) == pytest.approx(2.0)

    def test_filtered_sum_modulo(self):
        def f(x):
            vals = [x * i for i in range(5) if i % 2 == 0]
            return np.sum(vals)

        # i in {0, 2, 4}: x * (0 + 2 + 4) = 6x -> gradient 6
        assert tangent.grad(f)(2.0) == pytest.approx(6.0)

    def test_over_list_literal(self):
        def f(x):
            vals = [x * c for c in [1.0, 2.0, 3.0]]
            return np.sum(vals)

        # x * (1 + 2 + 3) = 6x -> gradient 6
        assert tangent.grad(f)(2.0) == pytest.approx(6.0)

    def test_forward_mode(self):
        def f(x):
            vals = [x * i for i in range(4) if i > 1]
            return vals[0]

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(2.0, 1.0) == pytest.approx(2.0)


class TestUnsupportedComprehensions:
    """Comprehensions that cannot be unrolled must be rejected, not miscomputed."""

    def test_dynamic_range_dict_comp_rejected(self):
        def f(x, n=3):
            d = {i: x for i in range(n)}
            return d[0]

        with pytest.raises(TangentParseError):
            tangent.grad(f)

    def test_filtered_set_comp_rejected(self):
        def f(x):
            if x in {i for i in range(5) if i > 2}:
                y = x
            else:
                y = x * x
            return y

        with pytest.raises(TangentParseError):
            tangent.grad(f)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
