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

List comprehensions that cannot be unrolled are lowered into indexed loops
built on the differentiable `tangent.list_append` rebinding, so dynamic
iterables and runtime filters differentiate correctly (see
TestDynamicIterableComprehensions). Set/dict comprehensions do not support
filters, and comprehension forms the lowering cannot express (multiple
generators, tuple targets) fall through to the language fence and are rejected
with a clear TangentParseError - never silently mis-differentiated.
"""

import numpy as np
import pytest

import tangent
from tangent.errors import TangentParseError


class TestDictComprehension:
    def test_int_keys_subscript(self):
        def f(x):
            powers = {i: x**i for i in range(1, 3)}
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
            powers = {i: x**i for i in range(1, 3)}
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
        assert df(2.0) == pytest.approx(4.0)  # 2 in {0..4} -> 2x
        assert df(9.0) == pytest.approx(1.0)  # 9 not in set -> 1

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

    def test_multi_generator_listcomp_rejected(self):
        """Multiple generators cannot be lowered into a single indexed loop."""

        def f(x):
            vals = [u * v for u in x for v in x]
            return np.sum(vals)

        with pytest.raises(TangentParseError, match='form of list comprehension'):
            tangent.grad(f)

    def test_tuple_target_listcomp_rejected(self):
        def f(pairs):
            vals = [a + b for a, b in pairs]
            return vals[0]

        with pytest.raises(TangentParseError, match='form of list comprehension'):
            tangent.grad(f)


class TestDynamicIterableComprehensions:
    """Listcomps over runtime iterables lower into differentiable loops.

    These forms used to be rejected (and before that, silently returned zero
    gradients: the old `.append()`-loop lowering had a per-iteration binding
    the activity analysis could not see). They now lower onto the
    tangent.list_append rebinding primitives.
    """

    def test_dynamic_iterable_listcomp_assign(self):
        def f(x):
            vals = [v * 3.0 for v in x]
            return np.sum(vals)

        x = np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(tangent.grad(f)(x), 3.0 * np.ones_like(x))

    def test_dynamic_iterable_listcomp_in_return(self):
        """The `listcomp` corpus entry: comprehension in return position."""

        def f(x):
            return np.sum([v * 3.0 for v in x])

        x = np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(tangent.grad(f)(x), 3.0 * np.ones_like(x))

    def test_dynamic_range_listcomp(self):
        """range() over a runtime bound lowers like any other iterable."""

        def f(x, n=3):
            vals = [x * i for i in range(n)]
            return np.sum(vals)

        assert tangent.grad(f)(2.0, 4) == pytest.approx(0.0 + 1.0 + 2.0 + 3.0)

    def test_runtime_filter(self):
        def f(x):
            vals = [v * v for v in x if v > 0.0]
            s = 0.0
            for i in range(len(vals)):
                s = s + vals[i]
            return s

        x = np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(tangent.grad(f)(x), np.array([2.0, 0.0, 6.0]))

    def test_nested_dynamic_listcomp(self):
        def f(x):
            ys = [[u * v for u in x] for v in x]
            s = 0.0
            for i in range(len(ys)):
                for j in range(len(ys[i])):
                    s = s + ys[i][j]
            return s

        x = np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(tangent.grad(f)(x), 2.0 * np.sum(x) * np.ones_like(x))

    def test_loop_variable_does_not_leak_or_clobber(self):
        """The comprehension target is renamed, so it cannot clobber a user
        variable of the same name (Python comprehension scopes do not leak)."""

        def f(v):
            ys = [v * q for q in [v * 1.0, v * 2.0]]
            return ys[0] + ys[1] + v

        # 3v^2 + v -> 6v + 1
        assert tangent.grad(f)(2.0) == pytest.approx(13.0)

    def test_forward_and_second_order(self):
        def f(x):
            ys = [v * v for v in x]
            s = 0.0
            for i in range(len(ys)):
                s = s + ys[i]
            return s

        x = np.array([1.0, 2.0, 3.0])
        assert tangent.autodiff(f, mode='forward')(x, np.ones_like(x)) == pytest.approx(
            float(np.sum(2.0 * x))
        )
        np.testing.assert_allclose(tangent.grad(tangent.grad(f))(x), 2.0 * np.ones_like(x))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
