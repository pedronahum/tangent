"""Test suite for ``sum(d.values())`` over local dictionaries.

``sum(d.values())`` is the common "reduce a dict of contributions" pattern
(e.g. summing loss terms). Since a dict literal has statically known keys, it is
desugared into a fold of the values, which reuses the fully-supported subscript
addition::

    sum({'a': p, 'b': q}.values())  ->  p + q
    sum(d.values())                 ->  d['a'] + d['b']   # d = {'a':.., 'b':..}

The rewrite fires only when the keys are known unambiguously: a dict literal, or
a local variable assigned exactly one constant-keyed dict literal and never
reassigned. Ambiguous cases (reassignment, parameter dicts) are left untouched.
"""
import pytest

import tangent


class TestSumValuesReverseMode:
    def test_two_values(self):
        def f(x):
            d = {'a': x, 'b': x ** 2}
            return sum(d.values())

        # d/dx (x + x^2) = 1 + 2x = 5 at x = 2
        assert tangent.grad(f)(2.0) == pytest.approx(5.0)

    def test_single_value(self):
        def f(x):
            d = {'a': x ** 3}
            return sum(d.values())

        assert tangent.grad(f)(2.0) == pytest.approx(12.0)

    def test_three_values(self):
        def f(x):
            d = {'a': x, 'b': x ** 2, 'c': x ** 3}
            return sum(d.values())

        # 1 + 2x + 3x^2 = 1 + 4 + 12 = 17 at x = 2
        assert tangent.grad(f)(2.0) == pytest.approx(17.0)

    def test_with_start_value(self):
        def f(x):
            d = {'a': x, 'b': x ** 2}
            return sum(d.values(), 0.0)

        assert tangent.grad(f)(2.0) == pytest.approx(5.0)

    def test_dict_literal_receiver(self):
        def f(x):
            return sum({'a': x, 'b': x * 3}.values())

        # d/dx (x + 3x) = 4
        assert tangent.grad(f)(2.0) == pytest.approx(4.0)

    def test_scaled_sum(self):
        def f(x):
            d = {'a': x, 'b': x ** 2}
            return 2.0 * sum(d.values())

        # 2 * (1 + 2x) = 10 at x = 2
        assert tangent.grad(f)(2.0) == pytest.approx(10.0)

    def test_equivalent_to_explicit_fold(self):
        def with_sum(x):
            d = {'a': x, 'b': x ** 2, 'c': x}
            return sum(d.values())

        def with_fold(x):
            d = {'a': x, 'b': x ** 2, 'c': x}
            return d['a'] + d['b'] + d['c']

        assert tangent.grad(with_sum)(3.0) == pytest.approx(
            tangent.grad(with_fold)(3.0))


class TestSumValuesForwardMode:
    def test_two_values_forward(self):
        def f(x):
            d = {'a': x, 'b': x ** 2}
            return sum(d.values())

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(2.0, 1.0) == pytest.approx(5.0)


class TestSumValuesSafety:
    """Ambiguous dicts must not be rewritten (no silent-wrong gradients)."""

    def test_reassigned_dict_not_rewritten(self):
        # A reassigned dict has ambiguous keys, so it is left untouched and
        # fails loudly rather than being silently mis-folded.
        def f(x):
            d = {'a': x}
            d = {'a': x, 'b': x ** 2}
            return sum(d.values())

        with pytest.raises(Exception):
            tangent.grad(f)(2.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
