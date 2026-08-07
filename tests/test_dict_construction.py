"""Test suite for local dictionary construction with string keys.

Previously, constructing a dict inside a differentiated function and reading it
back failed whenever the dict variable was named ``d``: that name collided with
Tangent's internal ``d[x]`` gradient-operator sentinel, so ``d['a']`` was
misinterpreted and the generated code contained undefined ``_`` placeholders.

String keys can never denote the gradient operator (which only ever indexes by a
variable name or a numeric constant), so string-keyed subscripts are now always
treated as genuine dictionary access, regardless of the variable's name.
"""
import pytest

import tangent


class TestDictNamedD:
    """Dicts named ``d`` must behave like any other dict variable."""

    def test_single_key(self):
        def f(x):
            d = {'a': x}
            return d['a']

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(1.0)

    def test_multi_key(self):
        def f(x):
            d = {'a': x, 'b': x ** 2}
            return d['a'] + d['b']

        # d/dx (x + x^2) = 1 + 2x = 5 at x = 2
        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(5.0)

    def test_three_keys_mixed_expressions(self):
        def f(x):
            d = {'p': x ** 3, 'q': x * 2, 'r': x}
            return d['p'] + d['q'] + d['r']

        # d/dx (x^3 + 2x + x) = 3x^2 + 3 = 15 at x = 2
        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(15.0)


class TestDictOtherNames:
    """The same programs with a differently named dict (regression guard)."""

    def test_multi_key_named_config(self):
        def f(x):
            config = {'a': x, 'b': x ** 2}
            return config['a'] + config['b']

        df = tangent.grad(f)
        assert df(3.0) == pytest.approx(1.0 + 2 * 3.0)

    def test_equivalent_to_separate_vars(self):
        def with_dict(x):
            d = {'a': x, 'b': x ** 2}
            return d['a'] + d['b']

        def with_vars(x):
            a = x
            b = x ** 2
            return a + b

        assert tangent.grad(with_dict)(4.0) == pytest.approx(
            tangent.grad(with_vars)(4.0))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
