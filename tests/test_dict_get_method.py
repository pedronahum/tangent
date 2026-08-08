"""Test suite for the dict ``.get()`` method.

``.get()`` is the common value-access dict method and maps cleanly onto
subscripting, so Tangent desugars it before differentiation:

    d.get(k)           ->  d[k]
    d.get(k, default)  ->  d[k] if k in d else default

These tests cover reverse mode (the fully supported path), plus the two-argument
form in forward mode (its ternary is lowered to an if-statement).
"""
import pytest

import tangent


class TestGetReverseMode:
    def test_get_existing_key(self):
        def f(x):
            d = {'a': x}
            return d.get('a')

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(1.0)

    def test_get_with_default_key_present(self):
        def f(x):
            d = {'a': x}
            return d.get('a', 0.0)

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(1.0)

    def test_get_with_default_key_missing(self):
        def f(x):
            d = {'a': x}
            return d.get('b', x * 2)

        # 'b' missing -> default x*2 -> gradient 2
        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(2.0)

    def test_get_multi_key(self):
        def f(x):
            d = {'a': x, 'b': x ** 2}
            return d.get('a') + d.get('b')

        # d/dx (x + x^2) = 1 + 2x = 5 at x = 2
        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(5.0)

    def test_get_in_loop_accumulates(self):
        def f(x):
            d = {'w': x ** 2}
            total = 0.0
            for i in range(3):
                total = total + d.get('w')
            return total

        # d/dx (3 * x^2) = 6x = 12 at x = 2
        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(12.0)

    def test_get_nested_in_expression(self):
        def f(x):
            d = {'a': x}
            return (d.get('a') ** 2) * 3.0

        # d/dx (3 * x^2) = 6x = 12 at x = 2
        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(12.0)

    def test_get_on_parameter_dict(self):
        def f(x, config={'lr': 0.5}):
            return x * config.get('lr')

        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(0.5)

    def test_get_equivalent_to_subscript(self):
        def with_get(x):
            d = {'a': x, 'b': x ** 2}
            return d.get('a') + d.get('b')

        def with_subscript(x):
            d = {'a': x, 'b': x ** 2}
            return d['a'] + d['b']

        assert tangent.grad(with_get)(3.0) == pytest.approx(
            tangent.grad(with_subscript)(3.0))


class TestGetForwardMode:
    def test_get_with_default_forward(self):
        # The two-argument form desugars to a ternary, which forward mode now
        # lowers to an if-statement.
        def f(x):
            d = {'a': x * x}
            return d.get('a', 0.0)

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(3.0, 1.0) == pytest.approx(6.0)  # d(x^2)/dx = 2x


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
