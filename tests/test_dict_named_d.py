"""A local dictionary named ``d`` must not collide with the gradient sentinel.

Tangent's template layer uses ``d[x]`` to mean "gradient of x", so a user
variable literally named ``d`` used to collide: ``d['a']`` and especially
``d[1]`` (numeric keys) were misread as the gradient operator, producing invalid
code / DictConstructionError. A locally-assigned ``d`` is now alpha-renamed
before differentiation, resolving the collision for every key type.
"""
import pytest

import tangent


class TestLocalDictNamedD:
    def test_int_keys(self):
        def f(x):
            d = {1: x, 2: x ** 2}
            return d[1] + d[2]

        assert tangent.grad(f)(2.0) == pytest.approx(5.0)

    def test_string_keys(self):
        def f(x):
            d = {'a': x, 'b': x ** 2}
            return d['a'] + d['b']

        assert tangent.grad(f)(2.0) == pytest.approx(5.0)

    def test_int_key_dict_comprehension(self):
        def f(x):
            d = {i: x ** i for i in range(1, 3)}
            return d[1] + d[2]

        assert tangent.grad(f)(2.0) == pytest.approx(5.0)

    def test_numeric_key_arithmetic(self):
        def f(x):
            d = {0: x, 1: x * 2.0}
            return d[0] * d[1]

        # x * 2x = 2x^2 -> gradient 4x = 12 at x = 3
        assert tangent.grad(f)(3.0) == pytest.approx(12.0)

    def test_forward_mode(self):
        def f(x):
            d = {1: x, 2: x ** 2}
            return d[1] + d[2]

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(2.0, 1.0) == pytest.approx(5.0)


class TestDNameNotADict:
    """`d` used as an ordinary variable must keep working after renaming."""

    def test_scalar_named_d(self):
        def f(x):
            d = x * x
            return d + x

        # x^2 + x -> gradient 2x + 1 = 5 at x = 2
        assert tangent.grad(f)(2.0) == pytest.approx(5.0)

    def test_loop_variable_named_d(self):
        def f(x):
            total = 0.0
            for d in range(3):
                total = total + x
            return total

        assert tangent.grad(f)(2.0) == pytest.approx(3.0)


class TestNonLocalDNamedD:
    """A parameter dict named `d` (string keys) is untouched and still works."""

    def test_parameter_dict(self):
        def f(x, d={'a': 2.0}):
            return x * d['a']

        assert tangent.grad(f)(3.0) == pytest.approx(2.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
