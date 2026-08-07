"""Broken iteration must fail loudly, not return wrong gradients.

Iterating a literal collection or a dict view binds the loop variable to values
that are never differentiated, so gradients flowing through those values were
silently dropped (e.g. ``for v in [x, x*2]`` returned 0). Such loops are now
rejected with a clear error when the iterable carries active (differentiated)
values.

Iterating a *constant* collection (``for i in [0, 1, 2]``) is a legitimate fixed
loop and is still allowed, as is ``for i in range(n)`` and iterating a NumPy
array bound to a variable.
"""
import numpy as np
import pytest

import tangent
from tangent.errors import TangentParseError


class TestRejectedIteration:
    def test_list_literal_with_active_values(self):
        def f(x):
            total = 0.0
            for v in [x, x * 2]:
                total = total + v
            return total

        with pytest.raises(TangentParseError):
            tangent.grad(f)

    def test_tuple_literal_with_active_values(self):
        def f(x):
            total = 0.0
            for v in (x, x * 2):
                total = total + v
            return total

        with pytest.raises(TangentParseError):
            tangent.grad(f)

    def test_dict_values_iteration(self):
        def f(x):
            d = {'a': x, 'b': x * 2}
            total = 0.0
            for v in d.values():
                total = total + v
            return total

        with pytest.raises(TangentParseError):
            tangent.grad(f)

    def test_dict_items_iteration(self):
        def f(x):
            d = {'a': x, 'b': x * 2}
            total = 0.0
            for k, v in d.items():
                total = total + v
            return total

        with pytest.raises(TangentParseError):
            tangent.grad(f)

    def test_dict_keys_iteration(self):
        def f(x):
            d = {'a': x, 'b': x * 2}
            total = 0.0
            for k in d.keys():
                total = total + d[k]
            return total

        with pytest.raises(TangentParseError):
            tangent.grad(f)


class TestStillAllowedIteration:
    """Legitimate loops must keep working."""

    def test_range(self):
        def f(x):
            total = 0.0
            for i in range(3):
                total = total + x
            return total

        assert tangent.grad(f)(2.0) == pytest.approx(3.0)

    def test_constant_list(self):
        def f(x):
            total = 0.0
            for i in [0, 1, 2]:
                total = total + x
            return total

        assert tangent.grad(f)(2.0) == pytest.approx(3.0)

    def test_constant_value_list(self):
        def f(x):
            total = 0.0
            for c in [1.0, 2.0, 3.0]:
                total = total + x * c
            return total

        # x * (1 + 2 + 3) = 6x -> gradient 6
        assert tangent.grad(f)(2.0) == pytest.approx(6.0)

    def test_iterate_numpy_array(self):
        def f(a):
            total = 0.0
            for v in a:
                total = total + v
            return total

        grad = tangent.grad(f)(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(grad, [1.0, 1.0, 1.0])

    def test_sum_values_still_works(self):
        # sum(d.values()) is desugared before the loop check and must keep working.
        def f(x):
            d = {'a': x, 'b': x ** 2}
            return sum(d.values())

        assert tangent.grad(f)(2.0) == pytest.approx(5.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
