"""Iteration either differentiates correctly or fails loudly - never silently.

Iterating a *named* active sequence has long been rewritten into an indexed
loop. Iterating a computed expression (``for v in x * 2``, ``np.flip(x)``, or
a list/tuple literal of active values) used to fall through that rewrite
unrewritten and unrejected, silently dropping gradients; such iterables are now
hoisted into a named intermediate and indexed the same way (see
``tangent/desugar.py``).

What cannot be indexed positionally is rejected with a clear error: dict views
(``d.values()``/``.keys()``/``.items()``) and set literals carrying active
values. Iterating a *constant* collection (``for i in [0, 1, 2]``) is a
legitimate fixed loop and is left alone, as is ``for i in range(n)``.
"""

import numpy as np
import pytest

import tangent
from tangent.errors import TangentParseError


class TestHoistedIterableIteration:
    """Computed iterables are hoisted and indexed, and differentiate correctly."""

    def test_list_literal_with_active_values(self):
        def f(x):
            total = 0.0
            for v in [x, x * 2]:
                total = total + v
            return total

        assert tangent.grad(f)(1.5) == pytest.approx(3.0)

    def test_tuple_literal_with_active_values(self):
        def f(x):
            total = 0.0
            for v in (x, x * 2):
                total = total + v
            return total

        assert tangent.grad(f)(1.5) == pytest.approx(3.0)

    def test_binop_expression_iterable(self):
        """`for v in x * 2.0` was the silent-zero case: neither rewritten nor
        rejected."""

        def f(x):
            s = 0.0
            for v in x * 2.0:
                s = s + v
            return s

        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(tangent.grad(f)(x), 2.0 * np.ones_like(x))

    def test_call_expression_iterable(self):
        def f(x):
            s = 0.0
            for v in np.flip(x):
                s = s + v * v
            return s

        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(tangent.grad(f)(x), 2.0 * x)

    def test_tuple_unpacking_target(self):
        def f(x):
            ps = []
            for i in range(len(x)):
                ps.append((x[i], x[i] * 2.0))
            s = 0.0
            for a, b in ps:
                s = s + a * b
            return s

        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(tangent.grad(f)(x), 4.0 * x)

    def test_expression_iterable_forward_and_second_order(self):
        def f(x):
            s = 0.0
            for v in x * 1.0:
                s = s + v * v
            return s

        x = np.array([1.0, 2.0, 3.0])
        assert tangent.autodiff(f, mode='forward')(x, np.ones_like(x)) == pytest.approx(
            float(np.sum(2.0 * x))
        )
        np.testing.assert_allclose(tangent.grad(tangent.grad(f))(x), 2.0 * np.ones_like(x))


class TestRejectedIteration:
    def test_set_literal_with_active_values(self):
        def f(x):
            total = 0.0
            for v in {x, x * 2}:
                total = total + v
            return total

        with pytest.raises(TangentParseError, match='set literal'):
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
            d = {'a': x, 'b': x**2}
            return sum(d.values())

        assert tangent.grad(f)(2.0) == pytest.approx(5.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
