"""Test gradient accumulation for repeated subscript reads.

In reverse-mode AD, the gradient contribution to a container element must
accumulate across every read of that element. Previously the adjoint of a
subscript read *overwrote* the element's gradient (via ``update_grad_at_index``),
so when the same element was read more than once - e.g. ``a[0] * a[0]`` or a
subscript read inside a loop - only the last contribution survived, silently
producing gradients that were too small.

The adjoint now accumulates (``add_grad_at_index``), matching how Tangent
accumulates every other gradient.
"""
import numpy as np
import pytest

import tangent


class TestArrayElementAccumulation:
    """NumPy array elements read more than once."""

    def test_same_element_multiplied(self):
        def f(a):
            return a[0] * a[0]

        # d/da (a0^2) = [2*a0, 0]
        df = tangent.grad(f)
        grad = df(np.array([2.0, 5.0]))
        assert np.allclose(grad, [4.0, 0.0])

    def test_same_element_in_loop(self):
        def f(a):
            total = 0.0
            for i in range(3):
                total = total + a[0]
            return total

        df = tangent.grad(f)
        grad = df(np.array([2.0, 5.0, 7.0]))
        assert np.allclose(grad, [3.0, 0.0, 0.0])

    def test_two_elements_summed_repeatedly(self):
        def f(a):
            total = 0.0
            for i in range(4):
                total = total + a[0] + a[2]
            return total

        df = tangent.grad(f)
        grad = df(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(grad, [4.0, 0.0, 4.0])

    def test_element_read_once_unchanged(self):
        """Single reads must still be correct (regression guard)."""
        def f(a):
            return a[1] * 3.0

        df = tangent.grad(f)
        grad = df(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(grad, [0.0, 3.0, 0.0])


class TestDictValueAccumulation:
    """Dict values read more than once."""

    def test_dict_value_in_loop(self):
        def f(x):
            total = 0.0
            d = {'w': x ** 2}
            for i in range(3):
                total = total + d['w']
            return total

        # d/dx (3 * x^2) = 6x = 12 at x = 2
        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(12.0)

    def test_dict_value_read_twice(self):
        def f(x):
            d = {'w': x}
            return d['w'] * d['w']

        # d/dx (x^2) = 2x = 6 at x = 3
        df = tangent.grad(f)
        assert df(3.0) == pytest.approx(6.0)

    def test_multi_key_dict_in_loop(self):
        def f(x):
            total = 0.0
            d = {'a': x, 'b': x ** 2}
            for i in range(2):
                total = total + d['a'] + d['b']
            return total

        # d/dx (2 * (x + x^2)) = 2 * (1 + 2x) = 10 at x = 2
        df = tangent.grad(f)
        assert df(2.0) == pytest.approx(10.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
