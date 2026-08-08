"""Test suite for ``for a, b in zip(xs, ys)`` loops.

A tuple-target ``zip`` loop is desugared into an indexed range loop::

    for a, b in zip(xs, ys):    ->    for i in range(len(xs)):
        <body>                            a = xs[i]
                                          b = ys[i]
                                          <body>

This supports the common dot-product / paired-iteration patterns over constant
lists and array variables, N-way zips, and both reverse and forward mode.
"""
import numpy as np
import pytest

import tangent


class TestZipReverseMode:
    def test_constant_lists(self):
        def f(x):
            total = 0.0
            for a, b in zip([1.0, 2.0], [3.0, 4.0]):
                total = total + x * a * b
            return total

        # x * (1*3 + 2*4) = 11x -> gradient 11
        assert tangent.grad(f)(2.0) == pytest.approx(11.0)

    def test_dot_product_active_array(self):
        def f(v):
            w = np.array([1.0, 2.0, 3.0])
            total = 0.0
            for a, b in zip(v, w):
                total = total + a * b
            return total

        # dot(v, w) -> gradient wrt v is w
        grad = tangent.grad(f)(np.array([1.0, 1.0, 1.0]))
        assert np.allclose(grad, [1.0, 2.0, 3.0])

    def test_zip_array_with_itself(self):
        def f(v):
            total = 0.0
            for a, b in zip(v, v):
                total = total + a * b
            return total

        # sum(v_i^2) -> gradient 2*v
        grad = tangent.grad(f)(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(grad, [2.0, 4.0, 6.0])

    def test_three_way_zip(self):
        def f(x):
            total = 0.0
            for a, b, c in zip([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]):
                total = total + x * a * b * c
            return total

        # x * (1*3*5 + 2*4*6) = 63x -> gradient 63
        assert tangent.grad(f)(2.0) == pytest.approx(63.0)


class TestZipForwardMode:
    def test_constant_lists_forward(self):
        def f(x):
            total = 0.0
            for a, b in zip([1.0, 2.0], [3.0, 4.0]):
                total = total + x * a * b
            return total

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(2.0, 1.0) == pytest.approx(11.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
