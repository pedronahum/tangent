"""Test suite for ``for i, v in enumerate(...)`` loops.

Tangent iterates with a ``range(len(...))`` index and explicit subscripting.
A tuple-target ``enumerate`` loop is desugared into that form::

    for i, v in enumerate(seq):    ->    for i in range(len(seq)):
        <body>                                v = seq[i]
                                              <body>

This works for constant list/tuple literals and for array variables, with the
index and/or value used in the body, a start offset, and in both reverse and
forward mode.
"""
import numpy as np
import pytest

import tangent


class TestEnumerateReverseMode:
    def test_constant_list_value(self):
        def f(x):
            total = 0.0
            for i, c in enumerate([1.0, 2.0, 3.0]):
                total = total + x * c
            return total

        # x * (1 + 2 + 3) = 6x -> gradient 6
        assert tangent.grad(f)(2.0) == pytest.approx(6.0)

    def test_index_and_value_used(self):
        def f(x):
            total = 0.0
            for i, c in enumerate([1.0, 2.0, 3.0]):
                total = total + x * c * i
            return total

        # x * (1*0 + 2*1 + 3*2) = 8x -> gradient 8
        assert tangent.grad(f)(2.0) == pytest.approx(8.0)

    def test_active_array(self):
        def f(a):
            total = 0.0
            for i, v in enumerate(a):
                total = total + v * v
            return total

        # sum(a_i^2) -> gradient 2*a
        grad = tangent.grad(f)(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(grad, [2.0, 4.0, 6.0])

    def test_active_array_weighted_by_index(self):
        def f(a):
            total = 0.0
            for i, v in enumerate(a):
                total = total + v * i
            return total

        # sum(i * a_i) -> gradient [0, 1, 2]
        grad = tangent.grad(f)(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(grad, [0.0, 1.0, 2.0])

    def test_start_offset(self):
        def f(x):
            total = 0.0
            for i, c in enumerate([1.0, 2.0], 10):
                total = total + x * i
            return total

        # indices 10, 11 -> x * 21 -> gradient 21
        assert tangent.grad(f)(2.0) == pytest.approx(21.0)


class TestEnumerateForwardMode:
    def test_constant_list_forward(self):
        def f(x):
            total = 0.0
            for i, c in enumerate([1.0, 2.0, 3.0]):
                total = total + x * c
            return total

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        assert df(2.0, 1.0) == pytest.approx(6.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
