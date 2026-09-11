"""Compile-time shape/dtype checking (tangent.check_shapes).

Abstract interpretation on ShapedArray inputs applies NumPy's broadcasting and
linear-algebra rules without allocating data, raising a located ShapeError on a
rank/broadcast/matmul/reshape mismatch. Unmodeled ops yield an unknown shape,
so there are no false positives.
"""

import numpy as np
import pytest

import tangent
from tangent.errors import ShapeError
from tangent.shape_check import ShapedArray


# --- Functions under test (module level so tracebacks locate their lines). ---


def add_mismatch(a, b):
    c = a * 2.0
    return c + b


def matmul_mismatch(a, b):
    return a @ b


def reshape_mismatch(x):
    return np.reshape(x, (2, 5))


def valid_mlp(x, w):
    h = np.tanh(x @ w)
    return np.sum(h**2)


def valid_broadcast(a, b):
    return a * b + 1.0


def uses_len(x):
    s = 0.0
    for i in range(len(x)):
        s = s + x[i]
    return s


def uses_iter(x):
    t = 0.0
    for v in x:
        t = t + np.sum(v)
    return t


def uses_unmodeled(x):
    # np.sort has no shape rule here; must not raise (unknown shape, no guess).
    return np.sum(np.sort(x))


class TestErrorsCaught:
    def test_broadcast(self):
        with pytest.raises(ShapeError, match='broadcast'):
            tangent.check_shapes(add_mismatch, np.zeros(3), np.zeros(4))

    def test_matmul(self):
        with pytest.raises(ShapeError, match='matmul'):
            tangent.check_shapes(matmul_mismatch, np.zeros((3, 4)), np.zeros((5, 6)))

    def test_reshape(self):
        with pytest.raises(ShapeError, match='reshape'):
            tangent.check_shapes(reshape_mismatch, np.zeros((3, 3)))

    def test_error_locates_source_line(self):
        try:
            tangent.check_shapes(add_mismatch, np.zeros(3), np.zeros(4))
            assert False
        except ShapeError as e:
            assert 'test_shape_check.py' in str(e)
            assert 'c + b' in str(e)


class TestValid:
    def test_mlp(self):
        out = tangent.check_shapes(valid_mlp, np.zeros((8, 4)), np.zeros((4, 5)))
        assert out.shape == ()

    def test_matmul_output_shape(self):
        def mm(a, b):
            return a @ b

        assert tangent.check_shapes(mm, np.zeros((8, 4)), np.zeros((4, 5))).shape == (8, 5)

    def test_broadcast_no_false_positive(self):
        # (3,1) * (1,4) -> (3,4) is legal; must not raise.
        out = tangent.check_shapes(valid_broadcast, np.zeros((3, 1)), np.zeros((1, 4)))
        assert out.shape == (3, 4)

    def test_len_and_index(self):
        assert tangent.check_shapes(uses_len, np.zeros(5)).shape == ()

    def test_iteration(self):
        assert tangent.check_shapes(uses_iter, np.zeros((4, 3))).shape == ()

    def test_reduction_axis(self):
        def red(x):
            return np.sum(x, axis=1)

        assert tangent.check_shapes(red, np.zeros((3, 5))).shape == (3,)


class TestNoFalsePositives:
    def test_unmodeled_op_is_unknown_not_error(self):
        # Must complete without raising - an unmodeled op yields unknown shape.
        tangent.check_shapes(uses_unmodeled, np.zeros(5))

    def test_shaped_array_inputs_accepted(self):
        out = tangent.check_shapes(valid_mlp, ShapedArray((8, 4)), ShapedArray((4, 5)))
        assert out.shape == ()

    def test_dtype_propagates(self):
        def f(x):
            return x + 1.0

        out = tangent.check_shapes(f, np.zeros((3,), dtype=np.float32))
        assert out.shape == (3,)
