"""The same construct must behave the same way in every differentiation mode.

Pins the reject-or-be-correct principle along the mode axis: an op without a
registered derivative raises the clean NotImplemented error in BOTH modes
(never an opaque crash from recursing into NumPy internals), and trivial
builtins (float/int/abs/min/max casts and comparisons-of-two) have symmetric
reverse and forward rules.

Background: NumPy 2.x wraps most public functions in _ArrayFunctionDispatcher,
which the unimplemented-op scan did not recognize, so UNIMPLEMENTED_ADJOINTS /
UNIMPLEMENTED_TANGENTS were nearly empty and unregistered ops crashed with
"'_ArrayFunctionDispatcher' object has no attribute '__globals__'" instead of
a clean error.
"""

import numpy as np
import pytest

import tangent
from tangent import grads, tangents
from tangent.errors import ForwardNotImplementedError, ReverseNotImplementedError

from utils import numeric_grad


def uses_sort(x):
    return np.sum(np.sort(x))


def uses_median(x):
    return np.median(x) * 2.0


def float_active(x):
    return float(x) * 3.0


def int_active(x):
    return x * x + int(x)


def float_of_counter(x):
    s = 0.0
    for i in range(3):
        s = s + x * float(i + 1)
    return s


def counter_arithmetic(x):
    s = 0.0
    for i in range(3):
        s = s + x * i
    return s


def builtin_absminmax(x):
    return abs(x) + min(x, 2.0) + max(x, -1.0)


class TestCleanErrorsForUnregisteredOps:
    def test_unimplemented_sets_are_populated(self):
        # The dispatcher fix must keep these sets meaningfully populated; a
        # regression here silently reverts every clean error below to a crash.
        assert len(grads.UNIMPLEMENTED_ADJOINTS) > 300
        assert len(tangents.UNIMPLEMENTED_TANGENTS) > 300
        assert np.sort in grads.UNIMPLEMENTED_ADJOINTS
        # Implemented ops must not be swept in.
        assert np.flip not in grads.UNIMPLEMENTED_ADJOINTS
        assert np.cumsum not in grads.UNIMPLEMENTED_ADJOINTS

    @pytest.mark.parametrize('fn', [uses_sort, uses_median])
    def test_reverse_clean_error(self, fn):
        with pytest.raises(ReverseNotImplementedError):
            tangent.grad(fn)

    @pytest.mark.parametrize('fn', [uses_sort, uses_median])
    def test_forward_clean_error(self, fn):
        with pytest.raises(ForwardNotImplementedError):
            tangent.autodiff(fn, mode='forward')


class TestBuiltinModeParity:
    @pytest.mark.parametrize(
        'fn,pt',
        [
            (float_active, 2.0),
            (float_of_counter, 2.0),
            (counter_arithmetic, 2.0),
            (builtin_absminmax, 0.5),
            (builtin_absminmax, 3.0),
            (builtin_absminmax, -2.0),
        ],
    )
    def test_reverse_equals_forward_equals_fd(self, fn, pt):
        fd = numeric_grad(fn)(pt)
        assert tangent.grad(fn)(pt) == pytest.approx(fd, rel=1e-5, abs=1e-7)
        assert tangent.autodiff(fn, mode='forward')(pt, 1.0) == pytest.approx(
            fd, rel=1e-5, abs=1e-7
        )

    def test_int_truncation_zero_derivative(self):
        # d/dx [x*x + int(x)] = 2x almost everywhere.
        assert tangent.grad(int_active)(2.5) == pytest.approx(5.0)
        assert tangent.autodiff(int_active, mode='forward')(2.5, 1.0) == pytest.approx(5.0)

    def test_second_order_through_float(self):
        assert tangent.grad(tangent.grad(float_active))(2.0) == pytest.approx(0.0)
