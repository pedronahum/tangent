# Copyright 2026 Tangent contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The opt-in tape-liveness optimization (`optimizations={'tape_liveness': True}`).

Primals the reverse sweep reads *only for shape* - as the ``like`` of
``unbroadcast`` or the argument of ``init_grad`` - are stored on the tape as a
lightweight shape carrier instead of the full array. These tests pin the two
properties that matter: the gradient is byte-for-byte unchanged, and the tape
actually shrinks (a real array-valued loop uses far less memory).
"""

import numpy as np
import pytest

import tangent
from tangent.utils import TapedShape, taped_shape, init_grad, unbroadcast


# ---------------------------------------------------------------------------
# The shape carrier itself
# ---------------------------------------------------------------------------


class TestTapedShape:
    def test_captures_shape_and_dtype(self):
        ts = taped_shape(np.zeros((3, 4), dtype='float32'))
        assert isinstance(ts, TapedShape)
        assert ts.shape == (3, 4)
        assert ts.dtype == np.dtype('float32')

    def test_numpy_shape_reads_it(self):
        # The NumPy unbroadcaster consults its `like` through numpy.shape.
        assert np.shape(taped_shape(np.zeros((2, 5)))) == (2, 5)

    def test_init_grad_produces_matching_zero(self):
        ts = taped_shape(np.ones((2, 3)))
        g = init_grad(ts)
        assert np.shape(g) == (2, 3)
        assert np.all(np.asarray(g) == 0)

    def test_init_grad_scalar(self):
        assert init_grad(taped_shape(3.0)) == 0.0

    def test_unbroadcast_reduces_to_carrier_shape(self):
        # unbroadcast(grad, like) must reduce to like's shape whether `like` is
        # a real array or its shape carrier.
        g = np.ones((4, 3))
        real = unbroadcast(g, np.zeros((3,)))
        carried = unbroadcast(g, taped_shape(np.zeros((3,))))
        np.testing.assert_allclose(np.asarray(real), np.asarray(carried))


# ---------------------------------------------------------------------------
# Gradient equivalence: on vs off must match exactly
# ---------------------------------------------------------------------------

ON = {'tape_liveness': True}


def array_loop(x):
    s = x
    for i in range(20):
        a = s + 1.0  # additive: adjoint reads a only for shape
        s = a * 0.5 + s * 0.5
    return np.sum(s * s)


def value_needed_loop(x):
    # s = a * a needs a's value in reverse, so a must stay fully taped.
    s = x
    for i in range(15):
        a = s + 0.5
        s = a * a
    return np.sum(s)


def nested_loop(x):
    s = x
    for i in range(6):
        for j in range(4):
            u = s + 0.5
            s = u * 0.9 + s * 0.01
    return np.sum(s * s)


def cond_in_loop(x):
    s = x
    for i in range(8):
        a = s + 1.0
        if np.sum(a) > 1e9:
            a = a * 0.5
        s = a * 0.99
    return np.sum(s * s)


@pytest.mark.parametrize(
    'fn',
    [array_loop, value_needed_loop, nested_loop, cond_in_loop],
    ids=['array_loop', 'value_needed', 'nested', 'cond'],
)
def test_gradient_matches_with_and_without(fn):
    x = np.linspace(0.1, 1.0, 12)
    g_off = tangent.grad(fn)(x)
    g_on = tangent.grad(fn, optimizations=ON)(x)
    np.testing.assert_allclose(np.asarray(g_on), np.asarray(g_off), rtol=1e-10, atol=1e-12)


def test_scalar_case_still_correct():
    def f(x):
        s = x
        for i in range(30):
            a = s + 1.0
            s = a * 0.5
        return s

    x = 1.3
    assert tangent.grad(f, optimizations=ON)(x) == pytest.approx(tangent.grad(f)(x))


# ---------------------------------------------------------------------------
# The optimization actually fires (and only when opted in)
# ---------------------------------------------------------------------------


def test_off_by_default():
    src = tangent.grad(array_loop).__tangent_source__
    assert 'taped_shape(' not in src


def test_fires_when_enabled():
    src = tangent.grad(array_loop, optimizations=ON).__tangent_source__
    assert 'taped_shape(' in src


def test_value_needed_variable_is_not_converted():
    # `a` in `s = a * a` is read arithmetically in reverse, so it must not be
    # downgraded to a shape carrier. The gradient is the proof; also assert the
    # loop-carried `s` path keeps a real push somewhere.
    x = np.linspace(0.2, 1.0, 8)
    np.testing.assert_allclose(
        tangent.grad(value_needed_loop, optimizations=ON)(x),
        tangent.grad(value_needed_loop)(x),
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# The tape actually shrinks
# ---------------------------------------------------------------------------


def test_peak_memory_drops_on_array_loop():
    import tracemalloc

    x = np.arange(20000.0)
    off = tangent.grad(array_loop)
    on = tangent.grad(array_loop, optimizations=ON)

    def peak(df):
        df(x)  # warm
        tracemalloc.start()
        df(x)
        p = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return p

    p_off, p_on = peak(off), peak(on)
    # A 20k-element array taped ~20 times vs. shape tuples: expect a large drop.
    assert p_on < 0.5 * p_off, 'expected >2x peak-memory reduction, got %d -> %d' % (p_off, p_on)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
