"""Segment (sqrt-n) checkpointing: exact gradients at O(sqrt(n)) tape memory.

`tangent.grad(f, checkpoint=True)` runs eligible loops untaped, snapshotting
the loop-carried state every ceil(sqrt(n)) iterations; the adjoint restores
each snapshot in reverse, replays one segment with taping, and consumes it
(see grads.for_checkpointed / dfor_checkpointed and reverse_ad.visit_For).

Two properties are pinned here: checkpointed gradients are *identical* to the
fully-taped ones (replay is exact recomputation, not approximation), and peak
tape memory actually drops by ~sqrt(n) on a state-carrying loop.
"""

import tracemalloc

import numpy as np
import pytest

import tangent

from utils import numeric_grad


def state_loop(x):
    s = np.zeros(50)
    for i in range(400):
        s = s * 0.999 + x * x
    return np.sum(s)


def target_used(x):
    s = 0.0
    for i in range(150):
        s = s + x * float(i % 7)
    return s


def nonzero_stepped_range(x):
    s = 0.0
    for i in range(2, 500, 3):
        s = s * 0.99 + x * float(i)
    return s


def multi_state(x):
    a = 0.0
    b = 1.0
    for i in range(300):
        a = a + x * b
        b = b * 0.99
    return a


def with_branch(x):
    s = 0.0
    for i in range(250):
        if i % 2 == 0:
            s = s + x * x
        else:
            s = s - 0.1 * x
    return s


def subscript_state(x):
    h = np.zeros(3)
    for i in range(150):
        h[i % 3] = h[i % 3] + x * 0.5
    return np.sum(h)


def nested_loops(x):
    s = 0.0
    for i in range(120):
        for j in range(3):
            s = s + x * 0.01
    return s


def short_loop(x):
    # Below min_length: must silently fall back to the standard templates.
    s = 0.0
    for i in range(5):
        s = s + x * x
    return s


ALL = [target_used, nonzero_stepped_range, multi_state, with_branch, nested_loops, short_loop]


class TestExactness:
    @pytest.mark.parametrize('fn', ALL, ids=lambda f: f.__name__)
    @pytest.mark.parametrize('pt', [0.5, 2.0, -1.5])
    def test_checkpointed_equals_plain_equals_fd(self, fn, pt):
        g_plain = tangent.grad(fn)(pt)
        g_ckpt = tangent.grad(fn, checkpoint=True)(pt)
        # Replay is exact recomputation: bit-for-bit agreement, not approx.
        assert g_ckpt == g_plain
        assert g_ckpt == pytest.approx(numeric_grad(fn)(pt), rel=1e-4, abs=1e-6)

    def test_array_input(self):
        x = np.full(50, 0.5)
        g_plain = tangent.grad(state_loop)(x)
        g_ckpt = tangent.grad(state_loop, checkpoint=True)(x)
        np.testing.assert_array_equal(g_plain, g_ckpt)

    def test_subscript_mutated_state(self):
        # The snapshot must deep-copy arrays: h is mutated in place via
        # subscript stores, so a shallow snapshot would alias the live array.
        g_plain = tangent.grad(subscript_state)(np.array(2.0))
        g_ckpt = tangent.grad(subscript_state, checkpoint=True)(np.array(2.0))
        assert g_plain == g_ckpt == pytest.approx(75.0)

    @pytest.mark.parametrize('opt', [True, False])
    def test_optimized_and_not(self, opt):
        assert tangent.grad(multi_state, checkpoint=True, optimized=opt)(2.0) == pytest.approx(
            tangent.grad(multi_state)(2.0)
        )


class TestMemory:
    def test_peak_tape_memory_drops(self):
        x = np.full(50, 0.5)
        df_plain = tangent.grad(state_loop)
        df_ckpt = tangent.grad(state_loop, checkpoint=True)
        df_plain(x)
        df_ckpt(x)  # warm both

        def peak(df):
            tracemalloc.start()
            df(x)
            tracemalloc.reset_peak()
            df(x)
            _, p = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return p

        p_plain = peak(df_plain)
        p_ckpt = peak(df_ckpt)
        # 400 iterations of 50-float state: the full tape holds ~400 copies,
        # the checkpointed one ~2*sqrt(400)=40. Demand at least a 5x drop to
        # stay robust across allocator noise.
        assert p_ckpt * 5 < p_plain, 'peak %d vs %d - checkpointing saved no memory' % (
            p_ckpt,
            p_plain,
        )


class TestRuntimeHelpers:
    def test_segment_size(self):
        assert tangent.segment_size(0) == 1
        assert tangent.segment_size(1) == 1
        assert tangent.segment_size(400) == 20
        assert tangent.segment_size(401) == 21

    def test_segment_bounds_cover_exactly(self):
        for n in (1, 2, 5, 20, 399, 400, 401):
            seg = tangent.segment_size(n)
            spans = []
            for s in range(tangent.num_segments(n, seg)):
                start, length = tangent.segment_bounds(n, seg, s)
                assert length > 0
                spans.append((start, start + length))
            # Segments are emitted last-first and tile [0, n) exactly.
            spans.reverse()
            assert spans[0][0] == 0
            assert spans[-1][1] == n
            for (a, b), (c, d) in zip(spans, spans[1:]):
                assert b == c

    def test_snapshot_copies_mutables(self):
        arr = np.zeros(3)
        lst = [1.0]
        snap = tangent.snapshot((arr, lst, 5.0))
        arr[0] = 9.0
        lst.append(2.0)
        assert snap[0][0] == 0.0
        assert snap[1] == [1.0]
        assert snap[2] == 5.0
