"""Online (Stumm-Walther) checkpointing for while-loops of unknown length.

`with tangent.checkpoint():` around a while-loop, or `grad(..., checkpoint=True)`,
keeps loop-carried-state snapshots under a fixed budget, geometrically thinned
as the loop runs (tangent.online_store). The backward pass restores each
surviving snapshot in reverse, replays its segment with taping, and consumes
it - O(budget) peak tape memory for any trip count, gradients identical to the
fully-taped path.
"""

import tracemalloc

import numpy as np
import pytest

import tangent

from utils import numeric_grad


def while_annotated(x, lim):
    s = 0.0
    i = 0
    with tangent.checkpoint():
        while i < lim:
            s = s * 0.9 + x * x
            i = i + 1
    return s


def while_plain(x, lim):
    s = 0.0
    i = 0
    while i < lim:
        s = s * 0.9 + x * x
        i = i + 1
    return s


def while_break(x, lim):
    s = 0.0
    i = 0
    with tangent.checkpoint():
        while i < lim:
            s = s * 0.9 + x * x
            if s > 3.0:
                break
            i = i + 1
    return s


def while_break_plain(x, lim):
    s = 0.0
    i = 0
    while i < lim:
        s = s * 0.9 + x * x
        if s > 3.0:
            break
        i = i + 1
    return s


def while_array(x, lim):
    s = np.zeros(4)
    i = 0
    with tangent.checkpoint():
        while i < lim:
            s = s * 0.95 + x * x
            i = i + 1
    return np.sum(s)


def while_array_plain(x, lim):
    s = np.zeros(4)
    i = 0
    while i < lim:
        s = s * 0.95 + x * x
        i = i + 1
    return np.sum(s)


class TestExactness:
    # Iteration counts straddle the thinning point (2 * default budget = 64):
    # the geometric thinning must not change the gradient.
    @pytest.mark.parametrize('lim', [0, 1, 2, 5, 63, 64, 65, 200, 1000])
    def test_scalar_identical_to_plain(self, lim):
        assert tangent.grad(while_annotated)(0.7, lim) == tangent.grad(while_plain)(0.7, lim)

    @pytest.mark.parametrize('lim', [1, 5, 100])
    def test_scalar_matches_fd(self, lim):
        g = tangent.grad(while_annotated)(0.7, lim)
        assert g == pytest.approx(numeric_grad(while_plain)(0.7, lim), rel=1e-4, abs=1e-6)

    def test_array_state(self):
        x = np.full(4, 0.5)
        for lim in (3, 50, 500):
            np.testing.assert_array_equal(
                tangent.grad(while_array)(x, lim), tangent.grad(while_array_plain)(x, lim)
            )

    def test_data_dependent_break(self):
        g = tangent.grad(while_break)(0.9, 500)
        assert g == tangent.grad(while_break_plain)(0.9, 500)
        assert g == pytest.approx(numeric_grad(while_break_plain)(0.9, 500), rel=1e-4)

    def test_global_flag_engages_online(self):
        # grad(checkpoint=True) also checkpoints while-loops.
        g = tangent.grad(while_plain, checkpoint=True)(0.7, 300)
        assert g == tangent.grad(while_plain)(0.7, 300)
        assert 'online_store' in tangent.grad(while_plain, checkpoint=True).__tangent_source__

    @pytest.mark.parametrize('opt', [True, False])
    def test_optimized_and_not(self, opt):
        assert tangent.grad(while_annotated, optimized=opt)(0.7, 40) == pytest.approx(
            tangent.grad(while_plain)(0.7, 40)
        )


class TestMemory:
    def test_peak_bounded_by_budget(self):
        # 3000 iterations of 500-float state: full tape holds ~3000 copies,
        # the online scheme at most 2 * budget = 64.
        x = np.full(500, 0.3)

        def big_annotated(x, lim):
            s = np.zeros(500)
            i = 0
            with tangent.checkpoint():
                while i < lim:
                    s = s * 0.999 + x * x
                    i = i + 1
            return np.sum(s)

        def big_plain(x, lim):
            s = np.zeros(500)
            i = 0
            while i < lim:
                s = s * 0.999 + x * x
                i = i + 1
            return np.sum(s)

        da, dp = tangent.grad(big_annotated), tangent.grad(big_plain)
        np.testing.assert_array_equal(da(x, 3000), dp(x, 3000))

        def peak(df):
            tracemalloc.start()
            df(x, 3000)
            tracemalloc.reset_peak()
            df(x, 3000)
            _, p = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return p

        assert peak(da) * 5 < peak(dp)

    def test_runtime_noop(self):
        assert while_annotated(0.7, 10) == while_plain(0.7, 10)


class TestOnlineHelpers:
    def test_store_thins_at_budget(self):
        snaps, seg, budget = [], 1, 4
        for i in range(40):
            if i % seg == 0:
                snaps, seg = tangent.online_store(snaps, i, ('s', i), seg, budget)
        # Never exceeds 2 * budget, and the interval has grown.
        assert len(snaps) <= 2 * budget
        assert seg > 1

    def test_segments_tile_exactly(self):
        for total in (1, 2, 7, 64, 65, 300):
            snaps, seg = [], 1
            for i in range(total):
                if i % seg == 0:
                    snaps, seg = tangent.online_store(snaps, i, ('s', i), seg, 8)
            spans = []
            for s in range(len(snaps)):
                start, length, _ = tangent.online_segment(snaps, s, total)
                assert length > 0
                spans.append((start, start + length))
            spans.reverse()
            assert spans[0][0] == 0
            assert spans[-1][1] == total
            for (a, b), (c, d) in zip(spans, spans[1:]):
                assert b == c
