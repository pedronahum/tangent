"""break/continue differentiate exactly through the guard-flag lowering.

Historically `break` *miscomputed* gradients (the tape replayed the wrong
number of iterations), then both statements were rejected up front. They are
now lowered by tangent/loop_exit_desugar.py into flag assignments and guarded
bodies - constructs the AD core differentiates in both modes - so the gradient
counts exactly the iterations that executed. The docstring there shows the
lowering; every gradient here is pinned analytically and (where the function is
smooth in x) against finite differences.
"""

import numpy as np
import pytest

import tangent

from utils import numeric_grad


def break_in_for(x):
    total = 0.0
    for i in range(10):
        total = total + x * x
        if i >= 2:
            break
    return total  # 3 * x**2


def break_in_while(x):
    total = 0.0
    i = 0
    while i < 10:
        total = total + x * x
        i = i + 1
        if i >= 3:
            break
    return total  # 3 * x**2


def data_dependent_break(x):
    # The number of executed iterations depends on x itself.
    total = 0.0
    i = 0
    while i < 50:
        total = total + x
        if total > 5.0:
            break
        i = i + 1
    return total


def continue_in_for(x):
    total = 0.0
    for i in range(6):
        if i == 2:
            continue
        total = total + x * x
    return total  # 5 * x**2


def break_over_sequence(x):
    total = 0.0
    for v in x:
        if v < 0.0:
            break
        total = total + v * v
    return total


def nested_break_continue(x):
    total = 0.0
    for i in range(4):
        for j in range(4):
            if j > i:
                break
            total = total + x * x
        if i >= 2:
            continue
        total = total + x
    return total


def statements_after_break_are_dead(x):
    total = 0.0
    for i in range(5):
        if i == 3:
            break
            total = total + x * 100.0  # unreachable
        total = total + x
    return total  # 3 iterations before the break-iteration adds nothing


class TestReverse:
    @pytest.mark.parametrize('pt', [0.5, 2.0, -1.5])
    @pytest.mark.parametrize(
        'fn',
        [
            break_in_for,
            break_in_while,
            continue_in_for,
            nested_break_continue,
            statements_after_break_are_dead,
        ],
    )
    def test_matches_finite_differences(self, fn, pt):
        assert tangent.grad(fn)(pt) == pytest.approx(numeric_grad(fn)(pt), rel=1e-5, abs=1e-7)

    def test_break_counts_executed_iterations(self):
        assert tangent.grad(break_in_for)(2.0) == pytest.approx(12.0)
        assert tangent.grad(break_in_while)(2.0) == pytest.approx(12.0)

    def test_data_dependent_break(self):
        # At x=2: totals 2, 4, 6 -> breaks after 3 iterations; gradient 3.
        assert tangent.grad(data_dependent_break)(2.0) == pytest.approx(3.0)
        # At x=0.5: totals reach 5.5 > 5 on the 11th iteration.
        assert tangent.grad(data_dependent_break)(0.5) == pytest.approx(11.0)

    def test_break_over_active_sequence(self):
        x = np.array([1.0, 2.0, -3.0, 4.0])
        np.testing.assert_allclose(
            tangent.grad(break_over_sequence)(x), np.array([2.0, 4.0, 0.0, 0.0])
        )

    @pytest.mark.parametrize('opt', [True, False])
    def test_optimized_and_split_motion(self, opt):
        assert tangent.grad(break_in_for, optimized=opt)(2.0) == pytest.approx(12.0)
        df = tangent.autodiff(
            break_in_for,
            motion='split',
            mode='reverse',
            input_derivative=tangent.grad_util.INPUT_DERIVATIVE.DefaultOne,
        )
        assert df(2.0) == pytest.approx(12.0)


class TestForwardAndHigherOrder:
    def test_forward_mode(self):
        assert tangent.autodiff(break_in_for, mode='forward')(2.0, 1.0) == pytest.approx(12.0)
        assert tangent.autodiff(break_in_while, mode='forward')(2.0, 1.0) == pytest.approx(12.0)
        assert tangent.autodiff(continue_in_for, mode='forward')(2.0, 1.0) == pytest.approx(20.0)

    def test_forward_mode_over_sequence(self):
        x = np.array([1.0, 2.0, -3.0, 4.0])
        got = tangent.autodiff(break_over_sequence, mode='forward')(x, np.ones_like(x))
        assert got == pytest.approx(6.0)  # sum of [2, 4, 0, 0]

    def test_second_order(self):
        assert tangent.grad(tangent.grad(break_in_for))(2.0) == pytest.approx(6.0)
        assert tangent.grad(tangent.grad(break_in_while))(2.0) == pytest.approx(6.0)
        assert tangent.grad(tangent.grad(continue_in_for))(2.0) == pytest.approx(10.0)
