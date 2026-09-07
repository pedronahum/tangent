# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
"""Regression tests for tape-aware dead code elimination.

Gradient code communicates between the forward and backward pass through a
tape (`tangent.Stack`), with pushes and pops paired by an `op_id` string.
Differentiating gradient code AGAIN (higher-order derivatives) duplicates op
ids: the new primal re-executes the old push/pop and the new adjoint mirrors
them, so one op id names several distinct runtime pairs. The optimizer's dead
code elimination used to pair pushes and pops through annotations that kept
only the last-seen occurrence per op id, so removing a pop that is dead at
first order deleted the WRONG push - the stack stayed balanced in count but
crossed in dataflow, corrupting second derivatives (or tripping the op-id
assertion in `tangent.pop`).

These tests pin the fix (`tangent.optimization._tape_pairings`): tape entries
that look dead in first-order code but that `grad(grad(f))` still needs must
survive optimization, removal must stay balanced, and genuinely dead tape
traffic must still be eliminated.
"""

import numpy as np
import pytest

import tangent


def _numeric_second(func, x, eps=1e-4):
    """Central-difference second derivative of a scalar function."""
    return (func(x + eps) - 2 * func(x) + func(x - eps)) / eps**2


def dead_intermediate_tape(a):
    # `b`'s pop is dead at first order (the function returns `a`), but the
    # adjoints of these tape ops are live in grad(grad): unbalanced removal
    # makes ddf read a primal value and return 16 instead of 0 at a=2.
    _stack = tangent.Stack()
    b = a * a
    tangent.push(_stack, b, 'op1')
    b = tangent.pop(_stack, 'op1')
    return a


def tape_roundtrip(a):
    # The value itself flows through the tape; d2/da2 of a**2 is 2.
    _stack = tangent.Stack()
    b = a * a
    tangent.push(_stack, b, 'op2')
    b = tangent.pop(_stack, 'op2')
    return b


def dead_after_overwrite(a):
    # `x` is overwritten before the pop restores it; the intermediate list
    # binding is dead in first order but overwrite protection needs the tape.
    _stack = tangent.Stack()
    x = a
    tangent.push(_stack, x, 'op3')
    x = a * a
    x = tangent.pop(_stack, 'op3')
    return x * x


@pytest.mark.parametrize(
    'fn,expected_dd',
    [
        (dead_intermediate_tape, lambda a: 0.0),
        (tape_roundtrip, lambda a: 2.0),
        (dead_after_overwrite, lambda a: 2.0),
    ],
)
@pytest.mark.parametrize('val', [-1.5, 0.7, 2.0])
def test_second_derivative_through_tape_optimized(fn, expected_dd, val):
    df = tangent.grad(fn, optimized=True)
    ddf = tangent.grad(df, optimized=True)
    got = ddf(val)
    assert np.allclose(got, expected_dd(val)), 'ddf(%s) = %s, expected %s' % (
        val,
        got,
        expected_dd(val),
    )
    assert np.allclose(got, _numeric_second(fn, val), atol=1e-2)


@pytest.mark.parametrize('val', [0.8, 2.0])
def test_second_derivative_matches_unoptimized(val):
    """Optimized and unoptimized second derivatives must agree."""
    for fn in (dead_intermediate_tape, tape_roundtrip, dead_after_overwrite):
        df_o = tangent.grad(fn, optimized=True)
        df_u = tangent.grad(fn, optimized=False)
        ddf_o = tangent.grad(df_o, optimized=True)
        ddf_u = tangent.grad(df_u, optimized=False)
        assert np.allclose(ddf_o(val), ddf_u(val)), fn.__name__


def loop_grad(x):
    # Control-flow gradients rely on the optimizer REMOVING dead tape pairs
    # (loop counters, conditions) when possible; second derivatives through a
    # loop exercise duplicated op ids inside one function.
    y = 1.0
    for i in range(3):
        y = y * x
    return y


def test_second_derivative_through_loop_optimized():
    ddf = tangent.grad(tangent.grad(loop_grad, optimized=True), optimized=True)
    for x in (0.5, 1.7):
        assert np.allclose(ddf(x), 6.0 * x)  # d2/dx2 x**3 = 6x


def branch_grad(x):
    if x > 0:
        y = x * x * x
    else:
        y = -x * x
    return y


def test_second_derivative_through_branch_optimized():
    ddf = tangent.grad(tangent.grad(branch_grad, optimized=True), optimized=True)
    assert np.allclose(ddf(2.0), 12.0)  # d2/dx2 x**3 = 6x
    assert np.allclose(ddf(-2.0), -2.0)  # d2/dx2 -x**2 = -2


def _dce(source):
    from tangent import optimization
    from tangent import quoting

    node = quoting.parse_string(source)
    node = optimization.dead_code_elimination(node)
    return quoting.to_source(node)


def test_dce_removes_dead_tape_pair():
    """A push/pop pair that is genuinely dead must still be eliminated.

    The fix must not degrade first-order optimization: a dead pop takes its
    push down with it (balanced removal of the pair, not a blanket barrier),
    and the value that was only kept alive by the push dies with them.
    """
    out = _dce("""
def f(x, _stack):
    y = x * x
    tangent.push(_stack, y, 'dead_pair')
    z = x * 3.0
    w = tangent.pop(_stack, 'dead_pair')
    return z
""")
    assert 'dead_pair' not in out, out
    assert 'y = x * x' not in out, out


def test_dce_balanced_removal_with_duplicate_op_ids():
    """Duplicated op ids (as higher-order AD produces) pair LIFO in order.

    The dead pop `b` must be removed together with the push that feeds it (the
    SECOND push), never with the first pair, which is live.
    """
    out = _dce("""
def f(x, _stack):
    tangent.push(_stack, x, 'dup')
    a = tangent.pop(_stack, 'dup')
    tangent.push(_stack, a, 'dup')
    b = tangent.pop(_stack, 'dup')
    return a
""")
    assert out.count('tangent.push') == 1, out
    assert out.count('tangent.pop') == 1, out
    assert 'a = tangent.pop' in out, out


def test_dce_barriers_ambiguous_duplicate_op_ids():
    """Duplicated op ids across functions cannot be paired: keep everything."""
    out = _dce("""
def f(x, _stack):
    tangent.push(_stack, x, 'shared')
    tangent.push(_stack, x, 'shared')

def g(x, _stack):
    a = tangent.pop(_stack, 'shared')
    b = tangent.pop(_stack, 'shared')
    return a
""")
    assert out.count('tangent.push') == 2, out
    assert out.count('tangent.pop') == 2, out


def checkpointed_loop(x):
    y = x
    for i in range(150):
        y = y * 1.01 + 0.001 * x
    return y


def test_checkpointing_with_optimization():
    """Checkpointing no longer force-disables optimization.

    Balanced pair removal keeps the checkpoint bookkeeping consistent, so the
    optimized gradient must run without stack mismatches and match finite
    differences.
    """
    df = tangent.grad(checkpointed_loop, checkpoint=True, optimized=True)
    x = 0.5
    eps = 1e-6
    fd = (checkpointed_loop(x + eps) - checkpointed_loop(x - eps)) / (2 * eps)
    assert np.allclose(df(x), fd, rtol=1e-4)


def nonzero_based_loop(x):
    y = x
    for i in range(2, 152):
        y = y * i * 0.01
    return y


def test_checkpointing_skips_nonzero_based_range():
    """range(start, stop) loops must not be checkpointed.

    The checkpointed adjoint reconstructs the loop target as the 0-based
    iteration index, which is wrong for a non-zero-based range whenever the
    adjoint needs the target value. Such loops fall back to the standard
    templates, so checkpoint=True must match the standard gradient exactly.
    """
    x = 0.5
    d_checkpoint = tangent.grad(nonzero_based_loop, checkpoint=True)(x)
    d_standard = tangent.grad(nonzero_based_loop)(x)
    assert d_standard != 0.0
    assert np.allclose(d_checkpoint, d_standard, rtol=1e-12, atol=0.0)


if __name__ == '__main__':
    assert not pytest.main([__file__])
