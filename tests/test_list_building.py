"""Differentiable list building: `xs.append(v)` and `v = xs.pop()`.

These statements are desugared into rebindings through the primitives
`tangent.list_append` / `tangent.list_last` / `tangent.list_init` (see
tangent/list_method_desugar.py), which have adjoints and tangents written in
terms of each other so higher-order differentiation stays inside the set.
Before that pass existed, `xs.append(v)` was silently treated as
non-differentiable and gradients came back as zero.

Every gradient here is checked against the central finite-difference oracle.
"""

import numpy as np
import pytest

import tangent
from tangent.errors import TangentParseError

from utils import numeric_grad


# --- Functions under test (module level so their source is retrievable). ---


def straight_line_append(x):
    xs = []
    xs.append(x * x)
    xs.append(3.0 * x)
    return xs[0] + xs[1]


def append_to_nonempty_literal(x):
    xs = [x * 2.0]
    xs.append(x * x)
    return xs[0] * xs[1]


def loop_append_array(x):
    ys = []
    for i in range(len(x)):
        ys.append(x[i] * x[i])
    s = 0.0
    for j in range(len(ys)):
        s = s + ys[j]
    return s


def append_then_pop(x):
    xs = []
    xs.append(x * x)
    xs.append(2.0 * x)
    v = xs.pop()
    return v + xs[0]


def bare_pop(x):
    xs = []
    xs.append(x * 3.0)
    xs.append(x * x * x)
    xs.pop()
    return xs[0]


def inactive_bookkeeping(x):
    idxs = []
    idxs.append(1)
    idxs.append(0)
    return x * float(idxs[0] + 2)


def rejected_extend(x):
    xs = [x]
    xs.extend([x * 2.0])
    return xs[0] + xs[1]


def rejected_nested_append(x):
    ys = [[]]
    ys[0].append(x * x)
    return ys[0][0]


def rejected_sort(x):
    xs = [x * 2.0, x]
    xs.sort()
    return xs[0]


# --- Reverse mode, first order, against finite differences. ---


@pytest.mark.parametrize("pt", [0.7, 2.0, -1.3])
@pytest.mark.parametrize(
    'fn',
    [straight_line_append, append_to_nonempty_literal, append_then_pop, bare_pop],
)
def test_scalar_reverse_matches_fd(fn, pt):
    df = tangent.grad(fn)
    assert df(pt) == pytest.approx(numeric_grad(fn)(pt), rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("opt", [True, False])
def test_loop_append_array_reverse(opt):
    x = np.array([1.0, -2.0, 3.5])
    df = tangent.grad(loop_append_array, optimized=opt)
    np.testing.assert_allclose(df(x), 2.0 * x)
    np.testing.assert_allclose(df(x), numeric_grad(loop_append_array)(x), rtol=1e-5)


def test_split_motion():
    df = tangent.autodiff(
        straight_line_append,
        motion='split',
        mode='reverse',
        input_derivative=tangent.grad_util.INPUT_DERIVATIVE.DefaultOne,
    )
    assert df(2.0) == pytest.approx(7.0)


# --- Forward mode. ---


def test_forward_mode_scalar():
    df = tangent.autodiff(straight_line_append, mode='forward')
    assert df(2.0, 1.0) == pytest.approx(7.0)
    df_pop = tangent.autodiff(append_then_pop, mode='forward')
    assert df_pop(2.0, 1.0) == pytest.approx(6.0)


def test_forward_mode_loop_append():
    x = np.array([1.0, 2.0, 3.0])
    df = tangent.autodiff(loop_append_array, mode='forward')
    # Directional derivative along ones: sum of the gradient.
    assert df(x, np.ones_like(x)) == pytest.approx(float(np.sum(2.0 * x)))


# --- Higher order. ---


def test_second_derivative_scalar():
    ddf = tangent.grad(tangent.grad(straight_line_append))
    assert ddf(2.0) == pytest.approx(2.0)


def test_second_derivative_through_loop_append():
    x = np.array([1.0, -2.0, 3.5])
    ddf = tangent.grad(tangent.grad(loop_append_array))
    np.testing.assert_allclose(ddf(x), 2.0 * np.ones_like(x))


def test_third_derivative_scalar():
    dddf = tangent.grad(tangent.grad(tangent.grad(straight_line_append)))
    assert dddf(2.0) == pytest.approx(0.0)


# --- Lists that are not differentiated keep working. ---


def test_inactive_bookkeeping_list():
    assert tangent.grad(inactive_bookkeeping)(2.0) == pytest.approx(3.0)


# --- Unsupported mutations are rejected, not silently dropped. ---


@pytest.mark.parametrize("fn", [rejected_extend, rejected_nested_append, rejected_sort])
def test_unsupported_list_mutation_rejected(fn):
    with pytest.raises(TangentParseError, match='In-place list mutation'):
        tangent.grad(fn)


# --- The runtime primitives validate their inputs. ---


def test_primitives_type_checked():
    # Gradients of list variables can arrive as ndarrays (e.g. np.sum's
    # adjoint broadcasts over the whole sequence), so sequences are accepted
    # and normalized; non-sequences are rejected loudly.
    assert tangent.list_append((1.0,), 2.0) == [1.0, 2.0]
    assert tangent.list_last(np.array([1.0, 2.0])) == 2.0
    with pytest.raises(TypeError, match='list_append expected a sequence'):
        tangent.list_append({'a': 1.0}, 2.0)
    with pytest.raises(TypeError, match='list_last expected a sequence'):
        tangent.list_last(1.0)
    with pytest.raises(IndexError):
        tangent.list_last([])
    with pytest.raises(IndexError):
        tangent.list_init([])


def test_add_grad_list_length_mismatch_raises():
    with pytest.raises(ValueError, match='different lengths'):
        tangent.add_grad([1.0, 2.0], [1.0])
