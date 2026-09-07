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
"""Tests for pytree (container) handling in Tangent.

Tangent differentiates functions whose ARGUMENTS are pytrees - tuples, lists,
dicts and nested combinations of arrays - including indexing into them and
looping over them. These tests lock in that support.

RETURN values may also be pytrees: reverse mode reconciles the gradient seed
with the return value's structure at runtime (`tangent.match_seed`, emitted at
the top of the adjoint). The default scalar seed expands into a pytree of ones
- the gradient of the sum of all leaves, matching how scalar and tuple outputs
behave - and a caller-supplied seed of matching structure is used as the
cotangent. Structurally mismatched container seeds are rejected with a clean
ValueError. The container-output gradients below are verified against finite
differences on the flattened leaves.
"""

import math

import numpy as np
import pytest

import tangent
from tangent.utils import match_seed, seed_pytree

from autograd.misc.flatten import flatten as _flatten


def _fd_grad(func, x, seed=None, eps=1e-6):
    """Finite-difference gradient of leaf-sum(seed * func(x)), flattened.

    Perturbs every leaf entry of the (pytree) argument `x` with central
    differences and dots the output perturbation with the (pytree) `seed`
    (ones when None), returning the flat gradient vector.
    """

    def scalarize(v):
        out_flat, _ = _flatten(func(v))
        if seed is None:
            return np.sum(out_flat)
        seed_flat, _ = _flatten(seed)
        return np.dot(out_flat, seed_flat)

    x_flat, unflatten = _flatten(x)
    g = np.zeros_like(np.asarray(x_flat, dtype=float))
    for i in range(x_flat.size):
        xp = np.array(x_flat, dtype=float)
        xm = np.array(x_flat, dtype=float)
        xp[i] += eps / 2
        xm[i] -= eps / 2
        g[i] = (scalarize(unflatten(xp)) - scalarize(unflatten(xm))) / eps
    return g


def _flat(tree):
    return _flatten(tree)[0]


# ---------------------------------------------------------------------------
# Container inputs: gradients flow through tuples, lists, dicts, nesting
# ---------------------------------------------------------------------------


def test_tuple_input():
    def f(params):
        return np.sum(params[0] * params[1])

    df = tangent.grad(f)
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    ga, gb = df((a, b))
    assert np.allclose(ga, b)
    assert np.allclose(gb, a)


def test_list_input():
    def f(params):
        return np.sum(params[0] + params[1])

    df = tangent.grad(f)
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    ga, gb = df([a, b])
    assert np.allclose(ga, np.ones(2))
    assert np.allclose(gb, np.ones(2))


def test_dict_input():
    def f(d):
        return np.sum(d['a'] * d['b'])

    df = tangent.grad(f)
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    grads = df({'a': a, 'b': b})
    assert np.allclose(grads['a'], b)
    assert np.allclose(grads['b'], a)


def test_nested_container_input():
    def f(d):
        return np.sum(d['x'][0] * d['x'][1]) + np.sum(d['y'])

    df = tangent.grad(f)
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    grads = df({'x': (a, b), 'y': a})
    assert np.allclose(grads['x'][0], b)
    assert np.allclose(grads['x'][1], a)
    assert np.allclose(grads['y'], np.ones(2))


def test_loop_over_container_input():
    def f(params):
        s = 0.0
        for p in params:
            s = s + np.sum(p * p)
        return s

    df = tangent.grad(f)
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    ga, gb = df((a, b))
    assert np.allclose(ga, 2 * a)
    assert np.allclose(gb, 2 * b)


def test_wrt_container_argument():
    # wrt selects a function *argument*; when that argument is a container the
    # whole container's gradient is returned.
    def f(scale, params):
        return np.sum(scale * params[0])

    df = tangent.grad(f, wrt=(1,))
    scale = 2.0
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    got = df(scale, (a, b))
    # d/d(params[0]) sum(scale * params[0]) = scale
    assert np.allclose(got[0], scale)


# ---------------------------------------------------------------------------
# Container outputs: the seed is reconciled with the return value's structure
# ---------------------------------------------------------------------------


def test_dict_output_default_seed():
    def f(x):
        return {'a': x * x, 'b': 3.0 * x}

    df = tangent.grad(f)
    x = np.array([1.0, 2.0])
    got = df(x)
    # Default seed: gradient of the sum of all leaves = 2x + 3.
    assert np.allclose(got, 2 * x + 3.0)
    assert np.allclose(_flat(got), _fd_grad(f, x), atol=1e-4)


def test_nested_dict_output():
    def f(x):
        return {'u': [x * x, np.tanh(x)], 'v': np.sum(x * x)}

    df = tangent.grad(f)
    x = np.array([0.3, -0.7, 1.1])
    assert np.allclose(_flat(df(x)), _fd_grad(f, x), atol=1e-4)


def test_list_of_arrays_output():
    def f(x):
        return [x * x, np.sin(x)]

    df = tangent.grad(f)
    x = np.array([0.5, 1.5, -0.5])
    assert np.allclose(_flat(df(x)), _fd_grad(f, x), atol=1e-4)
    for optimized in (True, False):
        for check_dims in (True, False):
            dfv = tangent.grad(f, optimized=optimized, check_dims=check_dims)
            assert np.allclose(_flat(dfv(x)), _fd_grad(f, x), atol=1e-4)


def test_dict_arg_dict_output():
    def f(d):
        return {'out': d['w'] * d['b'], 'aux': np.sum(d['w'] * d['w'])}

    df = tangent.grad(f)
    arg = {'w': np.array([1.0, 2.0]), 'b': np.array([3.0, 4.0])}
    got = df(arg)
    assert set(got.keys()) == {'w', 'b'}
    assert np.allclose(_flat(got), _fd_grad(f, arg), atol=1e-4)


def test_container_output_custom_seed():
    def f(x):
        return {'a': x * x, 'b': 3.0 * x}

    df = tangent.grad(f)
    x = np.array([1.0, 2.0])
    seed = {'a': np.array([1.0, 1.0]), 'b': np.array([0.0, 0.0])}
    got = df(x, seed)
    # Only the 'a' output is seeded: gradient is 2x.
    assert np.allclose(got, 2 * x)
    assert np.allclose(_flat(got), _fd_grad(f, x, seed=seed), atol=1e-4)


def test_container_output_preserve_result():
    def f(x):
        return {'a': x * x}

    df = tangent.grad(f, preserve_result=True)
    x = np.array([1.0, 2.0])
    grad, result = df(x)
    assert np.allclose(grad, 2 * x)
    assert np.allclose(result['a'], x * x)


# ---------------------------------------------------------------------------
# Structurally mismatched seeds are rejected cleanly, not crashed on
# ---------------------------------------------------------------------------


def test_wrong_dict_seed_keys_rejected():
    def f(x):
        return {'a': x * x, 'b': 3.0 * x}

    df = tangent.grad(f)
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match='structure does not match'):
        df(x, {'a': np.ones(2), 'wrong': np.ones(2)})


def test_wrong_container_kind_seed_rejected():
    def f(x):
        return {'a': x * x, 'b': 3.0 * x}

    df = tangent.grad(f)
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match='structure does not match'):
        df(x, [np.ones(2), np.ones(2)])


def test_wrong_list_seed_length_rejected():
    def f(x):
        return [x * x, np.sin(x)]

    df = tangent.grad(f)
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match='structure does not match'):
        df(x, [np.ones(2)])


def test_wrong_leaf_shape_in_container_seed_rejected():
    def f(x):
        return {'a': x * x}

    df = tangent.grad(f)
    x = np.array([1.0, 2.0])
    with pytest.raises(AssertionError):
        df(x, {'a': np.ones(5)})


def test_container_seed_for_scalar_output_rejected():
    def f(x):
        return np.sum(x * x)

    df = tangent.grad(f)
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match='structure does not match'):
        df(x, {'a': np.ones(2)})


# ---------------------------------------------------------------------------
# Higher-order through container inputs must stay correct
# ---------------------------------------------------------------------------


def test_higher_order_through_container_input():
    def f(params):
        return np.sum(params[0] * params[0])

    df = tangent.grad(f)
    ddf = tangent.grad(df)
    a = np.array([1.0, 2.0])
    got = ddf((a,))
    # d^2/da^2 sum(a^2) = 2
    assert np.allclose(np.asarray(got[0]), [2.0, 2.0])
    # Verify against finite differences of the first-order gradient (whose
    # own output is a container, so its leaves are summed by the default
    # seed of the second-order gradient).
    assert np.allclose(_flat(list(got)), _fd_grad(lambda p: list(df(p)), (a,)), atol=1e-4)


def test_higher_order_scalar_unaffected():
    # Regression guard for the seed/grad machinery on plain scalar functions.
    def f(x):
        return np.sin(x) * x

    ddf = tangent.grad(tangent.grad(f))
    x = 0.7
    expected = -math.sin(x) * x + 2 * math.cos(x)
    assert abs(ddf(x) - expected) < 1e-5


# ---------------------------------------------------------------------------
# seed_pytree / match_seed building blocks
# ---------------------------------------------------------------------------


def test_seed_pytree_structure():
    x = np.array([1.0, 2.0])
    tree = {'a': x, 'b': (x, 3.0), 'c': [x]}
    seed = seed_pytree(tree)
    assert set(seed.keys()) == {'a', 'b', 'c'}
    assert np.allclose(seed['a'], np.ones(2))
    assert isinstance(seed['b'], tuple)
    assert np.allclose(seed['b'][0], np.ones(2))
    assert seed['b'][1] == 1.0
    assert isinstance(seed['c'], list)
    assert np.allclose(seed['c'][0], np.ones(2))


def test_match_seed_scalar_passthrough():
    assert match_seed(2.0, 1.0) == 1.0


def test_match_seed_array_expands_scalar():
    x = np.array([1.0, 2.0, 3.0])
    got = match_seed(x, 1.0)
    assert np.allclose(got, np.ones(3))


def test_match_seed_container_with_scalar_seed():
    x = np.array([1.0, 2.0])
    got = match_seed({'u': x, 'v': x}, 1.0)
    assert np.allclose(got['u'], np.ones(2))
    assert np.allclose(got['v'], np.ones(2))


def test_match_seed_reconciles_leaves():
    x = np.array([1.0, 2.0])
    got = match_seed((x, x), (1.0, 1.0))
    assert np.allclose(got[0], np.ones(2))
    assert np.allclose(got[1], np.ones(2))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
