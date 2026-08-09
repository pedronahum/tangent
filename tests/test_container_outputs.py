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

Returning containers is a separate (harder) capability: the reverse-mode seed
must be expanded into a pytree of ones matching the output, and doing that
inside the gradient currently interferes with higher-order differentiation.
tangent.seed_pytree / tangent.match_seed are the building blocks for that work
and are unit-tested here.
"""
import math

import numpy as np
import pytest

import tangent
from tangent.utils import match_seed, seed_pytree


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
# Higher-order through container inputs must stay correct
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="Second derivatives through container (pytree) arguments are not "
           "yet supported; the seed machinery raises a shape-check KeyError.")
def test_higher_order_through_container_input():
    def f(params):
        return np.sum(params[0] * params[0])

    ddf = tangent.grad(tangent.grad(f))
    a = np.array([1.0, 2.0])
    got = ddf((a,))
    # d^2/da^2 sum(a^2) = 2
    assert np.allclose(np.asarray(got[0]), [2.0, 2.0])


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
