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
"""Tests for differentiable jnp.concatenate / jnp.stack.

These ops take a *list* of arrays; Tangent desugars list-literal calls into
varargs helper calls (tangent.concat_seq / tangent.stack_seq) whose varargs
adjoints distribute the gradient back to each input array. Both optimized and
unoptimized modes are exercised because the varargs pack/unpack statements
must survive dead code elimination.
"""
import numpy as np
import pytest

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

if HAS_JAX:
    import tangent

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")


def _allclose(a, b, tol=1e-5):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol, rtol=1e-4)


def test_concatenate_gradient(optimized):
    def f(a, b, c):
        return jnp.sum(jnp.concatenate([a, b, c], axis=0))

    df = tangent.grad(f, wrt=(0, 1, 2), optimized=optimized)
    a = jnp.array([1.0, 2.0])
    b = jnp.array([3.0])
    c = jnp.array([4.0, 5.0, 6.0])
    ga, gb, gc = df(a, b, c)
    assert _allclose(ga, np.ones(2))
    assert _allclose(gb, np.ones(1))
    assert _allclose(gc, np.ones(3))


def test_stack_gradient(optimized):
    def f(a, b):
        return jnp.sum(jnp.stack([a, b], axis=0))

    df = tangent.grad(f, wrt=(0, 1), optimized=optimized)
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    gx, gy = df(x, y)
    assert _allclose(gx, np.ones(3))
    assert _allclose(gy, np.ones(3))


def test_concatenate_weighted(optimized):
    """Gradient flows correctly when the concatenated result is transformed."""
    def f(a, b):
        joined = jnp.concatenate([a, b], axis=0)
        return jnp.sum(joined * joined)

    df = tangent.grad(f, wrt=(0, 1), optimized=optimized)
    a = jnp.array([1.0, 2.0])
    b = jnp.array([3.0])
    ga, gb = df(a, b)
    # d/da sum(x^2) = 2x
    assert _allclose(ga, 2.0 * np.array([1.0, 2.0]))
    assert _allclose(gb, 2.0 * np.array([3.0]))


def test_concatenate_variable_list(optimized):
    """A list bound to a variable that is assigned exactly once to a literal
    and never mutated is inlined, so it differentiates like the literal form."""
    def f(a, b):
        parts = [a, b]
        return jnp.sum(jnp.concatenate(parts, axis=0))

    df = tangent.grad(f, wrt=(0, 1), optimized=optimized)
    ga, gb = df(jnp.array([1.0, 2.0]), jnp.array([3.0]))
    assert _allclose(ga, np.ones(2))
    assert _allclose(gb, np.ones(1))


def test_stack_variable_list(optimized):
    def g(a, b):
        parts = [a, b]
        return jnp.sum(jnp.stack(parts, axis=0))

    dg = tangent.grad(g, wrt=(0, 1), optimized=optimized)
    gx, gy = dg(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    assert _allclose(gx, np.ones(2))
    assert _allclose(gy, np.ones(2))


def test_concatenate_reassigned_list_raises():
    """A list that is reassigned is not statically known and must raise a
    clear NotImplementedError rather than generate broken code."""
    def f(a, b):
        parts = [a]
        parts = [a, b]
        return jnp.sum(jnp.concatenate(parts, axis=0))

    with pytest.raises(NotImplementedError):
        tangent.grad(f, wrt=(0, 1))(jnp.array([1.0]), jnp.array([2.0]))


def test_concatenate_mutated_list_raises():
    """A list that is modified in place (subscript assignment) is not
    statically known and must raise a clear NotImplementedError."""
    def f(a, b):
        parts = [a, a]
        parts[1] = b
        return jnp.sum(jnp.concatenate(parts, axis=0))

    with pytest.raises(NotImplementedError):
        tangent.grad(f, wrt=(0, 1))(jnp.array([1.0]), jnp.array([2.0]))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
