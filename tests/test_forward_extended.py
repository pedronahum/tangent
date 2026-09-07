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
"""Forward-mode coverage for the extended NumPy gradient definitions.

numpy_extended.py historically registered adjoints only; every op there now
also carries a forward-mode (tangent) rule. Each test here checks the
forward-mode directional derivative against a central finite-difference
oracle along a fixed direction, so newly added tangents are auto-verified
without a manual derivation.
"""

import numpy as np
import pytest

import tangent


def fwd(f, x, v):
    """Forward-mode directional derivative of scalar f at x along v."""
    df = tangent.autodiff(f, mode='forward', preserve_result=False, wrt=(0,))
    return np.asarray(df(x, v))


def fd(f, x, v, h=1e-5):
    """Central finite-difference directional derivative along v."""
    x = np.asarray(x, dtype='float64')
    v = np.asarray(v, dtype='float64')
    return (f(x + h * v) - f(x - h * v)) / (2.0 * h)


RS = np.random.RandomState(42)
X = RS.randn(2, 3)
V = RS.randn(2, 3)
X_POS = np.abs(X) + 0.5


def check(f, x=None, v=None, ref=None, tol=1e-4):
    x = X if x is None else x
    v = V if v is None else v
    got = fwd(f, x, v)
    expected = fd(ref or f, x, v)
    assert np.allclose(got, expected, atol=tol, rtol=1e-4), (got, expected)


# --- Elementwise -----------------------------------------------------------


def test_absolute_tangent():
    def f(x):
        return np.sum(np.absolute(x) ** 2)

    check(f)


def test_reciprocal_tangent():
    def f(x):
        return np.sum(np.reciprocal(x))

    check(f, x=X_POS)


def test_log10_tangent():
    def f(x):
        return np.sum(np.log10(x))

    check(f, x=X_POS)


def test_log2_tangent():
    def f(x):
        return np.sum(np.log2(x))

    check(f, x=X_POS)


def test_log1p_tangent():
    def f(x):
        return np.sum(np.log1p(x))

    check(f, x=X_POS)


def test_expm1_tangent():
    def f(x):
        return np.sum(np.expm1(x))

    check(f)


def test_sign_tangent():
    def f(x):
        return np.sum(np.sign(x) * x * x)

    # d/dx along v of sum(sign(x) * x^2): sign is piecewise constant.
    check(f)


def test_floor_ceil_tangent():
    def f(x):
        return np.sum(np.floor(x) + np.ceil(x))

    got = fwd(f, X + 0.3, V)
    assert np.allclose(got, 0.0)


def test_clip_tangent():
    def f(x):
        return np.sum(np.clip(x, -0.5, 0.5) ** 2)

    check(f)


def test_where_tangent():
    # The condition is passed in rather than computed inline: forward mode
    # does not yet differentiate through comparison expressions (a
    # pre-existing limitation unrelated to the where tangent).
    c = np.asarray(X > 0)

    def f(x):
        return np.sum(np.where(c, x * x, 2.0 * x))

    check(f)


# --- Reductions ------------------------------------------------------------


def test_min_tangent():
    def f(x):
        return np.min(x)

    check(f)


def test_max_tangent():
    def f(x):
        return np.max(x)

    check(f)


def test_min_axis_tangent():
    def f(x):
        return np.sum(np.min(x, axis=0) ** 2)

    check(f)


def test_max_axis_tangent():
    def f(x):
        return np.sum(np.max(x, axis=1) ** 2)

    check(f)


def test_prod_tangent():
    def f(x):
        return np.prod(x)

    check(f, x=X_POS)


def test_prod_axis_tangent():
    def f(x):
        return np.sum(np.prod(x, axis=0))

    check(f, x=X_POS)


def test_var_tangent():
    def f(x):
        return np.var(x)

    check(f)


def test_var_axis_tangent():
    def f(x):
        return np.sum(np.var(x, axis=1))

    check(f)


def test_std_tangent():
    def f(x):
        return np.std(x)

    check(f)


def test_std_axis_tangent():
    def f(x):
        return np.sum(np.std(x, axis=0))

    check(f)


# --- Linear algebra --------------------------------------------------------


def test_matmul_tangent():
    def f(x):
        return np.sum(np.matmul(x, np.transpose(x)))

    check(f)


def test_matmul_two_arg_tangent():
    # The second operand is a (differentiable) argument rather than a closure
    # variable: forward mode defines tangents only for function arguments.
    y = RS.randn(3, 2)

    def f(x, y):
        return np.sum(np.matmul(x, y) ** 2)

    df = tangent.autodiff(f, mode='forward', preserve_result=False, wrt=(0,))
    got = np.asarray(df(X, y, V))
    expected = fd(lambda x: f(x, y), X, V)
    assert np.allclose(got, expected, atol=1e-4), (got, expected)


def test_inv_tangent():
    a = np.eye(3) * 2.0 + RS.randn(3, 3) * 0.1
    va = RS.randn(3, 3)

    def f(x):
        return np.sum(np.linalg.inv(x))

    check(f, x=a, v=va)


def test_outer_tangent():
    b = RS.randn(4)
    xv = RS.randn(3)
    vv = RS.randn(3)

    def f(x, b):
        return np.sum(np.outer(x, b) ** 2)

    df = tangent.autodiff(f, mode='forward', preserve_result=False, wrt=(0,))
    got = np.asarray(df(xv, b, vv))
    expected = fd(lambda x: f(x, b), xv, vv)
    assert np.allclose(got, expected, atol=1e-4), (got, expected)


def test_trace_tangent():
    a = RS.randn(3, 3)
    va = RS.randn(3, 3)

    def f(x):
        return np.trace(x) ** 2

    check(f, x=a, v=va)


# --- Shape manipulation ----------------------------------------------------


def test_squeeze_tangent():
    a = RS.randn(1, 3, 1)
    va = RS.randn(1, 3, 1)

    def f(x):
        return np.sum(np.squeeze(x) ** 2)

    check(f, x=a, v=va)


def test_expand_dims_tangent():
    xv = RS.randn(3)
    vv = RS.randn(3)

    def f(x):
        return np.sum(np.expand_dims(x, 0) ** 2)

    check(f, x=xv, v=vv)


def test_concatenate_tangent():
    y = RS.randn(2, 3)

    def f(x, y):
        return np.sum(np.concatenate([x * x, y], axis=0))

    df = tangent.autodiff(f, mode='forward', preserve_result=False, wrt=(0,))
    got = np.asarray(df(X, y, V))
    expected = fd(lambda x: f(x, y), X, V)
    assert np.allclose(got, expected, atol=1e-4), (got, expected)


def test_stack_tangent():
    y = RS.randn(2, 3)

    def f(x, y):
        return np.sum(np.stack([x * x, y], axis=0))

    df = tangent.autodiff(f, mode='forward', preserve_result=False, wrt=(0,))
    got = np.asarray(df(X, y, V))
    expected = fd(lambda x: f(x, y), X, V)
    assert np.allclose(got, expected, atol=1e-4), (got, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
