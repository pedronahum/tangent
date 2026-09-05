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
"""Tests for NumPy array-method calls on computed values.

Tangent resolves calls against the function's globals, so a method on a local
(``x.sum()`` after ``x = a + b``) is rewritten to its numpy function
equivalent (numpy.sum(x)). This covers that rewrite, including the varargs
methods (reshape/transpose) whose numpy counterparts take a single sequence
argument.
"""
import numpy as np

import tangent


def test_method_sum():
    def f(x):
        return (x * 2.0).sum()

    df = tangent.grad(f)
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(df(x), np.full(3, 2.0))


def test_method_mean():
    def f(x):
        return (x * x).mean()

    df = tangent.grad(f)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(df(x), x / 2.0)


def test_method_reshape_varargs():
    # x.reshape(2, 3): the two ints must be packed into a shape tuple for
    # numpy.reshape.
    def f(x):
        return x.reshape(2, 3).sum()

    df = tangent.grad(f)
    x = np.arange(6.0)
    assert np.allclose(df(x), np.ones(6))


def test_method_reshape_tuple():
    def f(x):
        return (x.reshape((2, 3)) * 2.0).sum()

    df = tangent.grad(f)
    x = np.arange(6.0)
    assert np.allclose(df(x), np.full(6, 2.0))


def test_method_transpose_varargs():
    def f(x):
        return x.transpose(1, 0).sum()

    df = tangent.grad(f)
    x = np.arange(6.0).reshape(2, 3)
    assert np.allclose(df(x), np.ones((2, 3)))


def test_method_transpose_tuple():
    def f(x):
        return (x.transpose((1, 0)) * 3.0).sum()

    df = tangent.grad(f)
    x = np.arange(6.0).reshape(2, 3)
    assert np.allclose(df(x), np.full((2, 3), 3.0))


def test_method_transpose_three_axes():
    # Gradient of sum(T(x) * C) is transpose(C, inverse(axes)).
    c = np.arange(24.0).reshape(4, 2, 3)

    def f(x):
        return (x.transpose(2, 0, 1) * c).sum()

    df = tangent.grad(f)
    x = np.arange(24.0).reshape(2, 3, 4)
    assert np.allclose(df(x), np.transpose(c, (1, 2, 0)))


def test_method_transpose_no_args():
    def f(x):
        return (x.transpose() * 4.0).sum()

    df = tangent.grad(f)
    x = np.arange(6.0).reshape(2, 3)
    assert np.allclose(df(x), np.full((2, 3), 4.0))


def test_method_squeeze():
    def f(x):
        return x.squeeze().sum()

    df = tangent.grad(f)
    x = np.arange(3.0).reshape(1, 3, 1)
    assert np.allclose(df(x), np.ones((1, 3, 1)))
