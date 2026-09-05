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
"""Tests for the @ (matrix multiplication) operator.

The operator is backend-dispatched through tangent.utils.register_matmul_grad;
this module covers the NumPy registration and its rank-promotion cases
(vector inner product, matrix-vector, matrix-matrix, batched). Backend-specific
coverage lives in tests/test_tinygrad.py and tests/test_backend_coverage.py.
"""
import numpy as np
import pytest

import tangent


class TestMatMulOperator:

    def test_vector_inner_product(self):
        def f(x, y):
            return x @ y

        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.5, -1.0, 2.0])
        assert np.allclose(tangent.grad(f, wrt=(0,))(x, y), y)
        assert np.allclose(tangent.grad(f, wrt=(1,))(x, y), x)

    def test_matrix_vector(self):
        def f(x, w):
            return (x @ w).sum()

        x = np.random.RandomState(0).randn(2, 3)
        w = np.random.RandomState(1).randn(3)
        assert np.allclose(tangent.grad(f, wrt=(0,))(x, w),
                           np.ones(2)[:, None] * w[None, :])
        assert np.allclose(tangent.grad(f, wrt=(1,))(x, w), x.sum(0))

    def test_vector_matrix(self):
        def f(x, w):
            return (x @ w).sum()

        x = np.random.RandomState(0).randn(3)
        w = np.random.RandomState(1).randn(3, 4)
        assert np.allclose(tangent.grad(f, wrt=(0,))(x, w), w.sum(1))
        assert np.allclose(tangent.grad(f, wrt=(1,))(x, w),
                           x[:, None] * np.ones(4)[None, :])

    def test_matrix_matrix(self):
        def f(x, w):
            return (x @ w).sum()

        x = np.random.RandomState(0).randn(2, 3)
        w = np.random.RandomState(1).randn(3, 4)
        assert np.allclose(tangent.grad(f, wrt=(0,))(x, w),
                           np.ones((2, 4)) @ w.T)
        assert np.allclose(tangent.grad(f, wrt=(1,))(x, w),
                           x.T @ np.ones((2, 4)))

    def test_batched(self):
        def f(x, w):
            return (x @ w).sum()

        x = np.random.RandomState(0).randn(5, 2, 3)
        w = np.random.RandomState(1).randn(5, 3, 4)
        assert np.allclose(tangent.grad(f, wrt=(0,))(x, w),
                           np.matmul(np.ones((5, 2, 4)),
                                     np.swapaxes(w, -1, -2)))
        assert np.allclose(tangent.grad(f, wrt=(1,))(x, w),
                           np.matmul(np.swapaxes(x, -1, -2),
                                     np.ones((5, 2, 4))))

    def test_chained_with_operators(self):
        # @ combines with the ordinary operator adjoints.
        def f(x, w):
            return ((x @ w) ** 2).sum()

        x = np.random.RandomState(0).randn(2, 3)
        w = np.random.RandomState(1).randn(3, 2)
        z = x @ w
        assert np.allclose(tangent.grad(f, wrt=(0,))(x, w),
                           2.0 * z @ w.T)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
