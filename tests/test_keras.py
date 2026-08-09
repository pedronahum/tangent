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
"""Unit tests for Keras 3 integration (backend-agnostic keras.ops).

These tests exercise Tangent's gradients of keras.ops functions with
whichever Keras backend is active (TensorFlow, JAX or PyTorch).
"""
import numpy as np
import pytest

try:
    import keras
    import keras.ops as kops
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

if KERAS_AVAILABLE:
    import tangent

pytestmark = pytest.mark.skipif(not KERAS_AVAILABLE,
                                reason="Keras not installed")


def tensor(arr):
    """Convert a numpy array to a backend tensor via keras."""
    return kops.convert_to_tensor(np.asarray(arr, dtype='float32'))


def allclose(a, b, tol=1e-5):
    """Compare backend tensors / scalars numerically."""
    return np.allclose(np.asarray(kops.convert_to_numpy(a)),
                       np.asarray(b), atol=tol, rtol=tol)


class TestBasicOperations:
    """Basic arithmetic with keras.ops."""

    def test_square(self):
        def f(x):
            return kops.sum(x * x)

        df = tangent.grad(f)
        x = tensor([1.0, 2.0, 3.0])
        assert allclose(df(x), [2.0, 4.0, 6.0])

    def test_polynomial(self):
        def f(x):
            return kops.sum(3.0 * x * x + 2.0 * x + 1.0)

        df = tangent.grad(f)
        x = tensor([2.0])
        assert allclose(df(x), [14.0])

    def test_add(self):
        def f(x, y):
            return kops.sum(kops.add(x, y))

        x = tensor([1.0, 2.0])
        y = tensor([3.0, 4.0])
        assert allclose(tangent.grad(f, wrt=(0,))(x, y), [1.0, 1.0])
        assert allclose(tangent.grad(f, wrt=(1,))(x, y), [1.0, 1.0])

    def test_subtract(self):
        def f(x, y):
            return kops.sum(kops.subtract(x, y))

        x = tensor([5.0, 6.0])
        y = tensor([1.0, 2.0])
        assert allclose(tangent.grad(f, wrt=(0,))(x, y), [1.0, 1.0])
        assert allclose(tangent.grad(f, wrt=(1,))(x, y), [-1.0, -1.0])

    def test_multiply(self):
        def f(x, y):
            return kops.sum(kops.multiply(x, y))

        x = tensor([2.0, 3.0])
        y = tensor([4.0, 5.0])
        assert allclose(tangent.grad(f, wrt=(0,))(x, y), [4.0, 5.0])

    def test_divide(self):
        def f(x, y):
            return kops.sum(kops.divide(x, y))

        x = tensor([4.0, 9.0])
        y = tensor([2.0, 3.0])
        assert allclose(tangent.grad(f, wrt=(0,))(x, y), [0.5, 1.0 / 3.0])


class TestMathFunctions:
    """Elementwise math functions."""

    def test_exp(self):
        def f(x):
            return kops.sum(kops.exp(x))

        x = tensor([0.0, 1.0])
        assert allclose(tangent.grad(f)(x), np.exp([0.0, 1.0]))

    def test_log(self):
        def f(x):
            return kops.sum(kops.log(x))

        x = tensor([1.0, 2.0])
        assert allclose(tangent.grad(f)(x), 1.0 / np.array([1.0, 2.0]))

    def test_sqrt(self):
        def f(x):
            return kops.sum(kops.sqrt(x))

        x = tensor([1.0, 4.0])
        assert allclose(tangent.grad(f)(x), 1.0 / (2.0 * np.sqrt([1.0, 4.0])))

    def test_sin_cos(self):
        def fs(x):
            return kops.sum(kops.sin(x))

        def fc(x):
            return kops.sum(kops.cos(x))

        x = tensor([0.3, 0.7])
        assert allclose(tangent.grad(fs)(x), np.cos([0.3, 0.7]))
        assert allclose(tangent.grad(fc)(x), -np.sin([0.3, 0.7]))

    def test_tanh(self):
        def f(x):
            return kops.sum(kops.tanh(x))

        x = tensor([0.5, -0.5])
        assert allclose(tangent.grad(f)(x),
                        1.0 - np.tanh([0.5, -0.5]) ** 2)


class TestReductions:
    """Reduction operations."""

    def test_sum(self):
        def f(x):
            return kops.sum(x)

        assert allclose(tangent.grad(f)(tensor([1.0, 2.0, 3.0])),
                        [1.0, 1.0, 1.0])

    def test_sum_axis(self):
        def f(x):
            return kops.sum(kops.sum(x, axis=1))

        x = tensor([[1.0, 2.0], [3.0, 4.0]])
        assert allclose(tangent.grad(f)(x), np.ones((2, 2)))

    def test_mean(self):
        def f(x):
            return kops.mean(x)

        assert allclose(tangent.grad(f)(tensor([1.0, 2.0, 3.0, 4.0])),
                        [0.25] * 4)


class TestLinearAlgebra:
    """Matrix operations."""

    def test_matmul_matrix(self):
        def f(x, w):
            return kops.sum(kops.matmul(x, w))

        x = tensor([[1.0, 2.0], [3.0, 4.0]])
        w = tensor([[0.5, -0.5], [1.0, 0.0]])

        # dL/dX = ones @ W^T ; dL/dW = X^T @ ones
        grad_x = tangent.grad(f, wrt=(0,))(x, w)
        grad_w = tangent.grad(f, wrt=(1,))(x, w)
        ones = np.ones((2, 2))
        assert allclose(grad_x, ones @ np.array([[0.5, -0.5], [1.0, 0.0]]).T)
        assert allclose(grad_w, np.array([[1.0, 2.0], [3.0, 4.0]]).T @ ones)

    def test_matvec(self):
        def f(a, x):
            return kops.sum(kops.matmul(a, x))

        a = tensor([[1.0, 2.0], [3.0, 4.0]])
        x = tensor([1.0, -1.0])
        grad_x = tangent.grad(f, wrt=(1,))(a, x)
        assert allclose(grad_x, [4.0, 6.0])  # column sums of a


class TestShapeOperations:
    """Shape manipulation."""

    def test_reshape(self):
        def f(x):
            return kops.sum(kops.reshape(x, (2, 3)) * 2.0)

        x = tensor(np.arange(6.0))
        assert allclose(tangent.grad(f)(x), [2.0] * 6)

    def test_transpose(self):
        def f(x):
            return kops.sum(kops.transpose(x) * 3.0)

        x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert allclose(tangent.grad(f)(x), np.full((2, 3), 3.0))


class TestSelection:
    """Elementwise selection."""

    def test_relu(self):
        def f(x):
            return kops.sum(kops.relu(x))

        x = tensor([-1.0, 0.5, 2.0])
        assert allclose(tangent.grad(f)(x), [0.0, 1.0, 1.0])

    def test_sigmoid(self):
        def f(x):
            return kops.sum(kops.sigmoid(x))

        x = tensor([0.0, 1.0])
        sig = 1.0 / (1.0 + np.exp(-np.array([0.0, 1.0])))
        assert allclose(tangent.grad(f)(x), sig * (1.0 - sig))

    def test_maximum(self):
        def f(x, y):
            return kops.sum(kops.maximum(x, y))

        x = tensor([1.0, 5.0])
        y = tensor([3.0, 2.0])
        assert allclose(tangent.grad(f, wrt=(0,))(x, y), [0.0, 1.0])

    def test_clip(self):
        def f(x):
            return kops.sum(kops.clip(x, 0.0, 1.0))

        x = tensor([-1.0, 0.5, 2.0])
        assert allclose(tangent.grad(f)(x), [0.0, 1.0, 0.0])

    def test_where(self):
        def f(x, y):
            return kops.sum(kops.where(x > 0, x, y))

        x = tensor([1.0, -1.0])
        y = tensor([10.0, 20.0])
        assert allclose(tangent.grad(f, wrt=(0,))(x, y), [1.0, 0.0])
        assert allclose(tangent.grad(f, wrt=(1,))(x, y), [0.0, 1.0])


class TestForwardMode:
    """Forward-mode differentiation of keras.ops code."""

    def test_forward_square(self):
        def f(x):
            return x * x

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        x = tensor([1.0, 2.0, 3.0])
        seed = tensor([1.0, 1.0, 1.0])
        assert allclose(df(x, seed), [2.0, 4.0, 6.0])

    def test_forward_exp(self):
        def f(x):
            return kops.exp(x)

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        x = tensor([0.0, 1.0])
        seed = tensor([1.0, 1.0])
        assert allclose(df(x, seed), np.exp([0.0, 1.0]))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
