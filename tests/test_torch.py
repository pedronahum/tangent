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
"""Unit tests for PyTorch integration.

This module tests Tangent's automatic differentiation with torch tensors and
the functional torch API. Gradients are checked against torch.autograd where
practical, and against analytic values otherwise.
"""
import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    import tangent

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE,
                                reason="PyTorch not installed")


def autograd_ref(f, *args, wrt=0):
    """Reference gradient via torch.autograd."""
    ts = [a.detach().clone().requires_grad_(True) if isinstance(a, torch.Tensor)
          else a for a in args]
    out = f(*ts)
    if out.ndim > 0:
        out = out.sum()
    out.backward()
    return ts[wrt].grad


class TestBasicOperations:
    """Basic arithmetic with torch tensors."""

    def test_square(self):
        def f(x):
            return torch.sum(x * x)

        df = tangent.grad(f)
        x = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(df(x), 2.0 * x)

    def test_polynomial(self):
        def f(x):
            return torch.sum(3.0 * x * x + 2.0 * x + 1.0)

        df = tangent.grad(f)
        x = torch.tensor([2.0])
        assert torch.allclose(df(x), torch.tensor([14.0]))

    def test_add(self):
        def f(x, y):
            return torch.sum(x + y)

        df = tangent.grad(f, wrt=(0,))
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        assert torch.allclose(df(x, y), torch.ones_like(x))

    def test_sub(self):
        def f(x, y):
            return torch.sum(torch.sub(x, y))

        df = tangent.grad(f, wrt=(0,))
        x = torch.tensor([5.0, 6.0])
        y = torch.tensor([1.0, 2.0])
        assert torch.allclose(df(x, y), torch.ones_like(x))
        dfy = tangent.grad(f, wrt=(1,))
        assert torch.allclose(dfy(x, y), -torch.ones_like(y))

    def test_mul(self):
        def f(x, y):
            return torch.sum(torch.mul(x, y))

        df = tangent.grad(f, wrt=(0,))
        x = torch.tensor([2.0, 3.0])
        y = torch.tensor([4.0, 5.0])
        assert torch.allclose(df(x, y), y)

    def test_div(self):
        def f(x, y):
            return torch.sum(torch.div(x, y))

        df = tangent.grad(f, wrt=(0,))
        x = torch.tensor([4.0, 9.0])
        y = torch.tensor([2.0, 3.0])
        assert torch.allclose(df(x, y), 1.0 / y)

    def test_pow(self):
        def f(x):
            return torch.sum(torch.pow(x, 3.0))

        df = tangent.grad(f)
        x = torch.tensor([1.0, 2.0])
        assert torch.allclose(df(x), 3.0 * x ** 2)

    def test_neg(self):
        def f(x):
            return torch.sum(torch.neg(x))

        df = tangent.grad(f)
        x = torch.tensor([1.0, -2.0])
        assert torch.allclose(df(x), -torch.ones_like(x))


class TestMathFunctions:
    """Elementwise math functions."""

    def test_exp(self):
        def f(x):
            return torch.sum(torch.exp(x))

        df = tangent.grad(f)
        x = torch.tensor([0.0, 1.0])
        assert torch.allclose(df(x), torch.exp(x))

    def test_log(self):
        def f(x):
            return torch.sum(torch.log(x))

        df = tangent.grad(f)
        x = torch.tensor([1.0, 2.0])
        assert torch.allclose(df(x), 1.0 / x)

    def test_sqrt(self):
        def f(x):
            return torch.sum(torch.sqrt(x))

        df = tangent.grad(f)
        x = torch.tensor([1.0, 4.0])
        assert torch.allclose(df(x), 1.0 / (2.0 * torch.sqrt(x)))

    def test_sin(self):
        def f(x):
            return torch.sum(torch.sin(x))

        df = tangent.grad(f)
        x = torch.tensor([0.3, 0.7])
        assert torch.allclose(df(x), torch.cos(x))

    def test_cos(self):
        def f(x):
            return torch.sum(torch.cos(x))

        df = tangent.grad(f)
        x = torch.tensor([0.3, 0.7])
        assert torch.allclose(df(x), -torch.sin(x))

    def test_tanh(self):
        def f(x):
            return torch.sum(torch.tanh(x))

        df = tangent.grad(f)
        x = torch.tensor([0.5, -0.5])
        assert torch.allclose(df(x), 1.0 - torch.tanh(x) ** 2)

    def test_arctan(self):
        def f(x):
            return torch.sum(torch.arctan(x))

        df = tangent.grad(f)
        x = torch.tensor([0.5, -0.5])
        assert torch.allclose(df(x), 1.0 / (1.0 + x * x))


class TestReductions:
    """Reduction operations."""

    def test_sum_all(self):
        def f(x):
            return torch.sum(x)

        df = tangent.grad(f)
        x = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(df(x), torch.ones_like(x))

    def test_sum_axis(self):
        def f(x):
            return torch.sum(torch.sum(x, 1))

        df = tangent.grad(f)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(df(x), torch.ones_like(x))

    def test_mean(self):
        def f(x):
            return torch.mean(x)

        df = tangent.grad(f)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(df(x), torch.full_like(x, 0.25))

    def test_mean_axis(self):
        def f(x):
            return torch.sum(torch.mean(x, 0))

        df = tangent.grad(f)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(df(x), torch.full_like(x, 0.5))


class TestLinearAlgebra:
    """Matrix operations."""

    def test_matmul_matrix(self):
        def f(x, w):
            return torch.sum(torch.matmul(x, w))

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        w = torch.tensor([[0.5, -0.5], [1.0, 0.0]])

        dfx = tangent.grad(f, wrt=(0,))
        assert torch.allclose(dfx(x, w), autograd_ref(f, x, w, wrt=0))

        dfw = tangent.grad(f, wrt=(1,))
        assert torch.allclose(dfw(x, w), autograd_ref(f, x, w, wrt=1))

    def test_matvec(self):
        def f(a, x):
            return torch.sum(torch.mv(a, x))

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x = torch.tensor([1.0, -1.0])

        dfa = tangent.grad(f, wrt=(0,))
        assert torch.allclose(dfa(a, x), autograd_ref(f, a, x, wrt=0))

        dfx = tangent.grad(f, wrt=(1,))
        assert torch.allclose(dfx(a, x), autograd_ref(f, a, x, wrt=1))

    def test_dot(self):
        def f(x, y):
            return torch.dot(x, y)

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])

        dfx = tangent.grad(f, wrt=(0,))
        assert torch.allclose(dfx(x, y), y)

        dfy = tangent.grad(f, wrt=(1,))
        assert torch.allclose(dfy(x, y), x)

    def test_matmul_vector_inner(self):
        def f(x, y):
            return torch.matmul(x, y)

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([0.5, -1.0, 2.0])

        dfx = tangent.grad(f, wrt=(0,))
        assert torch.allclose(dfx(x, y), y)


class TestShapeOperations:
    """Shape manipulation."""

    def test_reshape(self):
        def f(x):
            return torch.sum(torch.reshape(x, (2, 3)) * 2.0)

        df = tangent.grad(f)
        x = torch.arange(6.0)
        assert torch.allclose(df(x), torch.full((6,), 2.0))

    def test_transpose(self):
        def f(x):
            return torch.sum(torch.transpose(x, 0, 1) * 3.0)

        df = tangent.grad(f)
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert torch.allclose(df(x), torch.full((2, 3), 3.0))

    def test_squeeze_unsqueeze(self):
        def f(x):
            return torch.sum(torch.squeeze(torch.unsqueeze(x, 0), 0))

        df = tangent.grad(f)
        x = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(df(x), torch.ones_like(x))


class TestSelection:
    """Elementwise selection and clamping."""

    def test_abs(self):
        def f(x):
            return torch.sum(torch.abs(x))

        df = tangent.grad(f)
        x = torch.tensor([-2.0, 3.0])
        assert torch.allclose(df(x), torch.tensor([-1.0, 1.0]))

    def test_maximum(self):
        def f(x, y):
            return torch.sum(torch.maximum(x, y))

        x = torch.tensor([1.0, 5.0])
        y = torch.tensor([3.0, 2.0])
        dfx = tangent.grad(f, wrt=(0,))
        assert torch.allclose(dfx(x, y), torch.tensor([0.0, 1.0]))

    def test_clamp(self):
        def f(x):
            return torch.sum(torch.clamp(x, min=0.0, max=1.0))

        df = tangent.grad(f)
        x = torch.tensor([-1.0, 0.5, 2.0])
        assert torch.allclose(df(x), torch.tensor([0.0, 1.0, 0.0]))

    def test_where(self):
        def f(x, y):
            return torch.sum(torch.where(x > 0, x, y))

        x = torch.tensor([1.0, -1.0])
        y = torch.tensor([10.0, 20.0])
        dfx = tangent.grad(f, wrt=(0,))
        assert torch.allclose(dfx(x, y), torch.tensor([1.0, 0.0]))
        dfy = tangent.grad(f, wrt=(1,))
        assert torch.allclose(dfy(x, y), torch.tensor([0.0, 1.0]))


class TestActivations:
    """Common neural-network activations."""

    def test_relu(self):
        def f(x):
            return torch.sum(torch.relu(x))

        df = tangent.grad(f)
        x = torch.tensor([-1.0, 0.5, 2.0])
        assert torch.allclose(df(x), torch.tensor([0.0, 1.0, 1.0]))

    def test_sigmoid(self):
        def f(x):
            return torch.sum(torch.sigmoid(x))

        df = tangent.grad(f)
        x = torch.tensor([0.0, 1.0])
        sig = torch.sigmoid(x)
        assert torch.allclose(df(x), sig * (1.0 - sig))


class TestHigherOrder:
    """Second derivatives through torch code."""

    def test_gradgrad_polynomial(self):
        def f(x):
            return x * x * x

        df = tangent.grad(f, optimized=False)
        ddf = tangent.grad(df, optimized=False)
        x = torch.tensor(2.0)
        assert torch.allclose(ddf(x), torch.tensor(12.0))


class TestForwardMode:
    """Forward-mode differentiation of torch code."""

    def test_forward_square(self):
        def f(x):
            return x * x

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        x = torch.tensor([1.0, 2.0, 3.0])
        seed = torch.ones_like(x)
        assert torch.allclose(df(x, seed), 2.0 * x)

    def test_forward_exp(self):
        def f(x):
            return torch.exp(x)

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        x = torch.tensor([0.0, 1.0])
        seed = torch.ones_like(x)
        assert torch.allclose(df(x, seed), torch.exp(x))

    def test_forward_sum(self):
        def f(x):
            return torch.sum(x * x)

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        x = torch.tensor([1.0, 2.0, 3.0])
        seed = torch.ones_like(x)
        # d(sum(x^2)) . seed = sum(2*x*seed) = 12
        assert torch.allclose(df(x, seed), torch.tensor(12.0))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
