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
"""Unit tests for tinygrad integration.

This module tests Tangent's automatic differentiation with tinygrad tensors.
tinygrad exposes its operations as methods on the tensor object
(``x.relu()``, ``x.sum()``, ``x.matmul(w)``), so these tests exercise the
method-resolution machinery in addition to the gradient rules themselves.

Gradients are checked against tinygrad's own autodiff (``Tensor.gradient``)
where practical, and against analytic values otherwise.
"""
import numpy as np
import pytest

try:
    from tinygrad import Tensor
    TINYGRAD_AVAILABLE = True
except ImportError:
    TINYGRAD_AVAILABLE = False

if TINYGRAD_AVAILABLE:
    import tangent

pytestmark = pytest.mark.skipif(not TINYGRAD_AVAILABLE,
                                reason="tinygrad not installed")


def _np(t):
    """Materialize a tinygrad tensor (or scalar) as a NumPy array."""
    if isinstance(t, Tensor):
        return t.numpy()
    return np.asarray(t)


def tinygrad_ref(f, *args, wrt=0):
    """Reference gradient via tinygrad's own autodiff."""
    ts = [a if isinstance(a, Tensor) else Tensor(a) for a in args]
    out = f(*ts)
    return out.gradient(ts[wrt])[0].numpy()


def _allclose(a, b, rtol=1e-4, atol=1e-4):
    return np.allclose(_np(a), _np(b), rtol=rtol, atol=atol)


class TestBasicOperations:
    """Basic arithmetic with tinygrad tensors (method and operator forms)."""

    def test_square(self):
        def f(x):
            return (x * x).sum()

        df = tangent.grad(f)
        x = Tensor([1.0, 2.0, 3.0])
        assert _allclose(df(x), 2.0 * x.numpy())

    def test_polynomial(self):
        def f(x):
            return (3.0 * x * x + 2.0 * x + 1.0).sum()

        df = tangent.grad(f)
        x = Tensor([2.0])
        assert _allclose(df(x), np.array([14.0]))

    def test_add_method(self):
        def f(x, y):
            return x.add(y).sum()

        df = tangent.grad(f, wrt=(0,))
        x = Tensor([1.0, 2.0])
        y = Tensor([3.0, 4.0])
        assert _allclose(df(x, y), np.ones(2))

    def test_sub_method(self):
        def f(x, y):
            return x.sub(y).sum()

        x = Tensor([5.0, 6.0])
        y = Tensor([1.0, 2.0])
        assert _allclose(tangent.grad(f, wrt=(0,))(x, y), np.ones(2))
        assert _allclose(tangent.grad(f, wrt=(1,))(x, y), -np.ones(2))

    def test_mul_method(self):
        def f(x, y):
            return x.mul(y).sum()

        df = tangent.grad(f, wrt=(0,))
        x = Tensor([2.0, 3.0])
        y = Tensor([4.0, 5.0])
        assert _allclose(df(x, y), y.numpy())

    def test_div_method(self):
        def f(x, y):
            return x.div(y).sum()

        df = tangent.grad(f, wrt=(0,))
        x = Tensor([4.0, 9.0])
        y = Tensor([2.0, 3.0])
        assert _allclose(df(x, y), 1.0 / y.numpy())

    def test_pow_method(self):
        def f(x):
            return x.pow(3.0).sum()

        df = tangent.grad(f)
        x = Tensor([1.0, 2.0])
        assert _allclose(df(x), 3.0 * x.numpy() ** 2)

    def test_neg_method(self):
        def f(x):
            return x.neg().sum()

        df = tangent.grad(f)
        x = Tensor([1.0, -2.0])
        assert _allclose(df(x), -np.ones(2))


class TestMathFunctions:
    """Elementwise math functions."""

    def test_exp(self):
        def f(x):
            return x.exp().sum()

        df = tangent.grad(f)
        x = Tensor([0.0, 1.0])
        assert _allclose(df(x), np.exp(x.numpy()))

    def test_log(self):
        def f(x):
            return x.log().sum()

        df = tangent.grad(f)
        x = Tensor([1.0, 2.0])
        assert _allclose(df(x), 1.0 / x.numpy())

    def test_sqrt(self):
        def f(x):
            return x.sqrt().sum()

        df = tangent.grad(f)
        x = Tensor([1.0, 4.0])
        assert _allclose(df(x), 1.0 / (2.0 * np.sqrt(x.numpy())))

    def test_sin(self):
        def f(x):
            return x.sin().sum()

        df = tangent.grad(f)
        x = Tensor([0.3, 0.7])
        assert _allclose(df(x), np.cos(x.numpy()))

    def test_cos(self):
        def f(x):
            return x.cos().sum()

        df = tangent.grad(f)
        x = Tensor([0.3, 0.7])
        assert _allclose(df(x), -np.sin(x.numpy()))

    def test_tanh(self):
        def f(x):
            return x.tanh().sum()

        df = tangent.grad(f)
        x = Tensor([0.5, -0.5])
        assert _allclose(df(x), 1.0 - np.tanh(x.numpy()) ** 2)

    def test_atan(self):
        def f(x):
            return x.atan().sum()

        df = tangent.grad(f)
        x = Tensor([0.5, -0.5])
        assert _allclose(df(x), 1.0 / (1.0 + x.numpy() ** 2))


class TestReductions:
    """Reduction operations."""

    def test_sum_all(self):
        def f(x):
            return x.sum()

        df = tangent.grad(f)
        x = Tensor([1.0, 2.0, 3.0])
        assert _allclose(df(x), np.ones(3))

    def test_sum_axis(self):
        def f(x):
            return x.sum(axis=1).sum()

        df = tangent.grad(f)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert _allclose(df(x), np.ones((2, 2)))

    def test_mean(self):
        def f(x):
            return x.mean()

        df = tangent.grad(f)
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        assert _allclose(df(x), np.full(4, 0.25))

    def test_mean_axis(self):
        def f(x):
            return x.mean(axis=0).sum()

        df = tangent.grad(f)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert _allclose(df(x), np.full((2, 2), 0.5))

    def test_sum_keepdim(self):
        def f(x):
            return x.sum(axis=1, keepdim=True).sum()

        df = tangent.grad(f)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert _allclose(df(x), np.ones((2, 2)))


class TestLinearAlgebra:
    """Matrix operations."""

    def test_matmul_matrix(self):
        def f(x, w):
            return x.matmul(w).sum()

        xm = np.random.randn(2, 3).astype(np.float32)
        wm = np.random.randn(3, 4).astype(np.float32)
        x, w = Tensor(xm), Tensor(wm)
        assert _allclose(tangent.grad(f, wrt=(0,))(x, w),
                         tinygrad_ref(f, x, w, wrt=0))
        assert _allclose(tangent.grad(f, wrt=(1,))(x, w),
                         tinygrad_ref(f, x, w, wrt=1))

    def test_matvec(self):
        def f(a, x):
            return a.matmul(x).sum()

        am = np.random.randn(3, 4).astype(np.float32)
        xv = np.random.randn(4).astype(np.float32)
        a, x = Tensor(am), Tensor(xv)
        assert _allclose(tangent.grad(f, wrt=(0,))(a, x),
                         tinygrad_ref(f, a, x, wrt=0))
        assert _allclose(tangent.grad(f, wrt=(1,))(a, x),
                         tinygrad_ref(f, a, x, wrt=1))

    def test_vec_matmul_inner(self):
        def f(x, y):
            return x.matmul(y)

        x = Tensor([1.0, 2.0, 3.0])
        y = Tensor([0.5, -1.0, 2.0])
        dfx = tangent.grad(f, wrt=(0,))
        assert _allclose(dfx(x, y), y.numpy())


class TestShapeOperations:
    """Shape manipulation."""

    def test_reshape(self):
        def f(x):
            return (x.reshape(2, 3) * 2.0).sum()

        df = tangent.grad(f)
        x = Tensor(np.arange(6.0).astype(np.float32))
        assert _allclose(df(x), np.full(6, 2.0))

    def test_transpose(self):
        def f(x):
            return (x.transpose() * 3.0).sum()

        df = tangent.grad(f)
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        assert _allclose(df(x), np.full((2, 3), 3.0))

    def test_permute(self):
        def f(x):
            return x.permute(1, 0).sum()

        df = tangent.grad(f)
        x = Tensor(np.random.randn(2, 3).astype(np.float32))
        assert _allclose(df(x), np.ones((2, 3)))

    def test_squeeze_unsqueeze(self):
        def f(x):
            return x.unsqueeze(0).squeeze(0).sum()

        df = tangent.grad(f)
        x = Tensor([1.0, 2.0, 3.0])
        assert _allclose(df(x), np.ones(3))

    def test_flip(self):
        def f(x):
            return (x.flip(0) * Tensor([1.0, 2.0, 3.0])).sum()

        df = tangent.grad(f)
        x = Tensor([1.0, 1.0, 1.0])
        assert _allclose(df(x), np.array([3.0, 2.0, 1.0]))


class TestSelection:
    """Elementwise selection and clamping."""

    def test_abs(self):
        def f(x):
            return x.abs().sum()

        df = tangent.grad(f)
        x = Tensor([-2.0, 3.0])
        assert _allclose(df(x), np.array([-1.0, 1.0]))

    def test_maximum(self):
        def f(x, y):
            return x.maximum(y).sum()

        x = Tensor([1.0, 5.0])
        y = Tensor([3.0, 2.0])
        dfx = tangent.grad(f, wrt=(0,))
        assert _allclose(dfx(x, y), np.array([0.0, 1.0]))

    def test_clip(self):
        def f(x):
            return x.clip(0.0, 1.0).sum()

        df = tangent.grad(f)
        x = Tensor([-1.0, 0.5, 2.0])
        assert _allclose(df(x), np.array([0.0, 1.0, 0.0]))

    def test_where(self):
        def f(x, y):
            return (x > 0).where(x, y).sum()

        x = Tensor([1.0, -1.0])
        y = Tensor([10.0, 20.0])
        assert _allclose(tangent.grad(f, wrt=(0,))(x, y), np.array([1.0, 0.0]))
        assert _allclose(tangent.grad(f, wrt=(1,))(x, y), np.array([0.0, 1.0]))


class TestActivations:
    """Common neural-network activations."""

    def test_relu(self):
        def f(x):
            return x.relu().sum()

        df = tangent.grad(f)
        x = Tensor([-1.0, 0.5, 2.0])
        assert _allclose(df(x), np.array([0.0, 1.0, 1.0]))

    def test_sigmoid(self):
        def f(x):
            return x.sigmoid().sum()

        df = tangent.grad(f)
        x = Tensor([0.0, 1.0])
        sig = 1.0 / (1.0 + np.exp(-x.numpy()))
        assert _allclose(df(x), sig * (1.0 - sig))

    def test_leaky_relu(self):
        def f(x):
            return x.leaky_relu().sum()

        df = tangent.grad(f)
        x = Tensor([-2.0, 3.0])
        assert _allclose(df(x), np.array([0.01, 1.0]))

    def test_softmax(self):
        def f(x):
            return x.softmax().sum()

        xm = np.random.randn(3, 5).astype(np.float32)
        x = Tensor(xm)
        assert _allclose(tangent.grad(f)(x), tinygrad_ref(f, x, wrt=0))

    def test_log_softmax(self):
        def f(x):
            return x.log_softmax().sum()

        xm = np.random.randn(3, 4).astype(np.float32)
        x = Tensor(xm)
        assert _allclose(tangent.grad(f)(x), tinygrad_ref(f, x, wrt=0))


class TestMLP:
    """A small multi-layer perceptron, the canonical tinygrad use case."""

    def test_mlp_all_args(self):
        def f(x, w, b):
            return x.matmul(w).add(b).relu().sum()

        xm = np.random.randn(2, 3).astype(np.float32)
        wm = np.random.randn(3, 4).astype(np.float32)
        bm = np.random.randn(4).astype(np.float32)
        x, w, b = Tensor(xm), Tensor(wm), Tensor(bm)
        for i in range(3):
            assert _allclose(tangent.grad(f, wrt=(i,))(x, w, b),
                             tinygrad_ref(f, x, w, b, wrt=i))

    def test_mlp_multi_wrt(self):
        def f(x, w, b):
            return x.matmul(w).add(b).relu().sum()

        xm = np.random.randn(2, 3).astype(np.float32)
        wm = np.random.randn(3, 4).astype(np.float32)
        bm = np.random.randn(4).astype(np.float32)
        x, w, b = Tensor(xm), Tensor(wm), Tensor(bm)
        grads = tangent.grad(f, wrt=(0, 1, 2))(x, w, b)
        for i, got in enumerate(grads):
            assert _allclose(got, tinygrad_ref(f, x, w, b, wrt=i))


class TestHigherOrder:
    """Second derivatives through tinygrad code."""

    def test_gradgrad_polynomial(self):
        def f(x):
            return (x * x * x).sum()

        dg = tangent.grad(f)
        ddg = tangent.grad(dg)
        x = Tensor([1.0, 2.0])
        assert _allclose(ddg(x), np.array([6.0, 12.0]))


class TestForwardMode:
    """Forward-mode differentiation of tinygrad code."""

    def test_forward_square(self):
        def f(x):
            return x * x

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        x = Tensor([1.0, 2.0, 3.0])
        seed = Tensor(np.ones(3).astype(np.float32))
        assert _allclose(df(x, seed), 2.0 * x.numpy())

    def test_forward_exp(self):
        def f(x):
            return x.exp()

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        x = Tensor([0.0, 1.0])
        seed = Tensor(np.ones(2).astype(np.float32))
        assert _allclose(df(x, seed), np.exp(x.numpy()))


class TestMethodResolution:
    """The method-resolution machinery and its scoping."""

    def test_numpy_methods_unaffected(self):
        # Even though tinygrad is imported in this module, a NumPy array's
        # .sum() defined in a module that does NOT import tinygrad must still
        # route to the NumPy fallback, not tinygrad. The resolver gates on the
        # differentiated function's own namespace.
        from tests import tinygrad_scope_helper

        df = tangent.grad(tinygrad_scope_helper.np_method_sum)
        result = df(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, np.ones(3))

        dm = tangent.grad(tinygrad_scope_helper.np_method_mean)
        result = dm(np.array([1.0, 2.0, 3.0, 4.0]))
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, np.full(4, 0.25))

    def test_tinygrad_tensor_claimed(self):
        def f(x):
            return x.exp().sum()

        df = tangent.grad(f)
        x = Tensor([0.0, 1.0])
        assert _allclose(df(x), np.exp(x.numpy()))


class TestMoreReductions:
    """prod and cumsum."""

    def test_prod_all(self):
        def f(x):
            return x.prod()

        df = tangent.grad(f)
        x = Tensor([2.0, 3.0, 4.0])
        assert _allclose(df(x), tinygrad_ref(f, x, wrt=0))

    def test_prod_axis(self):
        def f(x):
            return x.prod(axis=1).sum()

        xm = np.random.RandomState(0).rand(2, 3).astype(np.float32) + 0.5
        x = Tensor(xm)
        assert _allclose(tangent.grad(f)(x), tinygrad_ref(f, x, wrt=0))

    def test_cumsum(self):
        def f(x):
            return (x.cumsum(0) * Tensor([1.0, 2.0, 3.0, 4.0])).sum()

        x = Tensor(np.array([0.5, -1.0, 2.0, 0.2], dtype=np.float32))
        assert _allclose(tangent.grad(f)(x), tinygrad_ref(f, x, wrt=0))

    def test_cumsum_axis(self):
        def f(x):
            return x.cumsum(1).sum()

        x = Tensor(np.random.RandomState(1).randn(2, 4).astype(np.float32))
        assert _allclose(tangent.grad(f)(x), tinygrad_ref(f, x, wrt=0))


class TestNormalization:
    """layernorm and batchnorm."""

    def test_layernorm(self):
        def f(x):
            return x.layernorm().sum()

        x = Tensor(np.random.RandomState(0).randn(2, 5).astype(np.float32))
        assert _allclose(tangent.grad(f)(x), tinygrad_ref(f, x, wrt=0))

    def _batchnorm_fn(self):
        def f(x, w, b):
            m = x.mean(axis=(0, 2, 3))
            iv = (x.var(axis=(0, 2, 3)) + 1e-5).rsqrt()
            return x.batchnorm(w, b, m, iv).sum()
        return f

    def _batchnorm_args(self):
        rs = np.random.RandomState(0)
        return (Tensor(rs.randn(2, 3, 4, 4).astype(np.float32)),
                Tensor(rs.randn(3).astype(np.float32)),
                Tensor(rs.randn(3).astype(np.float32)))

    def test_batchnorm_wrt_x(self):
        f = self._batchnorm_fn()
        args = self._batchnorm_args()
        assert _allclose(tangent.grad(f, wrt=(0,))(*args),
                         tinygrad_ref(f, *args, wrt=0))

    def test_batchnorm_wrt_weight(self):
        f = self._batchnorm_fn()
        args = self._batchnorm_args()
        assert _allclose(tangent.grad(f, wrt=(1,))(*args),
                         tinygrad_ref(f, *args, wrt=1))

    def test_batchnorm_wrt_bias(self):
        f = self._batchnorm_fn()
        args = self._batchnorm_args()
        assert _allclose(tangent.grad(f, wrt=(2,))(*args),
                         tinygrad_ref(f, *args, wrt=2))


class TestDot:
    """tinygrad's dot (2-D x 2-D and 2-D x 1-D matmul)."""

    def test_dot_matrix_wrt_x(self):
        def f(x, w):
            return x.dot(w).sum()

        rs = np.random.RandomState(0)
        x, w = Tensor(rs.randn(2, 3).astype(np.float32)), \
            Tensor(rs.randn(3, 4).astype(np.float32))
        assert _allclose(tangent.grad(f, wrt=(0,))(x, w),
                         tinygrad_ref(f, x, w, wrt=0))

    def test_dot_matrix_wrt_w(self):
        def f(x, w):
            return x.dot(w).sum()

        rs = np.random.RandomState(0)
        x, w = Tensor(rs.randn(2, 3).astype(np.float32)), \
            Tensor(rs.randn(3, 4).astype(np.float32))
        assert _allclose(tangent.grad(f, wrt=(1,))(x, w),
                         tinygrad_ref(f, x, w, wrt=1))

    def test_dot_matvec(self):
        def f(x, w):
            return x.dot(w).sum()

        rs = np.random.RandomState(0)
        x, w = Tensor(rs.randn(2, 3).astype(np.float32)), \
            Tensor(rs.randn(3).astype(np.float32))
        assert _allclose(tangent.grad(f, wrt=(0,))(x, w),
                         tinygrad_ref(f, x, w, wrt=0))


class TestConvAndPool:
    """conv2d and avg_pool2d (cross-checked against tinygrad's autodiff)."""

    def _conv_args(self):
        rs = np.random.RandomState(0)
        return (Tensor(rs.randn(2, 3, 6, 6).astype(np.float32)),
                Tensor(rs.randn(4, 3, 3, 3).astype(np.float32)),
                Tensor(rs.randn(4).astype(np.float32)))

    def test_conv2d_wrt_input(self):
        def f(x, w):
            return x.conv2d(w).sum()

        x, w, _ = self._conv_args()
        assert _allclose(tangent.grad(f, wrt=(0,))(x, w),
                         tinygrad_ref(f, x, w, wrt=0))

    def test_conv2d_wrt_weight(self):
        def f(x, w):
            return x.conv2d(w).sum()

        x, w, _ = self._conv_args()
        assert _allclose(tangent.grad(f, wrt=(1,))(x, w),
                         tinygrad_ref(f, x, w, wrt=1))

    def test_conv2d_padding(self):
        def f(x, w):
            return x.conv2d(w, padding=1).sum()

        x, w, _ = self._conv_args()
        assert _allclose(tangent.grad(f, wrt=(0,))(x, w),
                         tinygrad_ref(f, x, w, wrt=0))
        assert _allclose(tangent.grad(f, wrt=(1,))(x, w),
                         tinygrad_ref(f, x, w, wrt=1))

    def test_conv2d_stride2_input_grad(self):
        # The input gradient supports stride > 1 via a transposed conv.
        def f(x, w):
            return x.conv2d(w, stride=2).sum()

        x, w, _ = self._conv_args()
        assert _allclose(tangent.grad(f, wrt=(0,))(x, w),
                         tinygrad_ref(f, x, w, wrt=0))

    def test_conv2d_stride2_weight_grad(self):
        # The weight gradient handles the floor-remainder of the output-size
        # computation by cropping the swapped stride/dilation correlation.
        def f(x, w):
            return x.conv2d(w, stride=2).sum()

        x, w, _ = self._conv_args()
        assert _allclose(tangent.grad(f, wrt=(1,))(x, w),
                         tinygrad_ref(f, x, w, wrt=1))

    def test_conv2d_dilation_weight_grad(self):
        def f(x, w):
            return x.conv2d(w, dilation=2).sum()

        x, w, _ = self._conv_args()
        assert _allclose(tangent.grad(f, wrt=(1,))(x, w),
                         tinygrad_ref(f, x, w, wrt=1))

    def test_conv2d_stride_dilation_weight_grad(self):
        def f(x, w):
            return x.conv2d(w, stride=2, dilation=2).sum()

        rs = np.random.RandomState(0)
        x = Tensor(rs.randn(2, 3, 9, 9).astype(np.float32))
        w = Tensor(rs.randn(4, 3, 3, 3).astype(np.float32))
        assert _allclose(tangent.grad(f, wrt=(1,))(x, w),
                         tinygrad_ref(f, x, w, wrt=1))

    def test_conv2d_bias(self):
        def f(x, w, b):
            return x.conv2d(w, b).sum()

        x, w, b = self._conv_args()
        assert _allclose(tangent.grad(f, wrt=(2,))(x, w, b),
                         tinygrad_ref(f, x, w, b, wrt=2))

    def test_avg_pool2d(self):
        def f(x):
            return x.avg_pool2d().sum()

        x = Tensor(np.random.RandomState(0).randn(2, 3, 8, 8).astype(np.float32))
        assert _allclose(tangent.grad(f)(x), tinygrad_ref(f, x, wrt=0))

    def test_avg_pool2d_stride_padding(self):
        def f(x):
            return x.avg_pool2d(kernel_size=3, stride=2, padding=1).sum()

        x = Tensor(np.random.RandomState(0).randn(2, 3, 8, 8).astype(np.float32))
        assert _allclose(tangent.grad(f)(x), tinygrad_ref(f, x, wrt=0))

    def test_max_pool2d(self):
        def f(x):
            return x.max_pool2d().sum()

        x = Tensor(np.random.RandomState(0).randn(2, 3, 8, 8).astype(np.float32))
        assert _allclose(tangent.grad(f)(x), tinygrad_ref(f, x, wrt=0))

    def test_max_pool2d_overlap_and_ties(self):
        def f(x):
            return x.max_pool2d(kernel_size=3, stride=2).sum()

        rs = np.random.RandomState(0)
        # Round to create duplicate maxima (ties) within windows.
        x = Tensor((np.round(rs.randn(2, 3, 8, 8) * 2) / 2).astype(np.float32))
        assert _allclose(tangent.grad(f)(x), tinygrad_ref(f, x, wrt=0))

    def test_max_pool2d_padding(self):
        def f(x):
            return x.max_pool2d(padding=1).sum()

        x = Tensor(np.random.RandomState(0).randn(2, 3, 8, 8).astype(np.float32))
        assert _allclose(tangent.grad(f)(x), tinygrad_ref(f, x, wrt=0))

    def test_max_pool2d_unsupported_raises(self):
        def f(x):
            return x.max_pool2d(dilation=2).sum()

        df = tangent.grad(f)
        x = Tensor(np.random.RandomState(0).randn(1, 1, 8, 8).astype(np.float32))
        with pytest.raises(NotImplementedError):
            df(x)


class TestMatmulOperator:
    """The @ operator on tinygrad tensors."""

    def test_matvec(self):
        def f(x, w):
            return (x @ w).sum()

        rs = np.random.RandomState(0)
        x, w = Tensor(rs.randn(2, 3).astype(np.float32)), \
            Tensor(rs.randn(3).astype(np.float32))
        assert _allclose(tangent.grad(f, wrt=(0,))(x, w),
                         tinygrad_ref(f, x, w, wrt=0))
        assert _allclose(tangent.grad(f, wrt=(1,))(x, w),
                         tinygrad_ref(f, x, w, wrt=1))

    def test_matmul(self):
        def f(x, w):
            return (x @ w).sum()

        rs = np.random.RandomState(0)
        x, w = Tensor(rs.randn(2, 3).astype(np.float32)), \
            Tensor(rs.randn(3, 4).astype(np.float32))
        assert _allclose(tangent.grad(f, wrt=(0,))(x, w),
                         tinygrad_ref(f, x, w, wrt=0))
        assert _allclose(tangent.grad(f, wrt=(1,))(x, w),
                         tinygrad_ref(f, x, w, wrt=1))

    def test_vector_inner_product(self):
        def f(x, y):
            return x @ y

        x = Tensor([1.0, 2.0, 3.0])
        y = Tensor([0.5, -1.0, 2.0])
        assert _allclose(tangent.grad(f, wrt=(0,))(x, y), y.numpy())
        assert _allclose(tangent.grad(f, wrt=(1,))(x, y), x.numpy())


class TestIndexing:
    """Subscript reads on tinygrad tensors (generic Tangent machinery)."""

    def test_single_index(self):
        def f(x):
            return x[2] * 3.0 + x[2]

        df = tangent.grad(f)
        w = Tensor(np.arange(6.0).astype(np.float32))
        assert _allclose(df(w), np.array([0., 0., 4., 0., 0., 0.]))

    def test_slice(self):
        def f(x):
            return x[0:2].sum()

        df = tangent.grad(f)
        x = Tensor(np.ones(5, dtype=np.float32))
        assert _allclose(df(x), np.array([1., 1., 0., 0., 0.]))

    def test_row_index(self):
        def f(x):
            return (x[1] * 2.0).sum()

        df = tangent.grad(f)
        x = Tensor(np.ones((3, 4), dtype=np.float32))
        expect = np.zeros((3, 4))
        expect[1] = 2.0
        assert _allclose(df(x), expect)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
