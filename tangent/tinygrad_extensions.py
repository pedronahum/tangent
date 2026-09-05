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
"""tinygrad extensions for Tangent automatic differentiation.

This module provides gradient definitions for tinygrad, enabling Tangent to
differentiate functions that use tinygrad tensors.

Unlike NumPy/JAX/TF/torch, tinygrad exposes its operations as *methods* on the
tensor object (``x.relu()``, ``x.sum()``, ``x.matmul(w)``) rather than as
module-level functions. Because Tangent resolves calls against the
differentiated function's global namespace, a method call on a computed value
(``x = a + b`` followed by ``x.sum()``) does not resolve statically. This
module therefore also registers a :class:`tangent.utils.MethodResolver` so that
such unresolved method calls are rewritten into unbound calls
(``Tensor.sum(x)``) and annotated with the corresponding tinygrad method,
letting the adjoints registered here apply. See ``annotate.ResolveCalls``.

The generated gradient code is itself a tinygrad program (a graph of
``Tensor.*`` calls) that tinygrad's own compiler schedules and fuses.

Example:
    from tinygrad import Tensor
    import tangent

    def f(x):
        return x.matmul(w).add(b).relu().sum()

    df = tangent.grad(f)
    gradient = df(x)
"""
from __future__ import absolute_import

import warnings

try:
    from tinygrad import Tensor
except ImportError as e:
    warnings.warn(f"tinygrad not available: {e}. Install with: pip install tinygrad")
    raise

import gast
import numpy as np

from tangent import non_differentiable
from tangent import utils
from tangent.grads import adjoint
from tangent.tangents import tangent_
from tangent.utils import register_shape_function
from tangent.utils import register_init_grad

TensorType = Tensor


# ============================================================================
# Runtime helpers
# ============================================================================

def tg_shape(x):
    """Shape of a tinygrad tensor as a list."""
    return list(x.shape)


def tg_size(x, axis):
    """Number of elements reduced over ``axis`` (all elements if None)."""
    if axis is not None and isinstance(axis, int):
        axis = (axis,)
    axis_shape = x.shape if axis is None else tuple(x.shape[a] for a in axis)
    return max(int(np.prod(axis_shape)), 1)


def tg_seed(g, like):
    """Coerce a gradient seed to a tinygrad tensor matching ``like``'s dtype.

    Gradient seeds arrive as plain Python floats when a gradient function is
    called without an explicit init_grad, and as NumPy arrays after passing
    through NumPy-generic helpers. tinygrad does not mix with those types in
    arithmetic, so every adjoint coerces its incoming seed first.
    """
    if isinstance(g, Tensor):
        return g
    seed = Tensor(np.asarray(g))
    if isinstance(like, Tensor):
        seed = seed.cast(like.dtype)
    return seed


def tg_shapes_match(a, b):
    """Check if two values have matching shapes (scalars have shape ())."""
    sa = tuple(a.shape) if isinstance(a, Tensor) else ()
    sb = tuple(b.shape) if isinstance(b, Tensor) else ()
    # A scalar seed (shape ()) broadcasts to any output shape, mirroring
    # Tangent's NumPy semantics (array_shapes_match): the default seed 1.0 is
    # accepted for an array-valued output and computes the gradient of the sum.
    if sa == () or sb == ():
        return True
    return sa == sb


def tg_inv_perm(order):
    """Inverse of a permutation (e.g. for the adjoint of ``permute``)."""
    inv = [0] * len(order)
    for i, ax in enumerate(order):
        inv[ax] = i
    return inv


def tg_clip_mask(x, min_, max_):
    """Gradient mask for clip/clamp: 1 where x is inside the bounds, else 0.

    Kept as a plain (non-differentiable) helper so the ``is not None`` checks
    never end up comparing against a numeric literal in generated code.
    """
    mask = Tensor.ones_like(x)
    if min_ is not None:
        mask = (x >= min_).where(mask, 0.0)
    if max_ is not None:
        mask = (x <= max_).where(mask, 0.0)
    return mask


def tg_broadcast_axis(x, param, axis):
    """Reshape ``param`` so it broadcasts along ``axis`` of ``x``.

    tinygrad's batchnorm/normalization params (weight, bias, mean, invstd) have
    reduced rank (e.g. ``(C,)`` for a 4-D ``(N, C, H, W)`` input). Arithmetic
    with the full-rank tensor only aligns if the param is first reshaped to put
    its dims on the normalized axes and 1s elsewhere (e.g. ``(1, C, 1, 1)``).
    """
    ndim = len(x.shape)
    axes = tuple(a % ndim for a in
                 ((axis,) if isinstance(axis, int) else tuple(axis)))
    shape = [1] * ndim
    pshape = tuple(param.shape)
    for i, a in enumerate(sorted(axes)):
        shape[a] = pshape[i] if i < len(pshape) else 1
    return param.reshape(tuple(shape))


def tg_reduce_except(x, axis):
    """Sum ``x`` over every axis except ``axis``.

    The gradient of a normalization param (weight/bias/mean/invstd of shape
    ``(C,)``) is the incoming gradient summed over all *other* axes. Those axes
    are not trailing-aligned with the param shape, so `unbroadcast` (which
    assumes trailing alignment) cannot compute this reduction.
    """
    ndim = len(x.shape)
    axes = set(a % ndim for a in
               ((axis,) if isinstance(axis, int) else tuple(axis)))
    reduce_axes = tuple(a for a in range(ndim) if a not in axes)
    if not reduce_axes:
        return x
    return x.sum(axis=reduce_axes)


def _tg_spatial_tuples(x, stride, dilation, padding):
    """Normalize conv2d stride/dilation/padding to per-spatial-axis tuples.

    Only an int or a symmetric per-axis tuple of paddings is supported; the
    explicit per-side form raises.
    """
    n = len(x.shape) - 2
    stride = (stride,) * n if isinstance(stride, int) else tuple(stride)
    dilation = (dilation,) * n if isinstance(dilation, int) else tuple(dilation)
    if isinstance(padding, int):
        pads = (padding,) * n
    else:
        pads = tuple(padding)
        if len(pads) != n:
            raise NotImplementedError(
                'tangent tinygrad: only int or per-axis symmetric padding is '
                'supported for conv2d/pool gradients, got %r' % (padding,))
    return stride, dilation, pads


def tg_conv2d_grad_input(dz, weight, x, stride, dilation, padding, groups=1):
    """Input gradient of conv2d, expressed as a transposed convolution."""
    stride, dilation, pads = _tg_spatial_tuples(x, stride, dilation, padding)
    n = len(x.shape) - 2
    output_padding = tuple(
        x.shape[2 + i] - ((dz.shape[2 + i] - 1) * stride[i]
                          + (weight.shape[2 + i] - 1) * dilation[i] + 1
                          - 2 * pads[i])
        for i in range(n))
    return dz.conv_transpose2d(weight, stride=stride, dilation=dilation,
                               padding=pads, output_padding=output_padding,
                               groups=groups)


def tg_conv2d_grad_weight(x, dz, weight, stride, dilation, padding, groups):
    """Weight gradient of conv2d (groups=1; any stride/dilation/padding).

    dw is the correlation of x and dz, expressible as a conv2d with batch and
    channel axes transposed and stride/dilation swapped. When the forward drops
    trailing input elements (the floor in the output-size computation), that
    conv overshoots the kernel size by floor(r/d) per axis and is cropped.
    """
    if groups != 1:
        raise NotImplementedError(
            'tangent tinygrad: conv2d weight gradient with groups != 1 is not '
            'supported')
    stride, dilation, pads = _tg_spatial_tuples(x, stride, dilation, padding)
    k = tuple(weight.shape[2:])
    grad_conv = Tensor.conv2d(x.transpose(1, 0), dz.transpose(1, 0),
                              stride=dilation, dilation=stride, padding=pads)
    if any((x.shape[2 + i] + 2 * pads[i] - dilation[i] * (k[i] - 1) - 1)
           % stride[i] // dilation[i] for i in range(len(k))):
        grad_conv = grad_conv[(slice(None),) * 2 + tuple(slice(None, ki)
                                                         for ki in k)]
    return grad_conv.transpose(0, 1)


def tg_avg_pool2d_grad_input(dz, x, kernel_size, stride, dilation, padding,
                             ceil_mode, count_include_pad):
    """Input gradient of avg_pool2d (default count_include_pad, no ceil_mode).

    Average pooling equals a depthwise conv with a uniform 1/k^2 kernel, so its
    input gradient is the transposed conv with the same uniform kernel.
    """
    if ceil_mode:
        raise NotImplementedError(
            'tangent tinygrad: avg_pool2d gradient with ceil_mode=True is not '
            'supported')
    if not count_include_pad:
        raise NotImplementedError(
            'tangent tinygrad: avg_pool2d gradient with count_include_pad='
            'False is not supported')
    k = (kernel_size,) * 2 if isinstance(kernel_size, int) else tuple(kernel_size)
    if any(ki != 1 for ki in ((dilation,) * 2 if isinstance(dilation, int)
                              else tuple(dilation))):
        raise NotImplementedError(
            'tangent tinygrad: avg_pool2d gradient with dilation > 1 is not '
            'supported')
    channels = x.shape[1]
    weight = Tensor.ones((channels, 1) + k, dtype=x.dtype) / float(k[0] * k[1])
    return tg_conv2d_grad_input(dz, weight, x, stride if stride is not None else k,
                                1, padding, groups=channels)


def tg_max_pool2d_grad_input(dz, x, kernel_size, stride, dilation, padding,
                             ceil_mode, return_indices):
    """Input gradient of max_pool2d.

    The incoming gradient of each window goes to the window's maximum element
    (split evenly among ties). Construction: unfold the (padded) input into
    windows, build the argmax mask, then fold the masked gradients back with
    one transposed conv per window offset using a single-1 (delta) depthwise
    kernel — a strided scatter that sums overlapping windows. Supports 3-D
    (C,H,W) and 4-D (N,C,H,W) inputs, dilation=1, no ceil_mode, symmetric
    padding.
    """
    if ceil_mode:
        raise NotImplementedError(
            'tangent tinygrad: max_pool2d gradient with ceil_mode=True is not '
            'supported')
    if return_indices:
        raise NotImplementedError(
            'tangent tinygrad: max_pool2d gradient with return_indices=True is '
            'not supported')
    k = (kernel_size,) * 2 if isinstance(kernel_size, int) else tuple(kernel_size)
    if len(k) != 2:
        raise NotImplementedError(
            'tangent tinygrad: max_pool2d gradient supports 2-D kernels only')
    d = (dilation,) * 2 if isinstance(dilation, int) else tuple(dilation)
    if any(di != 1 for di in d):
        raise NotImplementedError(
            'tangent tinygrad: max_pool2d gradient with dilation > 1 is not '
            'supported')
    s = k if stride is None else ((stride,) * 2 if isinstance(stride, int)
                                  else tuple(stride))
    p = (padding,) * 2 if isinstance(padding, int) else tuple(padding)
    if len(p) != 2:
        raise NotImplementedError(
            'tangent tinygrad: max_pool2d gradient supports int or 2-tuple '
            'symmetric padding only')
    if x.ndim not in (3, 4):
        raise NotImplementedError(
            'tangent tinygrad: max_pool2d gradient supports 3-D/4-D inputs')
    channels = x.shape[1] if x.ndim == 4 else x.shape[0]
    # Work in 4-D; unfold appends the window dimension last, so two chained
    # unfolds yield (N, C, Ho, Wo, kh, kw).
    x4 = x if x.ndim == 4 else x.reshape((1,) + x.shape)
    dz4 = dz if x.ndim == 4 else dz.reshape((1,) + dz.shape)
    xp = x4.pad(((0, 0), (0, 0), p, p), value=x4.dtype.min) if any(p) else x4
    xw = xp.unfold(2, k[0], s[0]).unfold(3, k[1], s[1])
    mask = xw == xw.max(axis=(4, 5), keepdim=True)
    ties = mask.sum(axis=(4, 5), keepdim=True)
    n, _, ho, wo = dz4.shape
    dye = dz4.reshape((n, channels, ho, wo, 1, 1)).expand(
        (n, channels, ho, wo, k[0], k[1]))
    dxw = dye * mask / ties                   # (N, C, Ho, Wo, kh, kw)
    dxwp = dxw.permute(4, 5, 0, 1, 2, 3)      # (kh, kw, N, C, Ho, Wo)
    eye = Tensor.eye(k[0] * k[1], dtype=x.dtype)
    padded_h, padded_w = xp.shape[2], xp.shape[3]
    total = None
    for pi in range(k[0]):
        for qi in range(k[1]):
            kernel = eye[pi * k[1] + qi].reshape(1, 1, k[0], k[1]).expand(
                channels, 1, k[0], k[1])
            output_padding = (padded_h - ((ho - 1) * s[0] + k[0]),
                              padded_w - ((wo - 1) * s[1] + k[1]))
            contrib = dxwp[pi, qi].conv_transpose2d(
                kernel, stride=s, padding=0, output_padding=output_padding,
                groups=channels)
            total = contrib if total is None else total + contrib
    if any(p):
        total = total.pad(((0, 0), (0, 0), (-p[0], -p[0]), (-p[1], -p[1])))
    return total if x.ndim == 4 else total.reshape(x.shape)


def tg_unbroadcast(array, like):
    """Reverse broadcasting for tinygrad tensors, matching ``like``'s shape."""
    if not isinstance(array, Tensor):
        array = tg_seed(array, like)
    like_shape = tuple(like.shape) if isinstance(like, Tensor) else ()
    axis = utils.create_unbroadcast_axis(like_shape, tuple(array.shape))
    if axis:
        array = array.sum(axis=tuple(axis))
    if tuple(array.shape) == like_shape:
        return array
    # A size-1 gradient broadcast back to a larger operand shape (e.g. scalar
    # seeds in second-order derivatives) expands rather than reshapes; a
    # reshape from size 1 to a larger size is invalid.
    if like_shape and int(np.prod(array.shape or (1,))) == 1:
        return array.expand(like_shape)
    return array.reshape(like_shape)


def tg_unreduce(array, shape, axis, keepdims):
    """Reverse a reduction over ``axis`` for tinygrad tensors."""
    if not isinstance(array, Tensor):
        array = tg_seed(array, None)
    if axis is not None and (keepdims is False):
        if isinstance(axis, int):
            axis = (axis,)
        for ax in sorted(a % len(shape) for a in axis):
            array = array.unsqueeze(ax)
    return array.expand(tuple(shape))


# ============================================================================
# Registrations
# ============================================================================

register_shape_function(TensorType, tg_shape)
register_init_grad(TensorType, Tensor.zeros_like)

non_differentiable.register_non_differentiable_functions(
    Tensor.zeros, Tensor.ones, Tensor.full,
    Tensor.zeros_like, Tensor.ones_like, Tensor.full_like,
    Tensor.rand, Tensor.randn, Tensor.randint,
    Tensor.arange, Tensor.linspace, Tensor.eye,
    Tensor.argmax, Tensor.argmin,
    tg_shape, tg_size, tg_seed, tg_inv_perm, tg_clip_mask, tg_broadcast_axis,
    tg_reduce_except, tg_conv2d_grad_input, tg_conv2d_grad_weight,
    tg_avg_pool2d_grad_input, tg_max_pool2d_grad_input,
)


def _tg_add(l, r):
    if not isinstance(l, Tensor):
        l = tg_seed(l, r)
    if not isinstance(r, Tensor):
        r = tg_seed(r, l)
    return l + r


for _t in (TensorType, float, int, np.ndarray):
    for _u in (TensorType, float, int, np.ndarray):
        if _t is TensorType or _u is TensorType:
            try:
                utils.register_add_grad(_t, _u, _tg_add)
            except ValueError:
                pass

for _t in (TensorType, float, int, np.ndarray):
    for _u in (TensorType, float, int, np.ndarray):
        if _t is TensorType or _u is TensorType:
            try:
                utils.register_shape_checker(_t, _u, tg_shapes_match)
            except ValueError:
                pass

try:
    utils.unbroadcasters[TensorType] = tg_unbroadcast
except (AttributeError, KeyError):
    pass

try:
    utils.unreducers[TensorType] = tg_unreduce
except (AttributeError, KeyError):
    pass


def tg_matmul_grad_x(dz, x, y):
    """d[x] for z = x @ y, mirroring the matmul adjoint case split."""
    dz = tg_seed(dz, x)
    if len(x.shape) == 1 and len(y.shape) == 1:
        return dz * y
    if len(x.shape) == 2 and len(y.shape) == 1:
        return dz.unsqueeze(1).matmul(y.unsqueeze(0))
    if len(x.shape) == 1 and len(y.shape) == 2:
        return dz.matmul(y.transpose())
    if len(x.shape) == 2 and len(y.shape) == 2:
        return dz.matmul(y.transpose())
    return dz.matmul(y.transpose(-2, -1))


def tg_matmul_grad_y(dz, x, y):
    """d[y] for z = x @ y, mirroring the matmul adjoint case split."""
    dz = tg_seed(dz, y)
    if len(x.shape) == 1 and len(y.shape) == 1:
        return dz * x
    if len(x.shape) == 1 and len(y.shape) == 2:
        return x.unsqueeze(1).matmul(dz.unsqueeze(0))
    if len(x.shape) == 2 and len(y.shape) == 1:
        return x.transpose().matmul(dz)
    if len(x.shape) == 2 and len(y.shape) == 2:
        return x.transpose().matmul(dz)
    return x.transpose(-2, -1).matmul(dz)


utils.register_matmul_grad(TensorType, tg_matmul_grad_x, tg_matmul_grad_y)


# ----------------------------------------------------------------------------
# Method-call resolution
# ----------------------------------------------------------------------------
# Map method names to the corresponding unbound tinygrad method. The value is
# the exact object the @adjoint/@tangent_ rules are registered against.
_TG_METHODS = {
    # Elementwise unary
    'relu': Tensor.relu,
    'leaky_relu': Tensor.leaky_relu,
    'elu': Tensor.elu,
    'gelu': Tensor.gelu,
    'silu': Tensor.silu,
    'swish': Tensor.swish,
    'softplus': Tensor.softplus,
    'softsign': Tensor.softsign,
    'sigmoid': Tensor.sigmoid,
    'tanh': Tensor.tanh,
    'exp': Tensor.exp,
    'exp2': Tensor.exp2,
    'log': Tensor.log,
    'log2': Tensor.log2,
    'sqrt': Tensor.sqrt,
    'rsqrt': Tensor.rsqrt,
    'square': Tensor.square,
    'reciprocal': Tensor.reciprocal,
    'neg': Tensor.neg,
    'abs': Tensor.abs,
    'sign': Tensor.sign,
    'sin': Tensor.sin,
    'cos': Tensor.cos,
    'tan': Tensor.tan,
    'asin': Tensor.asin,
    'acos': Tensor.acos,
    'atan': Tensor.atan,
    'sinh': Tensor.sinh,
    'cosh': Tensor.cosh,
    'floor': Tensor.floor,
    'ceil': Tensor.ceil,
    'round': Tensor.round,
    # Elementwise binary
    'mul': Tensor.mul,
    'add': Tensor.add,
    'sub': Tensor.sub,
    'div': Tensor.div,
    'pow': Tensor.pow,
    'maximum': Tensor.maximum,
    'minimum': Tensor.minimum,
    'where': Tensor.where,
    'clip': Tensor.clip,
    'clamp': Tensor.clamp,
    # Reductions
    'sum': Tensor.sum,
    'mean': Tensor.mean,
    'max': Tensor.max,
    'min': Tensor.min,
    'std': Tensor.std,
    'var': Tensor.var,
    'prod': Tensor.prod,
    'cumsum': Tensor.cumsum,
    'logsumexp': Tensor.logsumexp,
    # Linear algebra / NN
    'matmul': Tensor.matmul,
    'dot': Tensor.dot,
    'softmax': Tensor.softmax,
    'log_softmax': Tensor.log_softmax,
    'layernorm': Tensor.layernorm,
    'batchnorm': Tensor.batchnorm,
    'conv2d': Tensor.conv2d,
    'max_pool2d': Tensor.max_pool2d,
    'avg_pool2d': Tensor.avg_pool2d,
    'argmax': Tensor.argmax,
    'argmin': Tensor.argmin,
    # Shape manipulation
    'reshape': Tensor.reshape,
    'transpose': Tensor.transpose,
    'permute': Tensor.permute,
    'squeeze': Tensor.squeeze,
    'unsqueeze': Tensor.unsqueeze,
    'expand': Tensor.expand,
    'flip': Tensor.flip,
}


def _tg_uses_backend(namespace):
    """True when the differentiated function's namespace uses tinygrad."""
    if 'tinygrad' in namespace:
        return True
    return any(v is TensorType for v in namespace.values())


def _tg_base_node(namespace):
    """gast expression node for the tinygrad Tensor class in this namespace."""
    for name, value in namespace.items():
        if value is TensorType:
            return gast.Name(id=name, ctx=gast.Load(), annotation=None)
    return gast.Attribute(
        value=gast.Name(id='tinygrad', ctx=gast.Load(), annotation=None),
        attr='Tensor', ctx=gast.Load())


utils.register_method_resolver(utils.MethodResolver(
    methods=_TG_METHODS,
    uses_backend=_tg_uses_backend,
    base_node=_tg_base_node,
    # These take (shape_or_order, *args); pack multiple positional args into a
    # single tuple so the adjoint template gets one bindable parameter.
    tuple_arg_methods=('reshape', 'permute', 'expand', 'flip'),
))


# ============================================================================
# Reverse-mode (adjoint) gradient definitions
# ============================================================================

# --- Elementwise unary ---

@adjoint(Tensor.relu)
def adjoint_relu(y, x):
    d[x] = tangent.tg_seed(d[y], x) * (x > 0)


@adjoint(Tensor.leaky_relu)
def adjoint_leaky_relu(y, x, neg_slope=0.01):
    d[x] = tangent.tg_seed(d[y], x) * ((x > 0) + neg_slope * (x <= 0))


@adjoint(Tensor.elu)
def adjoint_elu(y, x, alpha=1.0):
    d[x] = tangent.tg_seed(d[y], x) * ((x > 0) + alpha * x.exp() * (x <= 0))


@adjoint(Tensor.gelu)
def adjoint_gelu(y, x, approximate='tanh'):
    # d/dx gelu(x) = 0.5*(1+tanh(z)) + x*sech^2(z)*dz/dx, z=sqrt(2/pi)*(x+0.044715x^3)
    z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    t = z.tanh()
    dz = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
    d[x] = tangent.tg_seed(d[y], x) * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dz)


@adjoint(Tensor.silu)
def adjoint_silu(y, x):
    s = x.sigmoid()
    d[x] = tangent.tg_seed(d[y], x) * (s + x * s * (1.0 - s))


@adjoint(Tensor.softplus)
def adjoint_softplus(y, x, beta=1.0):
    d[x] = tangent.tg_seed(d[y], x) * (beta * x).sigmoid()


@adjoint(Tensor.softsign)
def adjoint_softsign(y, x):
    d[x] = tangent.tg_seed(d[y], x) / ((1.0 + x.abs()) * (1.0 + x.abs()))


@adjoint(Tensor.sigmoid)
def adjoint_sigmoid(y, x):
    d[x] = tangent.tg_seed(d[y], x) * y * (1.0 - y)


@adjoint(Tensor.tanh)
def adjoint_tanh(y, x):
    d[x] = tangent.tg_seed(d[y], x) * (1.0 - y * y)


@adjoint(Tensor.exp)
def adjoint_exp(y, x):
    d[x] = tangent.tg_seed(d[y], x) * y


@adjoint(Tensor.exp2)
def adjoint_exp2(y, x):
    d[x] = tangent.tg_seed(d[y], x) * y * 0.6931471805599453


@adjoint(Tensor.log)
def adjoint_log(y, x):
    d[x] = tangent.tg_seed(d[y], x) / x


@adjoint(Tensor.log2)
def adjoint_log2(y, x):
    d[x] = tangent.tg_seed(d[y], x) / (x * 0.6931471805599453)


@adjoint(Tensor.sqrt)
def adjoint_sqrt(y, x):
    d[x] = tangent.tg_seed(d[y], x) / (2.0 * y)


@adjoint(Tensor.rsqrt)
def adjoint_rsqrt(y, x):
    d[x] = tangent.tg_seed(d[y], x) * (-0.5) * x.rsqrt() / x


@adjoint(Tensor.square)
def adjoint_square(y, x):
    d[x] = tangent.tg_seed(d[y], x) * 2.0 * x


@adjoint(Tensor.reciprocal)
def adjoint_reciprocal(y, x):
    d[x] = -tangent.tg_seed(d[y], x) / (x * x)


@adjoint(Tensor.neg)
def adjoint_neg(y, x):
    d[x] = -tangent.tg_seed(d[y], x)


@adjoint(Tensor.abs)
def adjoint_abs(y, x):
    d[x] = tangent.tg_seed(d[y], x) * x.sign()


@adjoint(Tensor.sign)
def adjoint_sign(y, x):
    d[x] = tangent.init_grad(x)


@adjoint(Tensor.sin)
def adjoint_sin(y, x):
    d[x] = tangent.tg_seed(d[y], x) * x.cos()


@adjoint(Tensor.cos)
def adjoint_cos(y, x):
    d[x] = -tangent.tg_seed(d[y], x) * x.sin()


@adjoint(Tensor.tan)
def adjoint_tan(y, x):
    cx = x.cos()
    d[x] = tangent.tg_seed(d[y], x) / (cx * cx)


@adjoint(Tensor.asin)
def adjoint_asin(y, x):
    d[x] = tangent.tg_seed(d[y], x) / (1.0 - x * x).sqrt()


@adjoint(Tensor.acos)
def adjoint_acos(y, x):
    d[x] = -tangent.tg_seed(d[y], x) / (1.0 - x * x).sqrt()


@adjoint(Tensor.atan)
def adjoint_atan(y, x):
    d[x] = tangent.tg_seed(d[y], x) / (1.0 + x * x)


@adjoint(Tensor.sinh)
def adjoint_sinh(y, x):
    d[x] = tangent.tg_seed(d[y], x) * x.cosh()


@adjoint(Tensor.cosh)
def adjoint_cosh(y, x):
    d[x] = tangent.tg_seed(d[y], x) * x.sinh()


@adjoint(Tensor.floor)
def adjoint_floor(y, x):
    d[x] = tangent.init_grad(x)


@adjoint(Tensor.ceil)
def adjoint_ceil(y, x):
    d[x] = tangent.init_grad(x)


@adjoint(Tensor.round)
def adjoint_round(y, x):
    d[x] = tangent.init_grad(x)


# --- Elementwise binary ---

@adjoint(Tensor.add)
def adjoint_add(z, x, y):
    dz = tangent.tg_seed(d[z], x)
    d[x] = tangent.unbroadcast(dz, x)
    d[y] = tangent.unbroadcast(dz, y)


@adjoint(Tensor.sub)
def adjoint_sub(z, x, y):
    dz = tangent.tg_seed(d[z], x)
    d[x] = tangent.unbroadcast(dz, x)
    d[y] = tangent.unbroadcast(-dz, y)


@adjoint(Tensor.mul)
def adjoint_mul(z, x, y):
    dz = tangent.tg_seed(d[z], x)
    d[x] = tangent.unbroadcast(dz * y, x)
    d[y] = tangent.unbroadcast(dz * x, y)


@adjoint(Tensor.div)
def adjoint_div(z, x, y):
    dz = tangent.tg_seed(d[z], x)
    d[x] = tangent.unbroadcast(dz / y, x)
    d[y] = tangent.unbroadcast(-dz * x / (y * y), y)


@adjoint(Tensor.pow)
def adjoint_pow(y, x, n):
    d[x] = tangent.unbroadcast(tangent.tg_seed(d[y], x) * n * x.pow(n - 1), x)


@adjoint(Tensor.maximum)
def adjoint_maximum(z, x, y):
    dz = tangent.tg_seed(d[z], x)
    d[x] = tangent.unbroadcast(dz * (x >= y), x)
    d[y] = tangent.unbroadcast(dz * (y > x), y)


@adjoint(Tensor.minimum)
def adjoint_minimum(z, x, y):
    dz = tangent.tg_seed(d[z], x)
    d[x] = tangent.unbroadcast(dz * (x <= y), x)
    d[y] = tangent.unbroadcast(dz * (y < x), y)


@adjoint(Tensor.where)
def adjoint_where(z, cond, x, y):
    dz = tangent.tg_seed(d[z], x)
    d[x] = tangent.unbroadcast(cond.where(dz, 0.0), x)
    d[y] = tangent.unbroadcast(cond.where(0.0, dz), y)


@adjoint(Tensor.clip)
def adjoint_clip(y, x, min_=None, max_=None):
    d[x] = tangent.tg_seed(d[y], x) * tangent.tg_clip_mask(x, min_, max_)


@adjoint(Tensor.clamp)
def adjoint_clamp(y, x, min_=None, max_=None):
    d[x] = tangent.tg_seed(d[y], x) * tangent.tg_clip_mask(x, min_, max_)


# --- Reductions ---

@adjoint(Tensor.sum)
def adjoint_sum(y, x, axis=None, keepdim=False):
    d[x] = tangent.unreduce(tangent.tg_seed(d[y], x), tangent.tg_shape(x), axis, keepdim)


@adjoint(Tensor.mean)
def adjoint_mean(y, x, axis=None, keepdim=False):
    n = tangent.tg_size(x, axis)
    d[x] = tangent.unreduce(tangent.tg_seed(d[y], x), tangent.tg_shape(x), axis, keepdim) / n


@adjoint(Tensor.max)
def adjoint_max(y, x, axis=None, keepdim=False):
    if axis is None:
        max_val = x.max()
        mask = x == max_val
        num_max = mask.sum()
    else:
        max_val = x.max(axis=axis, keepdim=True)
        mask = x == max_val
        num_max = mask.sum(axis=axis, keepdim=True)
    d[x] = tangent.unreduce(tangent.tg_seed(d[y], x), tangent.tg_shape(x), axis, keepdim) * mask / num_max


@adjoint(Tensor.min)
def adjoint_min(y, x, axis=None, keepdim=False):
    if axis is None:
        min_val = x.min()
        mask = x == min_val
        num_min = mask.sum()
    else:
        min_val = x.min(axis=axis, keepdim=True)
        mask = x == min_val
        num_min = mask.sum(axis=axis, keepdim=True)
    d[x] = tangent.unreduce(tangent.tg_seed(d[y], x), tangent.tg_shape(x), axis, keepdim) * mask / num_min


@adjoint(Tensor.var)
def adjoint_var(y, x, axis=None, keepdim=False, correction=1):
    n = tangent.tg_size(x, axis)
    mean = x.mean(axis=axis, keepdim=True)
    d[x] = tangent.unreduce(tangent.tg_seed(d[y], x), tangent.tg_shape(x), axis, keepdim) * 2.0 * (x - mean) / (n - correction)


@adjoint(Tensor.std)
def adjoint_std(y, x, axis=None, keepdim=False, correction=1):
    n = tangent.tg_size(x, axis)
    mean = x.mean(axis=axis, keepdim=True)
    d[x] = tangent.unreduce(tangent.tg_seed(d[y], x), tangent.tg_shape(x), axis, keepdim) * (x - mean) / (y * (n - correction))


@adjoint(Tensor.logsumexp)
def adjoint_logsumexp(y, x, axis=None, keepdim=False):
    d[x] = tangent.unreduce(tangent.tg_seed(d[y], x), tangent.tg_shape(x), axis, keepdim) * (x - y).exp()


@adjoint(Tensor.prod)
def adjoint_prod(y, x, axis=None, keepdim=False):
    # d/dx_i prod(x) = prod(x) / x_i; the keepdim product broadcasts back over
    # the reduced axes. Unstable where x contains zeros (matches tinygrad).
    d[x] = tangent.unreduce(tangent.tg_seed(d[y], x), tangent.tg_shape(x), axis, keepdim) * (x.prod(axis=axis, keepdim=True) / x)


@adjoint(Tensor.cumsum)
def adjoint_cumsum(y, x, axis=0):
    # The adjoint of a cumulative sum is a cumulative sum in reverse.
    d[x] = tangent.tg_seed(d[y], x).flip(axis).cumsum(axis).flip(axis)


# --- Linear algebra / NN ---

@adjoint(Tensor.matmul)
def adjoint_matmul(z, x, y):
    dz = tangent.tg_seed(d[z], x)
    if len(x.shape) == 1 and len(y.shape) == 1:
        d[x] = dz * y
        d[y] = dz * x
    elif len(x.shape) == 2 and len(y.shape) == 2:
        d[x] = dz.matmul(y.transpose())
        d[y] = x.transpose().matmul(dz)
    elif len(x.shape) == 2 and len(y.shape) == 1:
        d[x] = dz.unsqueeze(1).matmul(y.unsqueeze(0))
        d[y] = x.transpose().matmul(dz)
    elif len(x.shape) == 1 and len(y.shape) == 2:
        d[x] = dz.matmul(y.transpose())
        d[y] = x.unsqueeze(1).matmul(dz.unsqueeze(0))
    else:
        d[x] = dz.matmul(y.transpose(-2, -1))
        d[y] = x.transpose(-2, -1).matmul(dz)


@adjoint(Tensor.dot)
def adjoint_dot(z, x, y):
    # tinygrad's dot accepts a 2-D x 2-D or a 2-D x 1-D operand.
    dz = tangent.tg_seed(d[z], x)
    if len(y.shape) == 1:
        d[x] = dz.unsqueeze(1).matmul(y.unsqueeze(0))
        d[y] = x.transpose().matmul(dz)
    else:
        d[x] = dz.matmul(y.transpose())
        d[y] = x.transpose().matmul(dz)


@adjoint(Tensor.softmax)
def adjoint_softmax(y, x, axis=-1):
    s = tangent.tg_seed(d[y], x)
    d[x] = y * (s - (s * y).sum(axis=axis, keepdim=True))


@adjoint(Tensor.log_softmax)
def adjoint_log_softmax(y, x, axis=-1):
    s = tangent.tg_seed(d[y], x)
    d[x] = s - y.exp() * s.sum(axis=axis, keepdim=True)


@adjoint(Tensor.layernorm)
def adjoint_layernorm(y, x, axis=-1, eps=1e-05):
    # y = (x - mean) / sqrt(var + eps) over `axis`. Standard LayerNorm backward:
    # dx = rstd * (dy - mean(dy) - y * mean(dy * y)), all reduced over `axis`.
    s = tangent.tg_seed(d[y], x)
    rstd = (x.var(axis=axis, keepdim=True, correction=0) + eps).rsqrt()
    d[x] = rstd * (s - s.mean(axis=axis, keepdim=True) - y * (s * y).mean(axis=axis, keepdim=True))


@adjoint(Tensor.batchnorm)
def adjoint_batchnorm(y, x, weight, bias, mean, invstd, axis=1):
    # y = (x - mean) * invstd * weight + bias, with the rank-reduced params
    # broadcast along `axis`. n is the normalized input. The param gradients
    # reduce over the complement of `axis` (not trailing-aligned, so a plain
    # unbroadcast would pick the wrong axes).
    s = tangent.tg_seed(d[y], x)
    w_b = tangent.tg_broadcast_axis(x, weight, axis)
    m_b = tangent.tg_broadcast_axis(x, mean, axis)
    i_b = tangent.tg_broadcast_axis(x, invstd, axis)
    n = (x - m_b) * i_b
    d[x] = s * i_b * w_b
    d[weight] = tangent.tg_reduce_except(s * n, axis)
    d[bias] = tangent.tg_reduce_except(s, axis)
    d[mean] = tangent.tg_reduce_except(-s * i_b * w_b, axis)
    d[invstd] = tangent.tg_reduce_except(s * (x - m_b) * w_b, axis)


@adjoint(Tensor.conv2d)
def adjoint_conv2d(z, x, weight, bias=None, groups=1, stride=1, dilation=1, padding=0):
    # dx is a transposed conv (any stride/dilation/padding); dw is the
    # correlation of x and dz with stride/dilation swapped (groups=1 only -
    # the helper raises otherwise). dtype= calls are not supported.
    dz = tangent.tg_seed(d[z], x)
    d[x] = tangent.tg_conv2d_grad_input(dz, weight, x, stride, dilation, padding)
    d[weight] = tangent.tg_conv2d_grad_weight(x, dz, weight, stride, dilation, padding, groups)
    if bias is not None:
        d[bias] = tangent.tg_reduce_except(dz, 1)


@adjoint(Tensor.avg_pool2d)
def adjoint_avg_pool2d(y, x, kernel_size=(2, 2), stride=None, dilation=1, padding=0, ceil_mode=False, count_include_pad=True):
    # Supported for the defaults: no ceil_mode, count_include_pad=True (the
    # helper raises otherwise).
    d[x] = tangent.tg_avg_pool2d_grad_input(tangent.tg_seed(d[y], x), x, kernel_size, stride, dilation, padding, ceil_mode, count_include_pad)


@adjoint(Tensor.max_pool2d)
def adjoint_max_pool2d(y, x, kernel_size=(2, 2), stride=None, dilation=1, padding=0, ceil_mode=False, return_indices=False):
    # Window-argmax scatter via unfold + delta-kernel transposed convs; the
    # helper raises NotImplementedError for unsupported configurations
    # (dilation > 1, ceil_mode, return_indices, non-symmetric padding).
    d[x] = tangent.tg_max_pool2d_grad_input(tangent.tg_seed(d[y], x), x, kernel_size, stride, dilation, padding, ceil_mode, return_indices)


# --- Shape manipulation ---

@adjoint(Tensor.reshape)
def adjoint_reshape(y, x, shape):
    d[x] = tangent.tg_seed(d[y], x).reshape(x.shape)


@adjoint(Tensor.transpose)
def adjoint_transpose(y, x, dim0=1, dim1=0):
    d[x] = tangent.tg_seed(d[y], x).transpose(dim0, dim1)


@adjoint(Tensor.permute)
def adjoint_permute(y, x, order):
    d[x] = tangent.tg_seed(d[y], x).permute(tangent.tg_inv_perm(order))


@adjoint(Tensor.squeeze)
def adjoint_squeeze(y, x, dim=None):
    d[x] = tangent.tg_seed(d[y], x).reshape(x.shape)


@adjoint(Tensor.unsqueeze)
def adjoint_unsqueeze(y, x, dim):
    d[x] = tangent.tg_seed(d[y], x).reshape(x.shape)


@adjoint(Tensor.expand)
def adjoint_expand(y, x, shape):
    d[x] = tangent.unbroadcast(tangent.tg_seed(d[y], x), x)


@adjoint(Tensor.flip)
def adjoint_flip(y, x, axis):
    d[x] = tangent.tg_seed(d[y], x).flip(axis)


# ============================================================================
# Forward-mode (tangent) definitions
# ============================================================================

@tangent_(Tensor.add)
def tangent_add(z, x, y):
    d[z] = d[x] + d[y]


@tangent_(Tensor.sub)
def tangent_sub(z, x, y):
    d[z] = d[x] - d[y]


@tangent_(Tensor.mul)
def tangent_mul(z, x, y):
    d[z] = d[x] * y + x * d[y]


@tangent_(Tensor.div)
def tangent_div(z, x, y):
    d[z] = (d[x] * y - x * d[y]) / (y * y)


@tangent_(Tensor.pow)
def tangent_pow(y, x, n):
    d[y] = d[x] * n * x.pow(n - 1)


@tangent_(Tensor.neg)
def tangent_neg(y, x):
    d[y] = -d[x]


@tangent_(Tensor.exp)
def tangent_exp(y, x):
    d[y] = d[x] * y


@tangent_(Tensor.log)
def tangent_log(y, x):
    d[y] = d[x] / x


@tangent_(Tensor.sqrt)
def tangent_sqrt(y, x):
    d[y] = d[x] / (2.0 * y)


@tangent_(Tensor.square)
def tangent_square(y, x):
    d[y] = 2.0 * x * d[x]


@tangent_(Tensor.tanh)
def tangent_tanh(y, x):
    d[y] = d[x] * (1.0 - y * y)


@tangent_(Tensor.sigmoid)
def tangent_sigmoid(y, x):
    d[y] = d[x] * y * (1.0 - y)


@tangent_(Tensor.relu)
def tangent_relu(y, x):
    d[y] = (x > 0).where(d[x], 0.0)


@tangent_(Tensor.sin)
def tangent_sin(y, x):
    d[y] = d[x] * x.cos()


@tangent_(Tensor.cos)
def tangent_cos(y, x):
    d[y] = -d[x] * x.sin()


@tangent_(Tensor.sum)
def tangent_sum(y, x, axis=None, keepdim=False):
    d[y] = d[x].sum(axis=axis, keepdim=keepdim)


@tangent_(Tensor.mean)
def tangent_mean(y, x, axis=None, keepdim=False):
    d[y] = d[x].mean(axis=axis, keepdim=keepdim)


@tangent_(Tensor.matmul)
def tangent_matmul(z, x, y):
    d[z] = d[x].matmul(y) + x.matmul(d[y])


@tangent_(Tensor.reshape)
def tangent_reshape(y, x, shape):
    d[y] = d[x].reshape(shape)


@tangent_(Tensor.transpose)
def tangent_transpose(y, x, dim0=1, dim1=0):
    d[y] = d[x].transpose(dim0, dim1)


print(f"✓ tinygrad extensions loaded successfully")
