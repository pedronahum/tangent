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
"""PyTorch extensions for Tangent automatic differentiation.

This module provides gradient definitions for PyTorch operations, enabling
Tangent to differentiate functions that use torch tensors and the functional
torch API (torch.add, torch.matmul, torch.sum, ...).

Example:
    import torch
    import tangent

    def f(x):
        return torch.sum(torch.matmul(x, x) ** 2)

    df = tangent.grad(f)
    gradient = df(torch.tensor([1.0, 2.0, 3.0]))
"""

from __future__ import absolute_import

from numbers import Number

try:
    import torch
except ImportError:
    # Optional dependency; tangent/__init__.py reports the failure.
    raise

import numpy as np
from tangent import non_differentiable
from tangent import utils
from tangent.elementwise_rules import prefix_vocab
from tangent.elementwise_rules import register_elementwise
from tangent.grads import adjoint
from tangent.tangents import tangent_
from tangent.utils import register_shape_function
from tangent.utils import register_init_grad

from tangent import utils as _utils

TensorType = torch.Tensor


def size(x, axis):
    """Get the size of a tensor along the given axes."""
    if axis is not None and isinstance(axis, int):
        axis = (axis,)
    axis_shape = x.shape if axis is None else tuple(x.shape[a] for a in axis)
    return max(int(np.prod(axis_shape)), 1)


def dtype(arr):
    """Get the dtype of a torch tensor."""
    return arr.dtype


def shape_as_list(arr):
    """Get shape as a list."""
    return list(arr.shape)


def torch_shapes_match(a, b):
    """Check if two values have matching shapes (scalars have shape ())."""
    sa = tuple(a.shape) if hasattr(a, 'shape') else ()
    sb = tuple(b.shape) if hasattr(b, 'shape') else ()
    return sa == sb


def torch_seed(g, like):
    """Coerce a gradient seed to a torch tensor matching `like`'s dtype.

    Gradient seeds arrive as plain Python floats when a gradient function is
    called without an explicit init_grad, and as NumPy arrays after passing
    through NumPy-generic helpers. Torch does not mix with those types in
    arithmetic, so every adjoint coerces its incoming seed first.
    """
    if isinstance(g, torch.Tensor):
        return g
    return torch.as_tensor(g, dtype=like.dtype)


# Register shape functions
register_shape_function(TensorType, shape_as_list)

# Register non-differentiable functions (shape queries, constructors, etc.)
non_differentiable.register_non_differentiable_functions(
    torch.zeros,
    torch.ones,
    torch.empty,
    torch.zeros_like,
    torch.ones_like,
    torch.empty_like,
    torch.full,
    torch.full_like,
    torch.eye,
    torch.arange,
    torch.linspace,
    torch.logspace,
    torch.tensor,
    torch.as_tensor,
    torch.rand,
    torch.randn,
    size,
    shape_as_list,
    dtype,
    torch_seed,
)

# Register gradient initializers
register_init_grad(TensorType, torch.zeros_like)

# Register add_grad and shape checking for torch tensors
try:
    _utils.register_add_grad(TensorType, TensorType, torch.add)
except ValueError:
    pass  # Already registered

for num_type in [float, int, Number]:
    try:
        _utils.register_add_grad(TensorType, num_type, torch.add)
        _utils.register_add_grad(num_type, TensorType, torch.add)
    except ValueError:
        pass

try:
    _utils.register_shape_checker(TensorType, TensorType, torch_shapes_match)
except ValueError:
    pass

for num_type in [float, int, Number]:
    try:
        _utils.register_shape_checker(TensorType, num_type, torch_shapes_match)
        _utils.register_shape_checker(num_type, TensorType, torch_shapes_match)
    except ValueError:
        pass


# Type mixing support: NumPy <-> PyTorch conversion for gradient accumulation
def add_grad_numpy_to_torch(left, right):
    """Add a NumPy array to a torch tensor by converting to torch."""
    return torch.add(torch.as_tensor(left, dtype=right.dtype), right)


def add_grad_torch_to_numpy(left, right):
    """Add a torch tensor to a NumPy array by converting to torch."""
    return torch.add(left, torch.as_tensor(right, dtype=left.dtype))


try:
    _utils.register_add_grad(np.ndarray, TensorType, add_grad_numpy_to_torch)
    _utils.register_add_grad(TensorType, np.ndarray, add_grad_torch_to_numpy)
except ValueError as e:
    if "already mapped" not in str(e):
        raise


# Torch-specific unbroadcast and unreduce functions
def torch_unbroadcast_to(array, shape):
    """Reverse a broadcasting operation for torch tensors."""
    if not isinstance(array, torch.Tensor):
        array = torch.as_tensor(array)
    axis = utils.create_unbroadcast_axis(tuple(shape), tuple(array.shape))
    # Note: torch.sum(dim=()) reduces over ALL dimensions (unlike NumPy's
    # identity), so skip the reduction when there is nothing to reduce.
    if axis:
        array = torch.sum(array, dim=axis)
    return torch.reshape(array, tuple(shape))


def torch_unbroadcast(array, like):
    """Unbroadcast a torch tensor to match the shape of `like`."""
    if not isinstance(array, torch.Tensor):
        array = torch.as_tensor(array, dtype=getattr(like, 'dtype', None))
    # `like` may be a plain Python scalar (a float seed), which has shape ().
    like_shape = tuple(like.shape) if hasattr(like, 'shape') else ()
    return torch_unbroadcast_to(array, like_shape)


def torch_unreduce(array, shape, axis, keepdims):
    """Reverse summing over a dimension for torch tensors."""
    if not isinstance(array, torch.Tensor):
        array = torch.as_tensor(array)
    if axis is not None and not keepdims:
        if isinstance(axis, int):
            axis = (axis,)
        for ax in sorted(axis):
            array = torch.unsqueeze(array, ax)
    return torch.broadcast_to(array, tuple(shape))


try:
    _utils.unbroadcasters[TensorType] = torch_unbroadcast
except (AttributeError, KeyError):
    pass

try:
    _utils.unreducers[TensorType] = torch_unreduce
except (AttributeError, KeyError):
    pass


# `@` (matmul) operator gradients for torch tensors. Keras needs no separate
# registration: keras.ops tensors are the active backend's tensors at runtime
# (torch.Tensor here, tf/jax tensors under those backends), so operator
# dispatch on the runtime type reaches the right backend's rule.
def torch_matmul_grad_x(dz, x, y):
    """d[x] for z = x @ y, covering the vector/matrix rank promotions."""
    dz = torch_seed(dz, x)
    if x.ndim == 1 and y.ndim == 1:
        return dz * y
    if x.ndim == 2 and y.ndim == 1:
        return torch.outer(dz, y)
    return torch.matmul(dz, torch.transpose(y, -2, -1))


def torch_matmul_grad_y(dz, x, y):
    """d[y] for z = x @ y, covering the vector/matrix rank promotions."""
    dz = torch_seed(dz, y)
    if x.ndim == 1 and y.ndim == 1:
        return dz * x
    if x.ndim == 1 and y.ndim == 2:
        return torch.outer(x, dz)
    return torch.matmul(torch.transpose(x, -2, -1), dz)


_utils.register_matmul_grad(TensorType, torch_matmul_grad_x, torch_matmul_grad_y)


# ============================================================================
# Reverse-mode (adjoint) gradient definitions
# ============================================================================


# Basic arithmetic operations
@adjoint(torch.add)
def adjoint_add(z, x, y):
    """Adjoint for torch.add."""
    d[x] = tangent.unbroadcast(tangent.torch_seed(d[z], x), x)
    d[y] = tangent.unbroadcast(tangent.torch_seed(d[z], y), y)


@adjoint(torch.sub)
def adjoint_sub(z, x, y):
    """Adjoint for torch.sub."""
    dz = tangent.torch_seed(d[z], x)
    d[x] = tangent.unbroadcast(dz, x)
    d[y] = tangent.unbroadcast(-dz, y)


@adjoint(torch.subtract)
def adjoint_subtract(z, x, y):
    """Adjoint for torch.subtract (alias of sub)."""
    dz = tangent.torch_seed(d[z], x)
    d[x] = tangent.unbroadcast(dz, x)
    d[y] = tangent.unbroadcast(-dz, y)


@adjoint(torch.mul)
def adjoint_mul(z, x, y):
    """Adjoint for torch.mul."""
    dz = tangent.torch_seed(d[z], x)
    d[x] = tangent.unbroadcast(dz * y, x)
    d[y] = tangent.unbroadcast(dz * x, y)


@adjoint(torch.multiply)
def adjoint_multiply(z, x, y):
    """Adjoint for torch.multiply (alias of mul)."""
    dz = tangent.torch_seed(d[z], x)
    d[x] = tangent.unbroadcast(dz * y, x)
    d[y] = tangent.unbroadcast(dz * x, y)


@adjoint(torch.div)
def adjoint_div(z, x, y):
    """Adjoint for torch.div."""
    dz = tangent.torch_seed(d[z], x)
    d[x] = tangent.unbroadcast(dz / y, x)
    d[y] = tangent.unbroadcast(-dz * x / (y * y), y)


@adjoint(torch.divide)
def adjoint_divide(z, x, y):
    """Adjoint for torch.divide (alias of div)."""
    dz = tangent.torch_seed(d[z], x)
    d[x] = tangent.unbroadcast(dz / y, x)
    d[y] = tangent.unbroadcast(-dz * x / (y * y), y)


@adjoint(torch.pow)
def adjoint_pow(y, x, n):
    """Adjoint for torch.pow."""
    d[x] = tangent.unbroadcast(tangent.torch_seed(d[y], x) * n * torch.pow(x, n - 1), x)


# Unary elementwise ops: generated (adjoint AND tangent per op) from the
# backend-neutral rule table. Long-name aliases (torch.negative, torch.asin,
# ...) are distinct function objects, so they are registered alongside the
# short spellings.
register_elementwise(
    'torch',
    ops={
        'exp': torch.exp,
        'expm1': torch.expm1,
        'exp2': torch.exp2,
        'log': torch.log,
        'log2': torch.log2,
        'log10': torch.log10,
        'log1p': torch.log1p,
        'sqrt': torch.sqrt,
        'rsqrt': torch.rsqrt,
        'square': torch.square,
        'reciprocal': torch.reciprocal,
        'negative': (torch.neg, torch.negative),
        'abs': torch.abs,
        'sin': torch.sin,
        'cos': torch.cos,
        'tan': torch.tan,
        'arcsin': (torch.arcsin, torch.asin),
        'arccos': (torch.arccos, torch.acos),
        'arctan': (torch.arctan, torch.atan),
        'sinh': torch.sinh,
        'cosh': torch.cosh,
        'tanh': torch.tanh,
        'sigmoid': torch.sigmoid,
        'relu': torch.relu,
        'floor': torch.floor,
        'ceil': torch.ceil,
        'round': torch.round,
        'sign': torch.sign,
    },
    vocab=prefix_vocab('torch', mask_pos='(({arg}) > 0)'),
    seed='tangent.torch_seed({g}, x)',
)


# Reduction operations
@adjoint(torch.sum)
def adjoint_sum(y, x, axis=None, keepdims=False):
    """Adjoint for torch.sum."""
    d[x] = tangent.unreduce(tangent.torch_seed(d[y], x), tangent.shape_as_list(x), axis, keepdims)


@adjoint(torch.mean)
def adjoint_mean(y, x, axis=None, keepdims=False):
    """Adjoint for torch.mean."""
    n = tangent.size(x, axis)
    d[x] = (
        tangent.unreduce(tangent.torch_seed(d[y], x), tangent.shape_as_list(x), axis, keepdims) / n
    )


@adjoint(torch.max)
def adjoint_max(y, x, axis=None, keepdims=False):
    """Adjoint for torch.max: gradient flows to the maximum element(s)."""
    if axis is None:
        # Global reduction: torch.max(x, dim=None, ...) is not a valid call.
        max_val = torch.max(x)
        mask = (x == max_val).to(x.dtype)
        num_max = torch.sum(mask)
    else:
        max_val = torch.max(x, dim=axis, keepdim=True).values
        mask = (x == max_val).to(x.dtype)
        num_max = torch.sum(mask, dim=axis, keepdim=True)
    d[x] = (
        tangent.unreduce(tangent.torch_seed(d[y], x), tangent.shape_as_list(x), axis, keepdims)
        * mask
        / num_max
    )


@adjoint(torch.min)
def adjoint_min(y, x, axis=None, keepdims=False):
    """Adjoint for torch.min: gradient flows to the minimum element(s)."""
    if axis is None:
        # Global reduction: torch.min(x, dim=None, ...) is not a valid call.
        min_val = torch.min(x)
        mask = (x == min_val).to(x.dtype)
        num_min = torch.sum(mask)
    else:
        min_val = torch.min(x, dim=axis, keepdim=True).values
        mask = (x == min_val).to(x.dtype)
        num_min = torch.sum(mask, dim=axis, keepdim=True)
    d[x] = (
        tangent.unreduce(tangent.torch_seed(d[y], x), tangent.shape_as_list(x), axis, keepdims)
        * mask
        / num_min
    )


# Linear algebra
@adjoint(torch.matmul)
def adjoint_matmul(z, x, y):
    """Adjoint for torch.matmul covering vector and matrix cases."""
    dz = tangent.torch_seed(d[z], x)
    if x.ndim == 1 and y.ndim == 1:
        d[x] = dz * y
        d[y] = dz * x
    elif x.ndim == 2 and y.ndim == 2:
        d[x] = torch.matmul(dz, torch.transpose(y, 0, 1))
        d[y] = torch.matmul(torch.transpose(x, 0, 1), dz)
    elif x.ndim == 2 and y.ndim == 1:
        d[x] = torch.outer(dz, y)
        d[y] = torch.matmul(torch.transpose(x, 0, 1), dz)
    elif x.ndim == 1 and y.ndim == 2:
        d[x] = torch.matmul(dz, torch.transpose(y, 0, 1))
        d[y] = torch.outer(x, dz)
    else:
        d[x] = torch.matmul(dz, torch.transpose(y, -2, -1))
        d[y] = torch.matmul(torch.transpose(x, -2, -1), dz)


@adjoint(torch.mm)
def adjoint_mm(z, x, y):
    """Adjoint for torch.mm (2-D matrix multiplication)."""
    dz = tangent.torch_seed(d[z], x)
    d[x] = torch.matmul(dz, torch.transpose(y, 0, 1))
    d[y] = torch.matmul(torch.transpose(x, 0, 1), dz)


@adjoint(torch.mv)
def adjoint_mv(z, x, y):
    """Adjoint for torch.mv (matrix-vector product)."""
    dz = tangent.torch_seed(d[z], x)
    d[x] = torch.outer(dz, y)
    d[y] = torch.matmul(torch.transpose(x, 0, 1), dz)


@adjoint(torch.dot)
def adjoint_dot(z, x, y):
    """Adjoint for torch.dot (1-D inner product)."""
    dz = tangent.torch_seed(d[z], x)
    d[x] = dz * y
    d[y] = dz * x


# Shape manipulation
@adjoint(torch.reshape)
def adjoint_reshape(y, x, shape):
    """Adjoint for torch.reshape."""
    d[x] = torch.reshape(tangent.torch_seed(d[y], x), tuple(x.shape))


@adjoint(torch.transpose)
def adjoint_transpose(y, x, dim0=-2, dim1=-1):
    """Adjoint for torch.transpose (self-inverse)."""
    d[x] = torch.transpose(tangent.torch_seed(d[y], x), dim0, dim1)


@adjoint(torch.permute)
def adjoint_permute(y, x, dims):
    """Adjoint for torch.permute."""
    inv_dims = [0] * len(dims)
    for i, ax in enumerate(dims):
        inv_dims[ax] = i
    d[x] = torch.permute(tangent.torch_seed(d[y], x), inv_dims)


@adjoint(torch.squeeze)
def adjoint_squeeze(y, x, axis=None):
    """Adjoint for torch.squeeze."""
    d[x] = torch.reshape(tangent.torch_seed(d[y], x), tuple(x.shape))


@adjoint(torch.unsqueeze)
def adjoint_unsqueeze(y, x, axis):
    """Adjoint for torch.unsqueeze."""
    d[x] = torch.reshape(tangent.torch_seed(d[y], x), tuple(x.shape))


# Element-wise selection
@adjoint(torch.maximum)
def adjoint_maximum(z, x, y):
    """Adjoint for torch.maximum."""
    dz = tangent.torch_seed(d[z], x)
    d[x] = tangent.unbroadcast(dz * (x >= y).to(x.dtype), x)
    d[y] = tangent.unbroadcast(dz * (y > x).to(y.dtype), y)


@adjoint(torch.minimum)
def adjoint_minimum(z, x, y):
    """Adjoint for torch.minimum."""
    dz = tangent.torch_seed(d[z], x)
    d[x] = tangent.unbroadcast(dz * (x <= y).to(x.dtype), x)
    d[y] = tangent.unbroadcast(dz * (y < x).to(y.dtype), y)


@adjoint(torch.clamp)
def adjoint_clamp(y, x, min=None, max=None):
    """Adjoint for torch.clamp: gradient flows only where x is not clipped."""
    mask = torch.ones_like(x)
    if min is not None:
        mask = torch.where(x < min, torch.zeros_like(x), mask)
    if max is not None:
        mask = torch.where(x > max, torch.zeros_like(x), mask)
    d[x] = tangent.torch_seed(d[y], x) * mask


@adjoint(torch.where)
def adjoint_where(z, condition, x, y):
    """Adjoint for torch.where."""
    dz = tangent.torch_seed(d[z], x)
    d[x] = torch.where(condition, dz, torch.zeros_like(dz))
    d[y] = torch.where(condition, torch.zeros_like(dz), dz)


# Softmax family
@adjoint(torch.softmax)
def adjoint_softmax(y, x, dim=-1):
    """Adjoint for torch.softmax: d[x] = y * (dz - sum(dz * y, dim))."""
    s = tangent.torch_seed(d[y], x)
    d[x] = y * (s - torch.sum(s * y, dim=dim, keepdim=True))


@adjoint(torch.log_softmax)
def adjoint_log_softmax(y, x, dim=-1):
    """Adjoint for torch.log_softmax: d[x] = dz - exp(y) * sum(dz, dim)."""
    s = tangent.torch_seed(d[y], x)
    d[x] = s - torch.exp(y) * torch.sum(s, dim=dim, keepdim=True)


# The torch.nn.functional spellings are distinct function objects; register
# the same templates for them.
adjoint(torch.nn.functional.softmax)(adjoint_softmax)
adjoint(torch.nn.functional.log_softmax)(adjoint_log_softmax)


# Concatenation and stacking.
#
# torch.cat / torch.stack take a *list* of tensors, which Tangent cannot
# distribute gradients into. concat_desugar rewrites list-literal calls into
# the varargs helpers below (mirroring the JAX concat_seq/stack_seq
# machinery), whose varargs adjoints split the gradient back per input.


def torch_cat_seq(axis, *tensors):
    """Runtime helper: concatenate a varargs sequence of tensors."""
    return torch.cat(list(tensors), dim=axis)


def torch_stack_seq(axis, *tensors):
    """Runtime helper: stack a varargs sequence of tensors."""
    return torch.stack(list(tensors), dim=axis)


def torch_cat_grads(dz, tensors, axis):
    """Split a concatenated gradient back into per-input gradients."""
    dz = torch_seed(dz, tensors[0])
    sizes = [int(t.shape[axis]) for t in tensors]
    return tuple(torch.split(dz, sizes, dim=axis))


def torch_stack_grads(dz, tensors, axis):
    """Unstack a stacked gradient along the stacking axis."""
    dz = torch_seed(dz, tensors[0])
    return tuple(torch.unbind(dz, dim=axis))


non_differentiable.register_non_differentiable_functions(torch_cat_grads, torch_stack_grads)


@adjoint(torch_cat_seq)
def adjoint_torch_cat_seq(z, axis, *tensors):
    """Adjoint for torch_cat_seq: split the gradient back per input."""
    d[tensors] = tangent.torch_cat_grads(d[z], tensors, axis)


@adjoint(torch_stack_seq)
def adjoint_torch_stack_seq(z, axis, *tensors):
    """Adjoint for torch_stack_seq: unstack the gradient."""
    d[tensors] = tangent.torch_stack_grads(d[z], tensors, axis)


# The list-argument forms are only reachable when the desugar pass could not
# rewrite the call (e.g. the list is built dynamically). Raise a clear error
# rather than generating broken code.
@adjoint(torch.cat)
def adjoint_cat(dz, tensors, dim=0):
    """Not differentiable: pass a list literal so it can be desugared."""
    raise NotImplementedError(
        'tangent can only differentiate torch.cat/torch.stack when the list '
        'of tensors is a literal. Bind the list to a variable assigned once '
        'from a literal, or pass a list literal directly.'
    )


@adjoint(torch.stack)
def adjoint_stack(dz, tensors, dim=0):
    """Not differentiable: pass a list literal so it can be desugared."""
    raise NotImplementedError(
        'tangent can only differentiate torch.cat/torch.stack when the list '
        'of tensors is a literal. Bind the list to a variable assigned once '
        'from a literal, or pass a list literal directly.'
    )


# torch.concat / torch.concatenate are distinct alias objects for torch.cat.
adjoint(torch.concat)(adjoint_cat)
adjoint(torch.concatenate)(adjoint_cat)


#
# Forward mode (tangent) definitions
#


@tangent_(torch.add)
def tangent_add(z, x, y):
    """Forward mode for torch.add."""
    d[z] = d[x] + d[y]


@tangent_(torch.sub)
def tangent_sub(z, x, y):
    """Forward mode for torch.sub."""
    d[z] = d[x] - d[y]


@tangent_(torch.mul)
def tangent_mul(z, x, y):
    """Forward mode for torch.mul."""
    d[z] = d[x] * y + x * d[y]


@tangent_(torch.div)
def tangent_div(z, x, y):
    """Forward mode for torch.div."""
    d[z] = (d[x] * y - x * d[y]) / (y * y)


@tangent_(torch.pow)
def tangent_pow(y, x, n):
    """Forward mode for torch.pow."""
    d[y] = d[x] * n * torch.pow(x, n - 1)


# Unary elementwise tangents (exp, log, trig, ...) are generated alongside
# their adjoints by the register_elementwise call above.


@tangent_(torch.sum)
def tangent_sum(y, x, axis=None, keepdims=False):
    """Forward mode for torch.sum."""
    d[y] = torch.sum(d[x], dim=axis, keepdim=keepdims)


@tangent_(torch.mean)
def tangent_mean(y, x, axis=None, keepdims=False):
    """Forward mode for torch.mean."""
    d[y] = torch.mean(d[x], dim=axis, keepdim=keepdims)


@tangent_(torch.matmul)
def tangent_matmul(z, x, y):
    """Forward mode for torch.matmul."""
    d[z] = torch.matmul(d[x], y) + torch.matmul(x, d[y])


@tangent_(torch.mm)
def tangent_mm(z, x, y):
    """Forward mode for torch.mm."""
    d[z] = torch.matmul(d[x], y) + torch.matmul(x, d[y])


@tangent_(torch.dot)
def tangent_dot(z, x, y):
    """Forward mode for torch.dot."""
    d[z] = torch.dot(d[x], y) + torch.dot(x, d[y])


@tangent_(torch.reshape)
def tangent_reshape(y, x, shape):
    """Forward mode for torch.reshape."""
    d[y] = torch.reshape(d[x], tuple(shape))


@tangent_(torch.transpose)
def tangent_transpose(y, x, dim0=-2, dim1=-1):
    """Forward mode for torch.transpose."""
    d[y] = torch.transpose(d[x], dim0, dim1)


@tangent_(torch.squeeze)
def tangent_squeeze(y, x, axis=None):
    """Forward mode for torch.squeeze."""
    d[y] = torch.squeeze(d[x], dim=axis)


@tangent_(torch.unsqueeze)
def tangent_unsqueeze(y, x, axis):
    """Forward mode for torch.unsqueeze."""
    d[y] = torch.unsqueeze(d[x], dim=axis)


@tangent_(torch.softmax)
def tangent_softmax(y, x, dim=-1):
    """Forward mode for torch.softmax."""
    d[y] = y * (d[x] - torch.sum(d[x] * y, dim=dim, keepdim=True))


@tangent_(torch.log_softmax)
def tangent_log_softmax(y, x, dim=-1):
    """Forward mode for torch.log_softmax: dy = dx - sum(softmax(x) * dx)."""
    d[y] = d[x] - torch.sum(torch.exp(y) * d[x], dim=dim, keepdim=True)


tangent_(torch.nn.functional.softmax)(tangent_softmax)
tangent_(torch.nn.functional.log_softmax)(tangent_log_softmax)


@tangent_(torch_cat_seq)
def tangent_torch_cat_seq(z, axis, *tensors):
    """Forward mode for torch_cat_seq."""
    d[z] = tangent.torch_cat_seq(axis, *d[tensors])


@tangent_(torch_stack_seq)
def tangent_torch_stack_seq(z, axis, *tensors):
    """Forward mode for torch_stack_seq."""
    d[z] = tangent.torch_stack_seq(axis, *d[tensors])


import logging as _logging

_logging.getLogger('tangent').debug(
    'PyTorch extensions loaded successfully (torch %s)', torch.__version__
)
