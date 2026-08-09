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
"""Keras extensions for Tangent automatic differentiation.

This module provides gradient definitions for the backend-agnostic
``keras.ops`` API, enabling Tangent to differentiate functions that use
Keras 3 operations regardless of the active backend (TensorFlow, JAX or
PyTorch).

The adjoint templates use plain Python operators plus ``keras.ops`` calls,
so the generated gradient code dispatches to whichever backend Keras is
configured with. Array-type plumbing (init_grad, add_grad, unbroadcast,
unreduce, shape checkers) is inherited from the backend-specific
extensions (tf_extensions / jax_extensions / torch_extensions), which
register the concrete tensor types.

Example:
    import keras.ops as kops
    import tangent

    def f(x):
        return kops.sum(kops.matmul(x, x) ** 2)

    df = tangent.grad(f)
"""
from __future__ import absolute_import

import warnings
from numbers import Number

try:
    import keras
    import keras.ops as kops
except ImportError as e:
    warnings.warn(f"Keras not available: {e}. Install with: pip install keras")
    raise

import numpy as np
from tangent import non_differentiable
from tangent import utils
from tangent.grads import adjoint
from tangent.tangents import tangent_

from tangent import utils as _utils


def size(x, axis):
    """Get the size of a tensor along the given axes (int or tuple axis)."""
    if axis is not None and isinstance(axis, int):
        axis = (axis,)
    axis_shape = x.shape if axis is None else tuple(x.shape[a] for a in axis)
    return max(int(np.prod(axis_shape)), 1)


def keras_seed(g, like):
    """Coerce a gradient seed to a backend tensor matching `like`'s dtype.

    Gradient seeds arrive as plain Python floats when a gradient function is
    called without an explicit init_grad; convert them with the active Keras
    backend so the generated code stays backend-consistent.
    """
    if kops.is_tensor(g):
        return g
    return kops.convert_to_tensor(
        g, dtype=keras.backend.standardize_dtype(like.dtype))


def keras_max_mask(x, axis=None):
    """Normalized mask selecting the maximal element(s) along an axis.

    Gradient of max: the seed flows to the argmax positions; when several
    elements tie for the maximum the seed is split evenly between them.
    """
    max_val = kops.max(x, axis=axis, keepdims=True)
    mask = kops.cast(x == max_val, keras.backend.standardize_dtype(x.dtype))
    num_max = kops.sum(mask, axis=axis, keepdims=True)
    return mask / num_max


def keras_min_mask(x, axis=None):
    """Normalized mask selecting the minimal element(s) along an axis."""
    min_val = kops.min(x, axis=axis, keepdims=True)
    mask = kops.cast(x == min_val, keras.backend.standardize_dtype(x.dtype))
    num_min = kops.sum(mask, axis=axis, keepdims=True)
    return mask / num_min


# Shape queries and constructors are not differentiable.
non_differentiable.register_non_differentiable_functions(
    kops.zeros, kops.ones, kops.zeros_like, kops.ones_like,
    kops.full, kops.full_like, kops.eye, kops.arange,
    keras_seed, keras_max_mask, keras_min_mask
)


# ============================================================================
# Reverse-mode (adjoint) gradient definitions
# ============================================================================

# Basic arithmetic
@adjoint(kops.add)
def adjoint_add(z, x1, x2):
    """Adjoint for keras.ops.add."""
    d[x1] = tangent.unbroadcast(tangent.keras_seed(d[z], x1), x1)
    d[x2] = tangent.unbroadcast(tangent.keras_seed(d[z], x2), x2)


@adjoint(kops.subtract)
def adjoint_subtract(z, x1, x2):
    """Adjoint for keras.ops.subtract."""
    dz = tangent.keras_seed(d[z], x1)
    d[x1] = tangent.unbroadcast(dz, x1)
    d[x2] = tangent.unbroadcast(-dz, x2)


@adjoint(kops.multiply)
def adjoint_multiply(z, x1, x2):
    """Adjoint for keras.ops.multiply."""
    dz = tangent.keras_seed(d[z], x1)
    d[x1] = tangent.unbroadcast(dz * x2, x1)
    d[x2] = tangent.unbroadcast(dz * x1, x2)


@adjoint(kops.divide)
def adjoint_divide(z, x1, x2):
    """Adjoint for keras.ops.divide."""
    dz = tangent.keras_seed(d[z], x1)
    d[x1] = tangent.unbroadcast(dz / x2, x1)
    d[x2] = tangent.unbroadcast(-dz * x1 / (x2 * x2), x2)


@adjoint(kops.power)
def adjoint_power(z, x1, x2):
    """Adjoint for keras.ops.power (gradient wrt the base only)."""
    d[x1] = tangent.unbroadcast(
        tangent.keras_seed(d[z], x1) * x2 * kops.power(x1, x2 - 1), x1)


@adjoint(kops.negative)
def adjoint_negative(y, x):
    """Adjoint for keras.ops.negative."""
    d[x] = -tangent.keras_seed(d[y], x)


# Exponential and logarithmic
@adjoint(kops.exp)
def adjoint_exp(y, x):
    """Adjoint for keras.ops.exp."""
    d[x] = tangent.keras_seed(d[y], x) * y


@adjoint(kops.log)
def adjoint_log(y, x):
    """Adjoint for keras.ops.log."""
    d[x] = tangent.keras_seed(d[y], x) / x


@adjoint(kops.sqrt)
def adjoint_sqrt(y, x):
    """Adjoint for keras.ops.sqrt."""
    d[x] = tangent.keras_seed(d[y], x) / (2.0 * y)


@adjoint(kops.square)
def adjoint_square(y, x):
    """Adjoint for keras.ops.square."""
    d[x] = tangent.keras_seed(d[y], x) * (2.0 * x)


@adjoint(kops.abs)
def adjoint_abs(y, x):
    """Adjoint for keras.ops.abs."""
    d[x] = tangent.keras_seed(d[y], x) * kops.sign(x)


# Trigonometric
@adjoint(kops.sin)
def adjoint_sin(y, x):
    """Adjoint for keras.ops.sin."""
    d[x] = tangent.keras_seed(d[y], x) * kops.cos(x)


@adjoint(kops.cos)
def adjoint_cos(y, x):
    """Adjoint for keras.ops.cos."""
    d[x] = -tangent.keras_seed(d[y], x) * kops.sin(x)


@adjoint(kops.tan)
def adjoint_tan(y, x):
    """Adjoint for keras.ops.tan."""
    cx = kops.cos(x)
    d[x] = tangent.keras_seed(d[y], x) / (cx * cx)


@adjoint(kops.arcsin)
def adjoint_arcsin(y, x):
    """Adjoint for keras.ops.arcsin."""
    d[x] = tangent.keras_seed(d[y], x) / kops.sqrt(1.0 - x * x)


@adjoint(kops.arccos)
def adjoint_arccos(y, x):
    """Adjoint for keras.ops.arccos."""
    d[x] = -tangent.keras_seed(d[y], x) / kops.sqrt(1.0 - x * x)


@adjoint(kops.arctan)
def adjoint_arctan(y, x):
    """Adjoint for keras.ops.arctan."""
    d[x] = tangent.keras_seed(d[y], x) / (1.0 + x * x)


# Hyperbolic
@adjoint(kops.sinh)
def adjoint_sinh(y, x):
    """Adjoint for keras.ops.sinh."""
    d[x] = tangent.keras_seed(d[y], x) * kops.cosh(x)


@adjoint(kops.cosh)
def adjoint_cosh(y, x):
    """Adjoint for keras.ops.cosh."""
    d[x] = tangent.keras_seed(d[y], x) * kops.sinh(x)


@adjoint(kops.tanh)
def adjoint_tanh(y, x):
    """Adjoint for keras.ops.tanh."""
    tx = kops.tanh(x)
    d[x] = tangent.keras_seed(d[y], x) * (1.0 - tx * tx)


# Activations
@adjoint(kops.relu)
def adjoint_relu(y, x):
    """Adjoint for keras.ops.relu."""
    mask = kops.cast(x > 0, keras.backend.standardize_dtype(x.dtype))
    d[x] = tangent.keras_seed(d[y], x) * mask


@adjoint(kops.sigmoid)
def adjoint_sigmoid(y, x):
    """Adjoint for keras.ops.sigmoid."""
    sig = kops.sigmoid(x)
    d[x] = tangent.keras_seed(d[y], x) * sig * (1.0 - sig)


# Reductions
@adjoint(kops.sum)
def adjoint_sum(y, x, axis=None, keepdims=False):
    """Adjoint for keras.ops.sum."""
    d[x] = tangent.unreduce(tangent.keras_seed(d[y], x),
                            tangent.shape_as_list(x), axis, keepdims)


@adjoint(kops.mean)
def adjoint_mean(y, x, axis=None, keepdims=False):
    """Adjoint for keras.ops.mean."""
    n = tangent.size(x, axis)
    d[x] = tangent.unreduce(tangent.keras_seed(d[y], x),
                            tangent.shape_as_list(x), axis, keepdims) / n


@adjoint(kops.max)
def adjoint_max(y, x, axis=None, keepdims=False, initial=None):
    """Adjoint for keras.ops.max."""
    d[x] = tangent.unreduce(tangent.keras_seed(d[y], x),
                            tangent.shape_as_list(x), axis,
                            keepdims) * tangent.keras_max_mask(x, axis)


@adjoint(kops.min)
def adjoint_min(y, x, axis=None, keepdims=False, initial=None):
    """Adjoint for keras.ops.min."""
    d[x] = tangent.unreduce(tangent.keras_seed(d[y], x),
                            tangent.shape_as_list(x), axis,
                            keepdims) * tangent.keras_min_mask(x, axis)


@adjoint(kops.prod)
def adjoint_prod(y, x, axis=None, keepdims=False):
    """Adjoint for keras.ops.prod: dL/dx_i = dL/dy * prod(x) / x_i."""
    d[x] = tangent.unreduce(tangent.keras_seed(d[y], x) * y,
                            tangent.shape_as_list(x), axis, keepdims) / x


# Linear algebra
@adjoint(kops.matmul)
def adjoint_matmul(z, x1, x2):
    """Adjoint for keras.ops.matmul covering vector and matrix cases."""
    dz = tangent.keras_seed(d[z], x1)
    if len(x1.shape) == 1 and len(x2.shape) == 1:
        d[x1] = dz * x2
        d[x2] = dz * x1
    elif len(x1.shape) == 2 and len(x2.shape) == 2:
        d[x1] = kops.matmul(dz, kops.transpose(x2))
        d[x2] = kops.matmul(kops.transpose(x1), dz)
    elif len(x1.shape) == 2 and len(x2.shape) == 1:
        d[x1] = kops.outer(dz, x2)
        d[x2] = kops.matmul(kops.transpose(x1), dz)
    elif len(x1.shape) == 1 and len(x2.shape) == 2:
        d[x1] = kops.matmul(dz, kops.transpose(x2))
        d[x2] = kops.outer(x1, dz)
    else:
        d[x1] = kops.matmul(dz, kops.transpose(x2, axes=(-2, -1)))
        d[x2] = kops.matmul(kops.transpose(x1, axes=(-2, -1)), dz)


# Shape manipulation
@adjoint(kops.reshape)
def adjoint_reshape(y, x, newshape):
    """Adjoint for keras.ops.reshape."""
    d[x] = kops.reshape(tangent.keras_seed(d[y], x), tuple(x.shape))


@adjoint(kops.transpose)
def adjoint_transpose(y, x, axes=None):
    """Adjoint for keras.ops.transpose."""
    if axes is None:
        d[x] = kops.transpose(tangent.keras_seed(d[y], x))
    else:
        inv_axes = [0] * len(axes)
        for i, ax in enumerate(axes):
            inv_axes[ax] = i
        d[x] = kops.transpose(tangent.keras_seed(d[y], x), axes=inv_axes)


@adjoint(kops.squeeze)
def adjoint_squeeze(y, x, axis=None):
    """Adjoint for keras.ops.squeeze."""
    d[x] = kops.reshape(tangent.keras_seed(d[y], x), tuple(x.shape))


@adjoint(kops.expand_dims)
def adjoint_expand_dims(y, x, axis):
    """Adjoint for keras.ops.expand_dims."""
    d[x] = kops.reshape(tangent.keras_seed(d[y], x), tuple(x.shape))


# Selection
@adjoint(kops.maximum)
def adjoint_maximum(z, x1, x2):
    """Adjoint for keras.ops.maximum."""
    dz = tangent.keras_seed(d[z], x1)
    m1 = kops.cast(x1 >= x2, keras.backend.standardize_dtype(x1.dtype))
    m2 = kops.cast(x2 > x1, keras.backend.standardize_dtype(x2.dtype))
    d[x1] = tangent.unbroadcast(dz * m1, x1)
    d[x2] = tangent.unbroadcast(dz * m2, x2)


@adjoint(kops.minimum)
def adjoint_minimum(z, x1, x2):
    """Adjoint for keras.ops.minimum."""
    dz = tangent.keras_seed(d[z], x1)
    m1 = kops.cast(x1 <= x2, keras.backend.standardize_dtype(x1.dtype))
    m2 = kops.cast(x2 < x1, keras.backend.standardize_dtype(x2.dtype))
    d[x1] = tangent.unbroadcast(dz * m1, x1)
    d[x2] = tangent.unbroadcast(dz * m2, x2)


@adjoint(kops.clip)
def adjoint_clip(y, x, x_min, x_max):
    """Adjoint for keras.ops.clip."""
    inside = kops.cast(
        kops.logical_and(x >= x_min, x <= x_max),
        keras.backend.standardize_dtype(x.dtype))
    d[x] = tangent.keras_seed(d[y], x) * inside


@adjoint(kops.where)
def adjoint_where(z, condition, x1, x2):
    """Adjoint for keras.ops.where."""
    dz = tangent.keras_seed(d[z], x1)
    d[x1] = kops.where(condition, dz, kops.zeros_like(dz))
    d[x2] = kops.where(condition, kops.zeros_like(dz), dz)


#
# Forward mode (tangent) definitions
#

@tangent_(kops.add)
def tangent_add(z, x1, x2):
    """Forward mode for keras.ops.add."""
    d[z] = d[x1] + d[x2]


@tangent_(kops.subtract)
def tangent_subtract(z, x1, x2):
    """Forward mode for keras.ops.subtract."""
    d[z] = d[x1] - d[x2]


@tangent_(kops.multiply)
def tangent_multiply(z, x1, x2):
    """Forward mode for keras.ops.multiply."""
    d[z] = d[x1] * x2 + x1 * d[x2]


@tangent_(kops.divide)
def tangent_divide(z, x1, x2):
    """Forward mode for keras.ops.divide."""
    d[z] = (d[x1] * x2 - x1 * d[x2]) / (x2 * x2)


@tangent_(kops.negative)
def tangent_negative(y, x):
    """Forward mode for keras.ops.negative."""
    d[y] = -d[x]


@tangent_(kops.exp)
def tangent_exp(y, x):
    """Forward mode for keras.ops.exp."""
    d[y] = d[x] * y


@tangent_(kops.log)
def tangent_log(y, x):
    """Forward mode for keras.ops.log."""
    d[y] = d[x] / x


@tangent_(kops.sqrt)
def tangent_sqrt(y, x):
    """Forward mode for keras.ops.sqrt."""
    d[y] = d[x] / (2.0 * y)


@tangent_(kops.square)
def tangent_square(y, x):
    """Forward mode for keras.ops.square."""
    d[y] = 2.0 * x * d[x]


@tangent_(kops.sin)
def tangent_sin(y, x):
    """Forward mode for keras.ops.sin."""
    d[y] = d[x] * kops.cos(x)


@tangent_(kops.cos)
def tangent_cos(y, x):
    """Forward mode for keras.ops.cos."""
    d[y] = -d[x] * kops.sin(x)


@tangent_(kops.tanh)
def tangent_tanh(y, x):
    """Forward mode for keras.ops.tanh."""
    tx = kops.tanh(x)
    d[y] = d[x] * (1.0 - tx * tx)


@tangent_(kops.sum)
def tangent_sum(y, x, axis=None, keepdims=False):
    """Forward mode for keras.ops.sum."""
    d[y] = kops.sum(d[x], axis=axis, keepdims=keepdims)


@tangent_(kops.mean)
def tangent_mean(y, x, axis=None, keepdims=False):
    """Forward mode for keras.ops.mean."""
    d[y] = kops.mean(d[x], axis=axis, keepdims=keepdims)


@tangent_(kops.matmul)
def tangent_matmul(z, x1, x2):
    """Forward mode for keras.ops.matmul."""
    d[z] = kops.matmul(d[x1], x2) + kops.matmul(x1, d[x2])


@tangent_(kops.reshape)
def tangent_reshape(y, x, newshape):
    """Forward mode for keras.ops.reshape."""
    d[y] = kops.reshape(d[x], newshape)


@tangent_(kops.transpose)
def tangent_transpose(y, x, axes=None):
    """Forward mode for keras.ops.transpose."""
    d[y] = kops.transpose(d[x], axes=axes)


@tangent_(kops.relu)
def tangent_relu(y, x):
    """Forward mode for keras.ops.relu."""
    mask = kops.cast(x > 0, keras.backend.standardize_dtype(x.dtype))
    d[y] = d[x] * mask


@tangent_(kops.sigmoid)
def tangent_sigmoid(y, x):
    """Forward mode for keras.ops.sigmoid."""
    d[y] = d[x] * y * (1.0 - y)


print(f"✓ Keras extensions loaded successfully (keras {keras.__version__}, "
      f"backend: {keras.backend.backend()})")
