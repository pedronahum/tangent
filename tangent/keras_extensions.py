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


try:
    import keras
    import keras.ops as kops
except ImportError:
    # Optional dependency; tangent/__init__.py reports the failure.
    raise

import numpy as np
from tangent import non_differentiable
from tangent.elementwise_rules import prefix_vocab
from tangent.elementwise_rules import register_elementwise
from tangent.grads import adjoint
from tangent.tangents import tangent_



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


def keras_dtype_name(x):
    """The standardized dtype name of a backend tensor, as a string.

    Used by adjoint templates so generated code only needs the
    tangent namespace, not a `keras` import in the user's module.
    """
    return keras.backend.standardize_dtype(x.dtype)


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
    keras_seed, keras_dtype_name, keras_max_mask, keras_min_mask
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


# Unary elementwise ops: generated (adjoint AND tangent per op) from the
# backend-neutral rule table. Optional spellings are guarded with getattr
# so older Keras versions simply skip them.
register_elementwise(
    'keras',
    ops={
        'exp': kops.exp,
        'exp2': getattr(kops, 'exp2', None),
        'expm1': getattr(kops, 'expm1', None),
        'log': kops.log,
        'log2': getattr(kops, 'log2', None),
        'log10': getattr(kops, 'log10', None),
        'log1p': getattr(kops, 'log1p', None),
        'sqrt': kops.sqrt,
        'rsqrt': getattr(kops, 'rsqrt', None),
        'square': kops.square,
        'reciprocal': getattr(kops, 'reciprocal', None),
        'negative': kops.negative,
        'abs': (kops.abs, getattr(kops, 'absolute', None)),
        'sin': kops.sin,
        'cos': kops.cos,
        'tan': kops.tan,
        'arcsin': kops.arcsin,
        'arccos': kops.arccos,
        'arctan': kops.arctan,
        'sinh': kops.sinh,
        'cosh': kops.cosh,
        'tanh': kops.tanh,
        'sigmoid': kops.sigmoid,
        'relu': kops.relu,
        'floor': kops.floor,
        'ceil': kops.ceil,
        'round': kops.round,
        'sign': kops.sign,
    },
    vocab=prefix_vocab(
        'kops',
        mask_pos='kops.cast(({arg}) > 0, tangent.keras_dtype_name(x))'),
    seed='tangent.keras_seed({g}, x)',
)


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
    m1 = kops.cast(x1 >= x2, tangent.keras_dtype_name(x1))
    m2 = kops.cast(x2 > x1, tangent.keras_dtype_name(x2))
    d[x1] = tangent.unbroadcast(dz * m1, x1)
    d[x2] = tangent.unbroadcast(dz * m2, x2)


@adjoint(kops.minimum)
def adjoint_minimum(z, x1, x2):
    """Adjoint for keras.ops.minimum."""
    dz = tangent.keras_seed(d[z], x1)
    m1 = kops.cast(x1 <= x2, tangent.keras_dtype_name(x1))
    m2 = kops.cast(x2 < x1, tangent.keras_dtype_name(x2))
    d[x1] = tangent.unbroadcast(dz * m1, x1)
    d[x2] = tangent.unbroadcast(dz * m2, x2)


@adjoint(kops.clip)
def adjoint_clip(y, x, x_min, x_max):
    """Adjoint for keras.ops.clip."""
    inside = kops.cast(
        kops.logical_and(x >= x_min, x <= x_max),
        tangent.keras_dtype_name(x))
    d[x] = tangent.keras_seed(d[y], x) * inside


@adjoint(kops.where)
def adjoint_where(z, condition, x1, x2):
    """Adjoint for keras.ops.where."""
    dz = tangent.keras_seed(d[z], x1)
    d[x1] = kops.where(condition, dz, kops.zeros_like(dz))
    d[x2] = kops.where(condition, kops.zeros_like(dz), dz)


# Softmax family
@adjoint(kops.softmax)
def adjoint_softmax(y, x, axis=-1):
    """Adjoint for keras.ops.softmax: d[x] = y * (dz - sum(dz * y, axis))."""
    s = tangent.keras_seed(d[y], x)
    d[x] = y * (s - kops.sum(s * y, axis=axis, keepdims=True))


@adjoint(kops.log_softmax)
def adjoint_log_softmax(y, x, axis=-1):
    """Adjoint for keras.ops.log_softmax: d[x] = dz - exp(y) * sum(dz, axis)."""
    s = tangent.keras_seed(d[y], x)
    d[x] = s - kops.exp(y) * kops.sum(s, axis=axis, keepdims=True)


# Concatenation and stacking.
#
# keras.ops.concatenate / stack take a *list* of tensors, which Tangent
# cannot distribute gradients into. concat_desugar rewrites list-literal
# calls into the varargs helpers below (mirroring the JAX
# concat_seq/stack_seq machinery), whose varargs adjoints split the gradient
# back per input.

def keras_concat_seq(axis, *tensors):
    """Runtime helper: concatenate a varargs sequence of tensors."""
    return kops.concatenate(list(tensors), axis=axis)


def keras_stack_seq(axis, *tensors):
    """Runtime helper: stack a varargs sequence of tensors."""
    return kops.stack(list(tensors), axis=axis)


def keras_concat_grads(dz, tensors, axis):
    """Split a concatenated gradient back into per-input gradients."""
    dz = keras_seed(dz, tensors[0])
    points = []
    total = 0
    for t in tensors[:-1]:
        total += int(t.shape[axis])
        points.append(total)
    return tuple(kops.split(dz, points, axis=axis))


def keras_stack_grads(dz, tensors, axis):
    """Unstack a stacked gradient along the stacking axis."""
    dz = keras_seed(dz, tensors[0])
    return tuple(kops.unstack(dz, axis=axis))


non_differentiable.register_non_differentiable_functions(
    keras_concat_grads, keras_stack_grads)


@adjoint(keras_concat_seq)
def adjoint_keras_concat_seq(z, axis, *tensors):
    """Adjoint for keras_concat_seq: split the gradient back per input."""
    d[tensors] = tangent.keras_concat_grads(d[z], tensors, axis)


@adjoint(keras_stack_seq)
def adjoint_keras_stack_seq(z, axis, *tensors):
    """Adjoint for keras_stack_seq: unstack the gradient."""
    d[tensors] = tangent.keras_stack_grads(d[z], tensors, axis)


# The list-argument forms are only reachable when the desugar pass could not
# rewrite the call (e.g. the list is built dynamically). Raise a clear error
# rather than generating broken code.
@adjoint(kops.concatenate)
def adjoint_concatenate(dz, xs, axis=0):
    """Not differentiable: pass a list literal so it can be desugared."""
    raise NotImplementedError(
        'tangent can only differentiate keras.ops.concatenate/stack when the '
        'list of tensors is a literal. Bind the list to a variable assigned '
        'once from a literal, or pass a list literal directly.')


@adjoint(kops.stack)
def adjoint_stack(dz, x, axis=0):
    """Not differentiable: pass a list literal so it can be desugared."""
    raise NotImplementedError(
        'tangent can only differentiate keras.ops.concatenate/stack when the '
        'list of tensors is a literal. Bind the list to a variable assigned '
        'once from a literal, or pass a list literal directly.')


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


# Unary elementwise tangents (exp, log, trig, ...) are generated alongside
# their adjoints by the register_elementwise call above.


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


@tangent_(kops.softmax)
def tangent_softmax(y, x, axis=-1):
    """Forward mode for keras.ops.softmax."""
    d[y] = y * (d[x] - kops.sum(d[x] * y, axis=axis, keepdims=True))


@tangent_(kops.log_softmax)
def tangent_log_softmax(y, x, axis=-1):
    """Forward mode for keras.ops.log_softmax: dy = dx - sum(softmax(x) * dx)."""
    d[y] = d[x] - kops.sum(kops.exp(y) * d[x], axis=axis, keepdims=True)


@tangent_(keras_concat_seq)
def tangent_keras_concat_seq(z, axis, *tensors):
    """Forward mode for keras_concat_seq."""
    d[z] = tangent.keras_concat_seq(axis, *d[tensors])


@tangent_(keras_stack_seq)
def tangent_keras_stack_seq(z, axis, *tensors):
    """Forward mode for keras_stack_seq."""
    d[z] = tangent.keras_stack_seq(axis, *d[tensors])


import logging as _logging
_logging.getLogger('tangent').debug(
    'Keras extensions loaded successfully (keras %s, backend: %s)',
    keras.__version__, keras.backend.backend())
