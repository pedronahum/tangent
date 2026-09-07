# Copyright 2017 Google Inc.
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
"""JAX extensions for Tangent automatic differentiation.

This module provides gradient definitions for JAX operations, enabling
Tangent to differentiate functions that use JAX's numpy-like API (jax.numpy).

JAX is a natural fit for Tangent because:
1. Both use source transformation approaches
2. JAX's functional style aligns with AD requirements
3. JAX arrays are immutable (no in-place modifications)
4. JIT compilation provides excellent performance

Example:
    import jax.numpy as jnp
    import tangent

    def f(x):
        return jnp.sum(jnp.dot(x, x) ** 2)

    df = tangent.grad(f)
    gradient = df(jnp.array([1.0, 2.0, 3.0]))
"""
from __future__ import absolute_import

from numbers import Number

try:
    import jax
    import jax.numpy as jnp
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
from tangent.utils import register_init_grad
from tangent.utils import register_shape_function


# JAX array type - detect the actual implementation type
# In JAX, arrays are instances of ArrayImpl, not the abstract Array type
# We need to get the concrete type for registration
_test_array = jnp.array(1.0)
ArrayType = type(_test_array)  # This will be jaxlib._jax.ArrayImpl
del _test_array


def size(x, axis):
    """Get the size of array along given axes."""
    axis_shape = x.shape if axis is None else tuple(x.shape[a] for a in axis)
    return max(int(np.prod(axis_shape)), 1)


def dtype(arr):
    """Get the dtype of a JAX array."""
    return arr.dtype


def shape_as_list(arr):
    """Get shape as a list."""
    return list(arr.shape)


def jax_shapes_match(a, b):
    """Check if two JAX arrays have matching shapes."""
    return jnp.shape(a) == jnp.shape(b)


# Register shape functions
register_shape_function(ArrayType, shape_as_list)

# Register non-differentiable functions (shape queries, constructors, etc.)
non_differentiable.register_non_differentiable_functions(
    jnp.shape, jnp.size, jnp.ndim,
    jnp.zeros, jnp.ones, jnp.empty,
    jnp.zeros_like, jnp.ones_like, jnp.empty_like,
    jnp.full, jnp.full_like,
    jnp.eye, jnp.identity,
    jnp.arange, jnp.linspace, jnp.logspace,
    size, shape_as_list, dtype
)

# Register gradient initializers
register_init_grad(ArrayType, jnp.zeros_like)

# Register add_grad and shape checking for JAX arrays
# Only register JAX-specific combinations to avoid conflicts with TF extensions
from tangent import utils as _utils

# Register for JAX Array type only
try:
    _utils.register_add_grad(ArrayType, ArrayType, jnp.add)
except ValueError:
    pass  # Already registered

# Register JAX Array with Python numerics (if not already registered)
for num_type in [float, int, Number]:
    try:
        _utils.register_add_grad(ArrayType, num_type, jnp.add)
        _utils.register_add_grad(num_type, ArrayType, jnp.add)
    except ValueError:
        pass  # Already registered, skip

# Register shape checker for JAX arrays
try:
    _utils.register_shape_checker(ArrayType, ArrayType, jax_shapes_match)
except ValueError:
    pass  # Already registered

# Register shape checker for JAX arrays with Python numerics
for num_type in [float, int, Number]:
    try:
        _utils.register_shape_checker(ArrayType, num_type, jax_shapes_match)
        _utils.register_shape_checker(num_type, ArrayType, jax_shapes_match)
    except ValueError:
        pass  # Already registered


# `@` (matmul) operator gradients for JAX arrays
def jax_matmul_grad_x(dz, x, y):
    dz = jnp.asarray(dz)
    if x.ndim == 1 and y.ndim == 1:
        return dz * y
    if x.ndim == 2 and y.ndim == 1:
        return jnp.outer(dz, y)
    return jnp.matmul(dz, jnp.swapaxes(y, -1, -2))


def jax_matmul_grad_y(dz, x, y):
    dz = jnp.asarray(dz)
    if x.ndim == 1 and y.ndim == 1:
        return dz * x
    if x.ndim == 1 and y.ndim == 2:
        return jnp.outer(x, dz)
    return jnp.matmul(jnp.swapaxes(x, -1, -2), dz)


_utils.register_matmul_grad(ArrayType, jax_matmul_grad_x, jax_matmul_grad_y)


# Type mixing support: NumPy <-> JAX conversion
# This handles cases where Python operators (like **) return NumPy arrays
# when operating on JAX arrays, causing type mixing in gradient accumulation
def add_grad_numpy_to_jax(left, right):
    """Add NumPy array to JAX array by converting to JAX.

    This is needed because Python's ** operator on JAX arrays can return NumPy arrays,
    causing type mixing in gradient accumulation.

    Args:
        left: NumPy array
        right: JAX array

    Returns:
        JAX array (sum)
    """
    # Convert NumPy array to JAX array
    left_jax = jnp.array(left)
    return jnp.add(left_jax, right)


def add_grad_jax_to_numpy(left, right):
    """Add JAX array to NumPy array by converting to JAX.

    Args:
        left: JAX array
        right: NumPy array

    Returns:
        JAX array (sum)
    """
    # Convert NumPy array to JAX array
    right_jax = jnp.array(right)
    return jnp.add(left, right_jax)


# Register NumPy <-> JAX conversions
try:
    _utils.register_add_grad(np.ndarray, ArrayType, add_grad_numpy_to_jax)
    _utils.register_add_grad(ArrayType, np.ndarray, add_grad_jax_to_numpy)
except ValueError as e:
    # Already registered, skip
    if "already mapped" not in str(e):
        raise

# JAX-specific unbroadcast and unreduce functions
def jax_unbroadcast_to(array, shape):
    """Reverse the broadcasting operation for JAX arrays."""
    axis = utils.create_unbroadcast_axis(shape, jnp.shape(array))
    return jnp.reshape(jnp.sum(array, axis=axis), shape)


def jax_unbroadcast(array, like):
    """Unbroadcast a JAX array to match the shape of 'like'."""
    return jax_unbroadcast_to(array, jnp.shape(like))


def jax_unreduce(array, shape, axis, keepdims):
    """Reverse summing over a dimension for JAX arrays.

    This matches the NumPy implementation: when keepdims=False, we need to
    expand dims along the reduced axes before broadcasting.
    """
    # When axis is not None and keepdims is False, need to expand dims
    if axis is not None and not keepdims:
        if isinstance(axis, int):
            axis = (axis,)
        # Expand dims along the reduced axes
        for ax in sorted(axis):
            array = jnp.expand_dims(array, ax)
    return jnp.broadcast_to(array, shape)


# Register unbroadcast for JAX array type
try:
    _utils.unbroadcasters[ArrayType] = jax_unbroadcast
except (AttributeError, KeyError):
    pass

# Register unreduce for JAX array type
try:
    _utils.unreducers[ArrayType] = jax_unreduce
except (AttributeError, KeyError):
    pass


# ============================================================================
# Reverse-mode (adjoint) gradient definitions
# ============================================================================

# Basic arithmetic operations
@adjoint(jnp.add)
def adjoint_add(z, x, y):
    """Adjoint for jnp.add: ∂L/∂x = ∂L/∂z, ∂L/∂y = ∂L/∂z"""
    d[x] = tangent.unbroadcast(d[z], x)
    d[y] = tangent.unbroadcast(d[z], y)


@adjoint(jnp.subtract)
def adjoint_subtract(z, x, y):
    """Adjoint for jnp.subtract: ∂L/∂x = ∂L/∂z, ∂L/∂y = -∂L/∂z"""
    d[x] = tangent.unbroadcast(d[z], x)
    d[y] = tangent.unbroadcast(-d[z], y)


@adjoint(jnp.multiply)
def adjoint_multiply(z, x, y):
    """Adjoint for jnp.multiply: ∂L/∂x = y·∂L/∂z, ∂L/∂y = x·∂L/∂z"""
    d[x] = tangent.unbroadcast(d[z] * y, x)
    d[y] = tangent.unbroadcast(d[z] * x, y)


@adjoint(jnp.divide)
def adjoint_divide(z, x, y):
    """Adjoint for jnp.divide: ∂L/∂x = ∂L/∂z/y, ∂L/∂y = -x·∂L/∂z/y²"""
    d[x] = tangent.unbroadcast(d[z] / y, x)
    d[y] = tangent.unbroadcast(-d[z] * x / (y ** 2), y)


@adjoint(jnp.true_divide)
def adjoint_true_divide(z, x, y):
    """Adjoint for jnp.true_divide (same as divide)"""
    d[x] = tangent.unbroadcast(d[z] / y, x)
    d[y] = tangent.unbroadcast(-d[z] * x / (y ** 2), y)


@adjoint(jnp.power)
def adjoint_power(y, x, n):
    """Adjoint for jnp.power: ∂L/∂x = n·x^(n-1)·∂L/∂z"""
    d[x] = tangent.unbroadcast(d[y] * n * jnp.power(x, n - 1), x)


# Unary elementwise ops: generated (adjoint AND tangent per op) from the
# backend-neutral rule table. jnp exposes several NumPy-style aliases as
# distinct objects in some versions, so both spellings are registered where
# they exist.
register_elementwise(
    'jax',
    ops={
        'exp': jnp.exp,
        'expm1': jnp.expm1,
        'exp2': jnp.exp2,
        'log': jnp.log,
        'log2': jnp.log2,
        'log10': jnp.log10,
        'log1p': jnp.log1p,
        'sqrt': jnp.sqrt,
        'square': jnp.square,
        'reciprocal': jnp.reciprocal,
        'negative': jnp.negative,
        'abs': (jnp.abs, jnp.absolute),
        'sin': jnp.sin,
        'cos': jnp.cos,
        'tan': jnp.tan,
        'arcsin': (jnp.arcsin, getattr(jnp, 'asin', None)),
        'arccos': (jnp.arccos, getattr(jnp, 'acos', None)),
        'arctan': (jnp.arctan, getattr(jnp, 'atan', None)),
        'sinh': jnp.sinh,
        'cosh': jnp.cosh,
        'tanh': jnp.tanh,
        'floor': jnp.floor,
        'ceil': jnp.ceil,
        'round': jnp.round,
        'sign': jnp.sign,
    },
    vocab=prefix_vocab('jnp', mask_pos='(({arg}) > 0)'),
)


# Activation functions (common in ML)
@adjoint(jax.nn.relu)
def adjoint_relu(y, x):
    """Adjoint for relu: ∂L/∂x = (x > 0)·∂L/∂z"""
    d[x] = d[y] * (x > 0)


@adjoint(jax.nn.sigmoid)
def adjoint_sigmoid(y, x):
    """Adjoint for sigmoid: ∂L/∂x = sigmoid(x)·(1-sigmoid(x))·∂L/∂z"""
    sig = jax.nn.sigmoid(x)
    d[x] = d[y] * sig * (1.0 - sig)


@adjoint(jax.nn.softplus)
def adjoint_softplus(y, x):
    """Adjoint for softplus: ∂L/∂x = sigmoid(x)·∂L/∂z"""
    d[x] = d[y] * jax.nn.sigmoid(x)


# Reduction operations
@adjoint(jnp.sum)
def adjoint_sum(y, x, axis=None, keepdims=False):
    """Adjoint for jnp.sum: ∂L/∂x = unreduce(∂L/∂z)"""
    d[x] = tangent.unreduce(d[y], tangent.shape_as_list(x), axis, keepdims)


@adjoint(jnp.mean)
def adjoint_mean(y, x, axis=None, keepdims=False):
    """Adjoint for jnp.mean: ∂L/∂x = unreduce(∂L/∂z) / size"""
    n = tangent.size(x, axis)
    d[x] = tangent.unreduce(d[y], tangent.shape_as_list(x), axis, keepdims) / n


@adjoint(jnp.max)
def adjoint_max(y, x, axis=None, keepdims=False):
    """Adjoint for jnp.max: ∂L/∂x_i = ∂L/∂z if x_i == max(x) else 0"""
    # Gradient flows only to the maximum element(s)
    max_val = jnp.max(x, axis=axis, keepdims=True)
    mask = (x == max_val).astype(x.dtype)
    # Normalize if multiple maxima
    num_max = jnp.sum(mask, axis=axis, keepdims=True)
    d[x] = tangent.unreduce(d[y], tangent.shape_as_list(x), axis, keepdims) * mask / num_max


@adjoint(jnp.min)
def adjoint_min(y, x, axis=None, keepdims=False):
    """Adjoint for jnp.min: ∂L/∂x_i = ∂L/∂z if x_i == min(x) else 0"""
    min_val = jnp.min(x, axis=axis, keepdims=True)
    mask = (x == min_val).astype(x.dtype)
    num_min = jnp.sum(mask, axis=axis, keepdims=True)
    d[x] = tangent.unreduce(d[y], tangent.shape_as_list(x), axis, keepdims) * mask / num_min


# Linear algebra operations
@adjoint(jnp.dot)
def adjoint_dot(z, x, y):
    """Adjoint for jnp.dot (matrix/vector multiplication).

    For vectors: x·y = Σ x_i·y_i
        ∂L/∂x_i = y_i·∂L/∂z
        ∂L/∂y_i = x_i·∂L/∂z

    For matrices: (X @ Y) = Z
        ∂L/∂X = ∂L/∂Z @ Y^T
        ∂L/∂Y = X^T @ ∂L/∂Z
    """
    if x.ndim == 1 and y.ndim == 1:
        # Vector dot product
        d[x] = d[z] * y
        d[y] = d[z] * x
    elif x.ndim == 2 and y.ndim == 2:
        # Matrix multiplication
        d[x] = jnp.dot(d[z], y.T)
        d[y] = jnp.dot(x.T, d[z])
    elif x.ndim == 2 and y.ndim == 1:
        # Matrix-vector multiplication
        d[x] = jnp.outer(d[z], y)
        d[y] = jnp.dot(x.T, d[z])
    elif x.ndim == 1 and y.ndim == 2:
        # Vector-matrix multiplication
        d[x] = jnp.dot(d[z], y.T)
        d[y] = jnp.outer(x, d[z])
    else:
        # General case
        d[x] = jnp.tensordot(d[z], y, axes=[[-1], [-1]])
        d[y] = jnp.tensordot(x, d[z], axes=[[-2], [0]])


@adjoint(jnp.matmul)
def adjoint_matmul(z, x, y):
    """Adjoint for jnp.matmul covering vector and matrix cases."""
    if x.ndim == 1 and y.ndim == 1:
        d[x] = d[z] * y
        d[y] = d[z] * x
    elif x.ndim == 2 and y.ndim == 1:
        d[x] = jnp.outer(d[z], y)
        d[y] = jnp.matmul(jnp.swapaxes(x, -2, -1), d[z])
    elif x.ndim == 1 and y.ndim == 2:
        d[x] = jnp.matmul(d[z], jnp.swapaxes(y, -2, -1))
        d[y] = jnp.outer(x, d[z])
    else:
        d[x] = jnp.matmul(d[z], jnp.swapaxes(y, -2, -1))
        d[y] = jnp.matmul(jnp.swapaxes(x, -2, -1), d[z])


@adjoint(jnp.transpose)
def adjoint_transpose(y, x, axes=None):
    """Adjoint for jnp.transpose: ∂L/∂x = transpose(∂L/∂z)"""
    if axes is None:
        d[x] = jnp.transpose(d[y])
    else:
        # Invert the permutation
        inv_axes = [0] * len(axes)
        for i, ax in enumerate(axes):
            inv_axes[ax] = i
        d[x] = jnp.transpose(d[y], inv_axes)


@adjoint(jnp.reshape)
def adjoint_reshape(y, x, newshape):
    """Adjoint for jnp.reshape: ∂L/∂x = reshape(∂L/∂z, original_shape)"""
    d[x] = jnp.reshape(d[y], x.shape)


@adjoint(jnp.squeeze)
def adjoint_squeeze(y, x, axis=None):
    """Adjoint for jnp.squeeze: ∂L/∂x = reshape back to the original shape"""
    d[x] = jnp.reshape(d[y], x.shape)


@adjoint(jnp.expand_dims)
def adjoint_expand_dims(y, x, axis):
    """Adjoint for jnp.expand_dims: ∂L/∂x = squeeze(∂L/∂z)"""
    d[x] = jnp.squeeze(d[y], axis=axis)


# Element-wise operations
@adjoint(jnp.maximum)
def adjoint_maximum(z, x, y):
    """Adjoint for jnp.maximum: gradient flows to the larger argument."""
    d[x] = tangent.unbroadcast(d[z] * (x >= y).astype(x.dtype), x)
    d[y] = tangent.unbroadcast(d[z] * (y > x).astype(y.dtype), y)


@adjoint(jnp.minimum)
def adjoint_minimum(z, x, y):
    """Adjoint for jnp.minimum: gradient flows to the smaller argument."""
    d[x] = tangent.unbroadcast(d[z] * (x <= y).astype(x.dtype), x)
    d[y] = tangent.unbroadcast(d[z] * (y < x).astype(y.dtype), y)


@adjoint(jnp.clip)
def adjoint_clip(y, x, a_min, a_max):
    """Adjoint for jnp.clip: gradient flows only where x is not clipped"""
    mask = ((x >= a_min) & (x <= a_max)).astype(x.dtype)
    d[x] = d[y] * mask


@adjoint(jnp.where)
def adjoint_where(z, condition, x, y):
    """Adjoint for jnp.where: gradient goes to x if condition else y"""
    d[x] = jnp.where(condition, d[z], jnp.zeros_like(d[z]))
    d[y] = jnp.where(condition, jnp.zeros_like(d[z]), d[z])


# Indexing operations
@adjoint(jnp.take)
def adjoint_take(y, x, indices, axis=None):
    """Adjoint for jnp.take: scatter the gradient back to the indexed
    positions. Only the axis=None (flattened-index) form is supported."""
    d[x] = jnp.reshape(
        jnp.zeros_like(jnp.ravel(x)).at[jnp.ravel(indices)].add(
            jnp.ravel(d[y])), x.shape)


# Concatenation and stacking.
#
# jnp.concatenate / jnp.stack take a *list* of arrays, but Tangent can only
# distribute gradients to varargs (a list is not a differentiable container).
# concat_desugar rewrites list-literal calls into the varargs helpers below
# (tangent.concat_seq / tangent.stack_seq), which carry varargs adjoints
# modelled on the numpy.broadcast_arrays adjoint.

def concat_seq(axis, *arrays):
    """Runtime helper: concatenate a varargs sequence of arrays."""
    return jnp.concatenate(list(arrays), axis=axis)


def stack_seq(axis, *arrays):
    """Runtime helper: stack a varargs sequence of arrays."""
    return jnp.stack(list(arrays), axis=axis)


def concat_split_points(arrays, axis):
    """Cumulative sizes of all but the last array (split indices for jnp.split).

    Shapes are static runtime values, so this is plain (non-differentiable)
    integer arithmetic. Exposed via the tangent module so the generated code
    can resolve it.
    """
    points = []
    total = 0
    for arr in arrays[:-1]:
        total = total + arr.shape[axis]
        points.append(total)
    return points


non_differentiable.register_non_differentiable_functions(
    concat_split_points)


@adjoint(concat_seq)
def adjoint_concat_seq(z, axis, *arrays):
    """Adjoint for concat_seq: split the gradient back to the original arrays."""
    d[arrays] = tuple(jnp.split(d[z], tangent.concat_split_points(arrays, axis),
                                axis=axis))


@adjoint(stack_seq)
def adjoint_stack_seq(z, axis, *arrays):
    """Adjoint for stack_seq: unstack the gradient along the stacked axis."""
    d[arrays] = tuple(jnp.moveaxis(d[z], axis, 0))


# The list-argument forms cannot be differentiated directly; they are only
# reachable when the desugar pass could not rewrite the call (e.g. the list is
# a variable, not a literal). Raise a clear error rather than generating broken
# code.
@adjoint(jnp.concatenate)
def adjoint_concatenate(dz, arrays, axis=0):
    """Not differentiable: pass a list literal so it can be desugared."""
    raise NotImplementedError(
        'tangent can only differentiate jnp.concatenate/stack when the list of '
        'arrays is a literal. Bind the list to a variable and pass its '
        'elements explicitly, or use a list literal.')


@adjoint(jnp.stack)
def adjoint_stack(dz, arrays, axis=0):
    """Not differentiable: pass a list literal so it can be desugared."""
    raise NotImplementedError(
        'tangent can only differentiate jnp.concatenate/stack when the list of '
        'arrays is a literal. Bind the list to a variable and pass its '
        'elements explicitly, or use a list literal.')


# JAX neural network activations (jax.nn.*)
import jax.nn

# Register adjoints for all wrapped versions of relu
# JAX wraps functions in custom_jvp -> PjitFunction -> function
# We need to register for all levels so Tangent finds the adjoint no matter which one it resolves
@adjoint(jax.nn.relu)
def adjoint_jax_relu(y, x):
    """Adjoint for jax.nn.relu: gradient flows where x > 0."""
    # Use (x > 0) which JAX automatically converts to float in multiplication
    # This works with both scalars and arrays without needing jnp.where
    d[x] = d[y] * (x > 0)

# Also register for the unwrapped versions
if hasattr(jax.nn.relu, '__wrapped__'):
    @adjoint(jax.nn.relu.__wrapped__)
    def adjoint_jax_relu_pjit(y, x):
        """Adjoint for jax.nn.relu (PjitFunction version)."""
        # Use (x > 0) which JAX automatically converts to float in multiplication
        d[x] = d[y] * (x > 0)

    if hasattr(jax.nn.relu.__wrapped__, '__wrapped__'):
        @adjoint(jax.nn.relu.__wrapped__.__wrapped__)
        def adjoint_jax_relu_fn(y, x):
            """Adjoint for jax.nn.relu (unwrapped function)."""
            # Use (x > 0) which JAX automatically converts to float in multiplication
            d[x] = d[y] * (x > 0)


@adjoint(jax.nn.sigmoid)
def adjoint_jax_sigmoid(y, x):
    """Adjoint for jax.nn.sigmoid: ∂L/∂x = sigmoid(x) * (1 - sigmoid(x)) * ∂L/∂y."""
    sig = jax.nn.sigmoid(x)
    d[x] = d[y] * sig * (1.0 - sig)

# Register unwrapped versions
if hasattr(jax.nn.sigmoid, '__wrapped__'):
    @adjoint(jax.nn.sigmoid.__wrapped__)
    def adjoint_jax_sigmoid_pjit(y, x):
        """Adjoint for jax.nn.sigmoid (PjitFunction version)."""
        sig = jax.nn.sigmoid(x)
        d[x] = d[y] * sig * (1.0 - sig)


@adjoint(jax.nn.softplus)
def adjoint_jax_softplus(y, x):
    """Adjoint for jax.nn.softplus: ∂L/∂x = sigmoid(x) * ∂L/∂y."""
    d[x] = d[y] * jax.nn.sigmoid(x)


@adjoint(jax.nn.log_sigmoid)
def adjoint_jax_log_sigmoid(y, x):
    """Adjoint for jax.nn.log_sigmoid: ∂L/∂x = (1 - sigmoid(x)) * ∂L/∂y."""
    d[x] = d[y] * (1.0 - jax.nn.sigmoid(x))


@adjoint(jax.nn.elu)
def adjoint_jax_elu(y, x, alpha=1.0):
    """Adjoint for jax.nn.elu: ∂L/∂x = (x > 0 ? 1 : alpha * exp(x)) * ∂L/∂y."""
    d[x] = d[y] * jnp.where(x > 0, 1.0, alpha * jnp.exp(x))


@adjoint(jax.nn.leaky_relu)
def adjoint_jax_leaky_relu(y, x, negative_slope=0.01):
    """Adjoint for jax.nn.leaky_relu: ∂L/∂x = (x > 0 ? 1 : negative_slope) * ∂L/∂y."""
    d[x] = d[y] * jnp.where(x > 0, 1.0, negative_slope)


@adjoint(jax.nn.selu)
def adjoint_jax_selu(y, x):
    """Adjoint for jax.nn.selu (scaled ELU)."""
    alpha = 1.67326324
    scale = 1.05070098
    d[x] = d[y] * scale * jnp.where(x > 0, 1.0, alpha * jnp.exp(x))


@adjoint(jax.nn.softmax)
def adjoint_jax_softmax(y, x, axis=-1):
    """Adjoint for jax.nn.softmax: d[x] = y * (dy - sum(dy * y, axis))."""
    d[x] = y * (d[y] - jnp.sum(d[y] * y, axis=axis, keepdims=True))


@adjoint(jax.nn.log_softmax)
def adjoint_jax_log_softmax(y, x, axis=-1):
    """Adjoint for jax.nn.log_softmax: d[x] = dy - exp(y) * sum(dy, axis)."""
    d[x] = d[y] - jnp.exp(y) * jnp.sum(d[y], axis=axis, keepdims=True)


@adjoint(jax.nn.gelu)
def adjoint_jax_gelu(y, x, approximate=True):
    """Adjoint for jax.nn.gelu: Gaussian Error Linear Unit."""
    # GELU gradient is complex; use JAX's built-in implementation
    import jax
    def gelu_fn(x_):
        return jax.nn.gelu(x_, approximate=approximate)
    # Use JAX to compute the gradient
    _, vjp_fn = jax.vjp(gelu_fn, x)
    d[x] = vjp_fn(d[y])[0]


#
# Forward Mode (Tangent) Definitions
#

# Arithmetic Operations
@tangent_(jnp.add)
def tangent_jnp_add(z, x, y):
    """Forward mode for jnp.add."""
    d[z] = jnp.add(d[x], d[y])


@tangent_(jnp.subtract)
def tangent_jnp_subtract(z, x, y):
    """Forward mode for jnp.subtract."""
    d[z] = jnp.subtract(d[x], d[y])


@tangent_(jnp.multiply)
def tangent_jnp_multiply(z, x, y):
    """Forward mode for jnp.multiply: d[z] = d[x]*y + x*d[y]."""
    d[z] = jnp.add(jnp.multiply(d[x], y), jnp.multiply(x, d[y]))


@tangent_(jnp.divide)
def tangent_jnp_divide(z, x, y):
    """Forward mode for jnp.divide: d[z] = (d[x]*y - x*d[y]) / y^2."""
    d[z] = jnp.divide(
        jnp.subtract(jnp.multiply(d[x], y), jnp.multiply(x, d[y])),
        jnp.multiply(y, y)
    )


@tangent_(jnp.true_divide)
def tangent_jnp_true_divide(z, x, y):
    """Forward mode for jnp.true_divide."""
    d[z] = jnp.divide(
        jnp.subtract(jnp.multiply(d[x], y), jnp.multiply(x, d[y])),
        jnp.multiply(y, y)
    )


@tangent_(jnp.power)
def tangent_jnp_power(z, x, y):
    """Forward mode for jnp.power: d[z] = d[x]*y*x^(y-1) + d[y]*x^y*log(x)."""
    d[z] = jnp.add(
        jnp.multiply(d[x], jnp.multiply(y, jnp.power(x, y - 1))),
        jnp.multiply(d[y], jnp.multiply(z, jnp.log(x)))
    )


# Unary elementwise tangents (exp, log, trig, ...) are generated alongside
# their adjoints by the register_elementwise call above.


# Reduction Operations
@tangent_(jnp.sum)
def tangent_jnp_sum(y, x, axis=None, dtype=None, keepdims=False):
    """Forward mode for jnp.sum."""
    d[y] = jnp.sum(d[x], axis=axis, dtype=dtype, keepdims=keepdims)


@tangent_(jnp.mean)
def tangent_jnp_mean(y, x, axis=None, dtype=None, keepdims=False):
    """Forward mode for jnp.mean."""
    d[y] = jnp.mean(d[x], axis=axis, dtype=dtype, keepdims=keepdims)


@tangent_(jnp.max)
def tangent_jnp_max(y, x, axis=None, keepdims=False):
    """Forward mode for jnp.max."""
    # Create mask where x equals the maximum
    if axis is None:
        mask = jnp.equal(x, y)
    else:
        y_expanded = jnp.expand_dims(y, axis) if not keepdims else y
        mask = jnp.equal(x, y_expanded)
    d[y] = jnp.sum(jnp.multiply(d[x], mask), axis=axis, keepdims=keepdims)


@tangent_(jnp.min)
def tangent_jnp_min(y, x, axis=None, keepdims=False):
    """Forward mode for jnp.min."""
    # Create mask where x equals the minimum
    if axis is None:
        mask = jnp.equal(x, y)
    else:
        y_expanded = jnp.expand_dims(y, axis) if not keepdims else y
        mask = jnp.equal(x, y_expanded)
    d[y] = jnp.sum(jnp.multiply(d[x], mask), axis=axis, keepdims=keepdims)


# Linear Algebra Operations
@tangent_(jnp.dot)
def tangent_jnp_dot(z, x, y):
    """Forward mode for jnp.dot: d[z] = dot(d[x], y) + dot(x, d[y])."""
    d[z] = jnp.add(jnp.dot(d[x], y), jnp.dot(x, d[y]))


@tangent_(jnp.matmul)
def tangent_jnp_matmul(z, x, y):
    """Forward mode for jnp.matmul."""
    d[z] = jnp.add(jnp.matmul(d[x], y), jnp.matmul(x, d[y]))


# Shape Manipulation
@tangent_(jnp.transpose)
def tangent_jnp_transpose(y, x, axes=None):
    """Forward mode for jnp.transpose."""
    d[y] = jnp.transpose(d[x], axes=axes)


@tangent_(jnp.reshape)
def tangent_jnp_reshape(y, x, shape):
    """Forward mode for jnp.reshape."""
    d[y] = jnp.reshape(d[x], shape)


@tangent_(jnp.squeeze)
def tangent_jnp_squeeze(y, x, axis=None):
    """Forward mode for jnp.squeeze."""
    d[y] = jnp.squeeze(d[x], axis=axis)


@tangent_(jnp.expand_dims)
def tangent_jnp_expand_dims(y, x, axis):
    """Forward mode for jnp.expand_dims."""
    d[y] = jnp.expand_dims(d[x], axis=axis)


# Comparison and Selection
@tangent_(jnp.maximum)
def tangent_jnp_maximum(z, x, y):
    """Forward mode for jnp.maximum."""
    d[z] = jnp.add(
        jnp.multiply(d[x], jnp.where(jnp.greater(x, y), 1.0, 0.0)),
        jnp.multiply(d[y], jnp.where(jnp.greater(y, x), 1.0, 0.0))
    )


@tangent_(jnp.minimum)
def tangent_jnp_minimum(z, x, y):
    """Forward mode for jnp.minimum."""
    d[z] = jnp.add(
        jnp.multiply(d[x], jnp.where(jnp.less(x, y), 1.0, 0.0)),
        jnp.multiply(d[y], jnp.where(jnp.less(y, x), 1.0, 0.0))
    )


@tangent_(jnp.clip)
def tangent_jnp_clip(y, x, a_min=None, a_max=None):
    """Forward mode for jnp.clip."""
    # Gradient is zero where clipped, d[x] elsewhere
    mask = jnp.ones_like(x)
    if a_min is not None:
        mask = jnp.where(jnp.less(x, a_min), 0.0, mask)
    if a_max is not None:
        mask = jnp.where(jnp.greater(x, a_max), 0.0, mask)
    d[y] = jnp.multiply(d[x], mask)


@tangent_(jnp.where)
def tangent_jnp_where(result, condition, x, y):
    """Forward mode for jnp.where."""
    d[result] = jnp.where(condition, d[x], d[y])


# Array Construction and Manipulation
@tangent_(jnp.concatenate)
def tangent_jnp_concatenate(result, arrays, axis=0):
    """Forward mode for jnp.concatenate."""
    # Get tangents of all input arrays
    tangent_arrays = [d[arr] for arr in arrays]
    d[result] = jnp.concatenate(tangent_arrays, axis=axis)


@tangent_(jnp.stack)
def tangent_jnp_stack(result, arrays, axis=0):
    """Forward mode for jnp.stack."""
    tangent_arrays = [d[arr] for arr in arrays]
    d[result] = jnp.stack(tangent_arrays, axis=axis)


# Varargs concatenation helpers (targets of the concat_desugar rewrite).
@tangent_(concat_seq)
def tangent_concat_seq(z, axis, *arrays):
    """Forward mode for concat_seq."""
    d[z] = tangent.concat_seq(axis, *d[arrays])


@tangent_(stack_seq)
def tangent_stack_seq(z, axis, *arrays):
    """Forward mode for stack_seq."""
    d[z] = tangent.stack_seq(axis, *d[arrays])


# Neural Network Activation Functions
@tangent_(jax.nn.relu)
def tangent_jax_relu(y, x):
    """Forward mode for jax.nn.relu: d[y] = d[x] where x > 0, else 0."""
    d[y] = jnp.where(jnp.greater(x, 0), d[x], 0.0)


@tangent_(jax.nn.sigmoid)
def tangent_jax_sigmoid(y, x):
    """Forward mode for jax.nn.sigmoid: d[y] = d[x] * sigmoid(x) * (1 - sigmoid(x))."""
    d[y] = jnp.multiply(d[x], jnp.multiply(y, 1.0 - y))


@tangent_(jax.nn.softplus)
def tangent_jax_softplus(y, x):
    """Forward mode for jax.nn.softplus: d[y] = d[x] * sigmoid(x)."""
    d[y] = jnp.multiply(d[x], jax.nn.sigmoid(x))


@tangent_(jax.nn.log_sigmoid)
def tangent_jax_log_sigmoid(y, x):
    """Forward mode for jax.nn.log_sigmoid: d[y] = d[x] * (1 - sigmoid(x))."""
    d[y] = jnp.multiply(d[x], 1.0 - jax.nn.sigmoid(x))


@tangent_(jax.nn.elu)
def tangent_jax_elu(y, x, alpha=1.0):
    """Forward mode for jax.nn.elu."""
    # ELU gradient: 1 if x > 0, else alpha * exp(x)
    grad = jnp.where(jnp.greater(x, 0), 1.0, alpha * jnp.exp(x))
    d[y] = jnp.multiply(d[x], grad)


@tangent_(jax.nn.leaky_relu)
def tangent_jax_leaky_relu(y, x, negative_slope=0.01):
    """Forward mode for jax.nn.leaky_relu."""
    grad = jnp.where(jnp.greater(x, 0), 1.0, negative_slope)
    d[y] = jnp.multiply(d[x], grad)


@tangent_(jax.nn.selu)
def tangent_jax_selu(y, x):
    """Forward mode for jax.nn.selu."""
    # SELU constants
    alpha = 1.67326324
    scale = 1.05070098
    # Gradient
    grad = jnp.where(jnp.greater(x, 0), scale, scale * alpha * jnp.exp(x))
    d[y] = jnp.multiply(d[x], grad)


@tangent_(jax.nn.softmax)
def tangent_jax_softmax(y, x, axis=-1):
    """Forward mode for jax.nn.softmax."""
    d[y] = y * (d[x] - jnp.sum(d[x] * y, axis=axis, keepdims=True))


@tangent_(jax.nn.log_softmax)
def tangent_jax_log_softmax(y, x, axis=-1):
    """Forward mode for jax.nn.log_softmax: dy = dx - sum(softmax(x) * dx)."""
    d[y] = d[x] - jnp.sum(jnp.exp(y) * d[x], axis=axis, keepdims=True)


@tangent_(jax.nn.gelu)
def tangent_jax_gelu(y, x, approximate=True):
    """Forward mode for jax.nn.gelu."""
    # Use JAX's built-in gradient
    import jax
    def gelu_fn(x_):
        return jax.nn.gelu(x_, approximate=approximate)
    # Compute JVP
    _, jvp_result = jax.jvp(gelu_fn, (x,), (d[x],))
    d[y] = jvp_result


import logging as _logging
_logging.getLogger('tangent').debug(
    'JAX extensions loaded successfully (JAX %s, %d gradient definitions)',
    jax.__version__, len([f for f in dir() if f.startswith('adjoint_')]))
