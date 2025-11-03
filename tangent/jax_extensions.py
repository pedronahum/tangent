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

import warnings
from numbers import Number

try:
    import jax
    import jax.numpy as jnp
    from jax import Array as JAXArray
except ImportError as e:
    warnings.warn(f"JAX not available: {e}. Install with: pip install jax jaxlib")
    raise

import numpy as np
from tangent import grads
from tangent import non_differentiable
from tangent import tangents
from tangent import utils
from tangent.grads import adjoint
from tangent.tangents import tangent_
from tangent.utils import array_shapes_match
from tangent.utils import register_all_add_grad
from tangent.utils import register_all_shape_checker
from tangent.utils import register_init_grad
from tangent.utils import register_shape_function
from tangent.utils import register_unbroadcast
from tangent.utils import register_unreduce


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


@adjoint(jnp.negative)
def adjoint_negative(y, x):
    """Adjoint for jnp.negative: ∂L/∂x = -∂L/∂z"""
    d[x] = -d[y]


# Exponential and logarithmic functions
@adjoint(jnp.exp)
def adjoint_exp(y, x):
    """Adjoint for jnp.exp: ∂L/∂x = exp(x)·∂L/∂z"""
    d[x] = d[y] * jnp.exp(x)


@adjoint(jnp.log)
def adjoint_log(y, x):
    """Adjoint for jnp.log: ∂L/∂x = ∂L/∂z/x"""
    d[x] = d[y] / x


@adjoint(jnp.log10)
def adjoint_log10(y, x):
    """Adjoint for jnp.log10: ∂L/∂x = ∂L/∂z/(x·ln(10))"""
    d[x] = d[y] / (x * jnp.log(10.0))


@adjoint(jnp.log2)
def adjoint_log2(y, x):
    """Adjoint for jnp.log2: ∂L/∂x = ∂L/∂z/(x·ln(2))"""
    d[x] = d[y] / (x * jnp.log(2.0))


@adjoint(jnp.sqrt)
def adjoint_sqrt(y, x):
    """Adjoint for jnp.sqrt: ∂L/∂x = ∂L/∂z/(2√x)"""
    d[x] = d[y] / (2.0 * jnp.sqrt(x))


@adjoint(jnp.square)
def adjoint_square(y, x):
    """Adjoint for jnp.square: ∂L/∂x = 2x·∂L/∂z"""
    d[x] = 2.0 * x * d[y]


# Trigonometric functions
@adjoint(jnp.sin)
def adjoint_sin(y, x):
    """Adjoint for jnp.sin: ∂L/∂x = cos(x)·∂L/∂z"""
    d[x] = d[y] * jnp.cos(x)


@adjoint(jnp.cos)
def adjoint_cos(y, x):
    """Adjoint for jnp.cos: ∂L/∂x = -sin(x)·∂L/∂z"""
    d[x] = -d[y] * jnp.sin(x)


@adjoint(jnp.tan)
def adjoint_tan(y, x):
    """Adjoint for jnp.tan: ∂L/∂x = sec²(x)·∂L/∂z = ∂L/∂z/cos²(x)"""
    d[x] = d[y] / (jnp.cos(x) ** 2)


# Inverse trigonometric functions
@adjoint(jnp.arcsin)
def adjoint_arcsin(y, x):
    """Adjoint for jnp.arcsin: ∂L/∂x = ∂L/∂z/√(1-x²)"""
    d[x] = d[y] / jnp.sqrt(1.0 - x**2)


@adjoint(jnp.arccos)
def adjoint_arccos(y, x):
    """Adjoint for jnp.arccos: ∂L/∂x = -∂L/∂z/√(1-x²)"""
    d[x] = -d[y] / jnp.sqrt(1.0 - x**2)


@adjoint(jnp.arctan)
def adjoint_arctan(y, x):
    """Adjoint for jnp.arctan: ∂L/∂x = ∂L/∂z/(1+x²)"""
    d[x] = d[y] / (1.0 + x**2)


# Hyperbolic functions
@adjoint(jnp.sinh)
def adjoint_sinh(y, x):
    """Adjoint for jnp.sinh: ∂L/∂x = cosh(x)·∂L/∂z"""
    d[x] = d[y] * jnp.cosh(x)


@adjoint(jnp.cosh)
def adjoint_cosh(y, x):
    """Adjoint for jnp.cosh: ∂L/∂x = sinh(x)·∂L/∂z"""
    d[x] = d[y] * jnp.sinh(x)


@adjoint(jnp.tanh)
def adjoint_tanh(y, x):
    """Adjoint for jnp.tanh: ∂L/∂x = (1 - tanh²(x))·∂L/∂z"""
    d[x] = d[y] * (1.0 - jnp.tanh(x) ** 2)


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
def adjoint_matmul(dz, x, y):
    """Adjoint for jnp.matmul (same as dot for 2D arrays)."""
    return jnp.matmul(dz, jnp.swapaxes(y, -2, -1)), jnp.matmul(jnp.swapaxes(x, -2, -1), dz)


@adjoint(jnp.transpose)
def adjoint_transpose(dz, x, axes=None):
    """Adjoint for jnp.transpose: ∂L/∂x = transpose(∂L/∂z)"""
    if axes is None:
        return jnp.transpose(dz)
    else:
        # Invert the permutation
        inv_axes = [0] * len(axes)
        for i, ax in enumerate(axes):
            inv_axes[ax] = i
        return jnp.transpose(dz, inv_axes)


@adjoint(jnp.reshape)
def adjoint_reshape(dz, x, newshape):
    """Adjoint for jnp.reshape: ∂L/∂x = reshape(∂L/∂z, original_shape)"""
    return jnp.reshape(dz, x.shape)


@adjoint(jnp.squeeze)
def adjoint_squeeze(dz, x, axis=None):
    """Adjoint for jnp.squeeze: ∂L/∂x = expand_dims(∂L/∂z)"""
    return jnp.reshape(dz, x.shape)


@adjoint(jnp.expand_dims)
def adjoint_expand_dims(dz, x, axis):
    """Adjoint for jnp.expand_dims: ∂L/∂x = squeeze(∂L/∂z)"""
    return jnp.squeeze(dz, axis=axis)


# Element-wise operations
@adjoint(jnp.abs)
def adjoint_abs(dz, x):
    """Adjoint for jnp.abs: ∂L/∂x = sign(x)·∂L/∂z"""
    return dz * jnp.sign(x)


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
def adjoint_clip(dz, x, a_min, a_max):
    """Adjoint for jnp.clip: gradient flows only where x is not clipped"""
    mask = ((x >= a_min) & (x <= a_max)).astype(x.dtype)
    return dz * mask, (), ()


@adjoint(jnp.where)
def adjoint_where(dz, condition, x, y):
    """Adjoint for jnp.where: gradient goes to x if condition else y"""
    return (), jnp.where(condition, dz, jnp.zeros_like(dz)), jnp.where(condition, jnp.zeros_like(dz), dz)


# Indexing operations
@adjoint(jnp.take)
def adjoint_take(dz, x, indices, axis=None):
    """Adjoint for jnp.take: scatter gradient back to indexed positions"""
    # This is a simplified version; full implementation needs scatter
    dx = jnp.zeros_like(x)
    if axis is None:
        # Flatten case
        return dx.flatten().at[indices].add(dz.flatten()).reshape(x.shape)
    else:
        # Need proper scatter along axis
        return dx


# Concatenation and stacking
@adjoint(jnp.concatenate)
def adjoint_concatenate(dz, arrays, axis=0):
    """Adjoint for jnp.concatenate: split gradient back to original arrays"""
    sizes = [arr.shape[axis] for arr in arrays]
    return tuple(jnp.split(dz, jnp.cumsum(jnp.array(sizes[:-1])), axis=axis))


@adjoint(jnp.stack)
def adjoint_stack(dz, arrays, axis=0):
    """Adjoint for jnp.stack: unstack gradient"""
    return tuple(jnp.moveaxis(dz, axis, 0))


# JAX neural network activations (jax.nn.*)
import jax.nn

# Register adjoints for all wrapped versions of relu
# JAX wraps functions in custom_jvp -> PjitFunction -> function
# We need to register for all levels so Tangent finds the adjoint no matter which one it resolves
@adjoint(jax.nn.relu)
def adjoint_jax_relu(y, x):
    """Adjoint for jax.nn.relu: gradient flows where x > 0."""
    d[x] = d[y] * (x > 0).astype(x.dtype)

# Also register for the unwrapped versions
if hasattr(jax.nn.relu, '__wrapped__'):
    @adjoint(jax.nn.relu.__wrapped__)
    def adjoint_jax_relu_pjit(y, x):
        """Adjoint for jax.nn.relu (PjitFunction version)."""
        d[x] = d[y] * (x > 0).astype(x.dtype)

    if hasattr(jax.nn.relu.__wrapped__, '__wrapped__'):
        @adjoint(jax.nn.relu.__wrapped__.__wrapped__)
        def adjoint_jax_relu_fn(y, x):
            """Adjoint for jax.nn.relu (unwrapped function)."""
            d[x] = d[y] * (x > 0).astype(x.dtype)


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


@tangent_(jnp.negative)
def tangent_jnp_negative(y, x):
    """Forward mode for jnp.negative."""
    d[y] = jnp.negative(d[x])


# Exponential and Logarithmic Functions
@tangent_(jnp.exp)
def tangent_jnp_exp(y, x):
    """Forward mode for jnp.exp: d[y] = d[x] * exp(x)."""
    d[y] = jnp.multiply(d[x], y)


@tangent_(jnp.log)
def tangent_jnp_log(y, x):
    """Forward mode for jnp.log: d[y] = d[x] / x."""
    d[y] = jnp.divide(d[x], x)


@tangent_(jnp.log10)
def tangent_jnp_log10(y, x):
    """Forward mode for jnp.log10: d[y] = d[x] / (x * ln(10))."""
    d[y] = jnp.divide(d[x], jnp.multiply(x, jnp.log(10.0)))


@tangent_(jnp.log2)
def tangent_jnp_log2(y, x):
    """Forward mode for jnp.log2: d[y] = d[x] / (x * ln(2))."""
    d[y] = jnp.divide(d[x], jnp.multiply(x, jnp.log(2.0)))


@tangent_(jnp.sqrt)
def tangent_jnp_sqrt(y, x):
    """Forward mode for jnp.sqrt: d[y] = d[x] / (2*sqrt(x))."""
    d[y] = jnp.divide(d[x], jnp.multiply(2.0, y))


@tangent_(jnp.square)
def tangent_jnp_square(y, x):
    """Forward mode for jnp.square: d[y] = 2*x*d[x]."""
    d[y] = jnp.multiply(jnp.multiply(2.0, x), d[x])


# Trigonometric Functions
@tangent_(jnp.sin)
def tangent_jnp_sin(y, x):
    """Forward mode for jnp.sin: d[y] = d[x] * cos(x)."""
    d[y] = jnp.multiply(d[x], jnp.cos(x))


@tangent_(jnp.cos)
def tangent_jnp_cos(y, x):
    """Forward mode for jnp.cos: d[y] = -d[x] * sin(x)."""
    d[y] = jnp.negative(jnp.multiply(d[x], jnp.sin(x)))


@tangent_(jnp.tan)
def tangent_jnp_tan(y, x):
    """Forward mode for jnp.tan: d[y] = d[x] / cos^2(x)."""
    cx = jnp.cos(x)
    d[y] = jnp.divide(d[x], jnp.multiply(cx, cx))


# Inverse Trigonometric Functions
@tangent_(jnp.arcsin)
def tangent_jnp_arcsin(y, x):
    """Forward mode for jnp.arcsin: d[y] = d[x] / √(1-x²)."""
    d[y] = jnp.divide(d[x], jnp.sqrt(1.0 - x**2))


@tangent_(jnp.arccos)
def tangent_jnp_arccos(y, x):
    """Forward mode for jnp.arccos: d[y] = -d[x] / √(1-x²)."""
    d[y] = jnp.negative(jnp.divide(d[x], jnp.sqrt(1.0 - x**2)))


@tangent_(jnp.arctan)
def tangent_jnp_arctan(y, x):
    """Forward mode for jnp.arctan: d[y] = d[x] / (1+x²)."""
    d[y] = jnp.divide(d[x], 1.0 + x**2)


# Hyperbolic Functions
@tangent_(jnp.sinh)
def tangent_jnp_sinh(y, x):
    """Forward mode for jnp.sinh: d[y] = d[x] * cosh(x)."""
    d[y] = jnp.multiply(d[x], jnp.cosh(x))


@tangent_(jnp.cosh)
def tangent_jnp_cosh(y, x):
    """Forward mode for jnp.cosh: d[y] = d[x] * sinh(x)."""
    d[y] = jnp.multiply(d[x], jnp.sinh(x))


@tangent_(jnp.tanh)
def tangent_jnp_tanh(y, x):
    """Forward mode for jnp.tanh: d[y] = d[x] / cosh^2(x)."""
    cx = jnp.cosh(x)
    d[y] = jnp.divide(d[x], jnp.multiply(cx, cx))


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
@tangent_(jnp.abs)
def tangent_jnp_abs(y, x):
    """Forward mode for jnp.abs: d[y] = d[x] * sign(x)."""
    d[y] = jnp.multiply(d[x], jnp.sign(x))


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


print(f"✓ JAX extensions loaded successfully (JAX {jax.__version__})")
print(f"✓ Registered {len([f for f in dir() if f.startswith('adjoint_')])} gradient definitions")
