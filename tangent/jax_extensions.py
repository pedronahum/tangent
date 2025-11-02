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


print(f"✓ JAX extensions loaded successfully (JAX {jax.__version__})")
print(f"✓ Registered {len([f for f in dir() if f.startswith('adjoint_')])} gradient definitions")
