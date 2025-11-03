"""Extended NumPy gradient definitions for Tangent.

This module adds gradient definitions for commonly-used NumPy operations
that are not yet covered in grads.py. These gradients mirror the JAX
implementations in jax_extensions.py, since NumPy and JAX have compatible APIs.

The additions focus on:
1. Element-wise operations (abs, square, negative, reciprocal)
2. Reduction operations (min, max, prod)
3. Linear algebra (matmul, inv, outer)
4. Shape operations (squeeze, expand_dims, concatenate, stack)
5. Element-wise comparison/selection (minimum, maximum, clip, where)
6. Additional math functions (log10, log2, log1p, expm1)
"""
from __future__ import absolute_import

import numpy
import tangent
from tangent.grads import adjoint

# ============================================================================
# Element-wise Operations
# ============================================================================

@adjoint(numpy.absolute)
def absolute(y, x):
    """Adjoint for numpy.absolute (numpy.abs): ∂L/∂x = sign(x)·∂L/∂z"""
    d[x] = d[y] * numpy.sign(x)


# Register alias
adjoint(numpy.abs)(absolute)


@adjoint(numpy.square)
def square(y, x):
    """Adjoint for numpy.square: ∂L/∂x = 2x·∂L/∂z"""
    d[x] = 2.0 * x * d[y]


@adjoint(numpy.reciprocal)
def reciprocal(y, x):
    """Adjoint for numpy.reciprocal: ∂L/∂x = -∂L/∂z/x²"""
    d[x] = -d[y] / (x ** 2)


# ============================================================================
# Logarithmic Functions (additional variants)
# ============================================================================

@adjoint(numpy.log10)
def log10(y, x):
    """Adjoint for numpy.log10: ∂L/∂x = ∂L/∂z/(x·ln(10))"""
    d[x] = d[y] / (x * numpy.log(10.0))


@adjoint(numpy.log2)
def log2(y, x):
    """Adjoint for numpy.log2: ∂L/∂x = ∂L/∂z/(x·ln(2))"""
    d[x] = d[y] / (x * numpy.log(2.0))


@adjoint(numpy.log1p)
def log1p(y, x):
    """Adjoint for numpy.log1p (log(1+x)): ∂L/∂x = ∂L/∂z/(1+x)"""
    d[x] = d[y] / (1.0 + x)


@adjoint(numpy.expm1)
def expm1(y, x):
    """Adjoint for numpy.expm1 (exp(x)-1): ∂L/∂x = exp(x)·∂L/∂z"""
    d[x] = d[y] * numpy.exp(x)


# ============================================================================
# Reduction Operations
# ============================================================================

@adjoint(numpy.min)
def min_(y, x, axis=None, keepdims=False):
    """Adjoint for numpy.min: gradient flows only to minimum element(s)"""
    # Find which elements equal the minimum
    if axis is None:
        min_val = y
    else:
        min_val = numpy.expand_dims(y, axis) if not keepdims else y

    # Create mask for minimum values
    mask = (x == min_val).astype(x.dtype)
    # Normalize if multiple minima (split gradient equally)
    num_min = numpy.sum(mask, axis=axis, keepdims=True)

    # Unreduce gradient and apply mask
    d[x] = tangent.unreduce(d[y], numpy.shape(x), axis, keepdims) * mask / num_min


@adjoint(numpy.max)
def max_(y, x, axis=None, keepdims=False):
    """Adjoint for numpy.max: gradient flows only to maximum element(s)"""
    # Find which elements equal the maximum
    if axis is None:
        max_val = y
    else:
        max_val = numpy.expand_dims(y, axis) if not keepdims else y

    # Create mask for maximum values
    mask = (x == max_val).astype(x.dtype)
    # Normalize if multiple maxima
    num_max = numpy.sum(mask, axis=axis, keepdims=True)

    # Unreduce gradient and apply mask
    d[x] = tangent.unreduce(d[y], numpy.shape(x), axis, keepdims) * mask / num_max


@adjoint(numpy.prod)
def prod(y, x, axis=None, keepdims=False):
    """Adjoint for numpy.prod: ∂L/∂x_i = ∂L/∂z · prod(x) / x_i"""
    # Gradient is: d[y] * y / x
    # This works because d(∏x_i)/dx_j = (∏x_i) / x_j
    d[x] = tangent.unreduce(d[y], numpy.shape(x), axis, keepdims) * y / x


# ============================================================================
# Linear Algebra Operations
# ============================================================================

@adjoint(numpy.matmul)
def matmul(z, x, y):
    """Adjoint for numpy.matmul (matrix multiplication).

    For matrices: Z = X @ Y
        ∂L/∂X = ∂L/∂Z @ Y^T
        ∂L/∂Y = X^T @ ∂L/∂Z
    """
    d[x] = numpy.matmul(d[z], numpy.swapaxes(y, -2, -1))
    d[y] = numpy.matmul(numpy.swapaxes(x, -2, -1), d[z])


@adjoint(numpy.linalg.inv)
def inv(y, x):
    """Adjoint for numpy.linalg.inv (matrix inverse).

    For Y = inv(X):
        ∂L/∂X = -Y^T @ ∂L/∂Y @ Y^T

    This is the classic formula for the gradient of matrix inverse.
    """
    # y = inv(x), so we use it directly
    d[x] = -numpy.matmul(numpy.matmul(y.T, d[y]), y.T)


@adjoint(numpy.outer)
def outer(z, a, b):
    """Adjoint for numpy.outer: Z = outer(a, b) = a[:,None] @ b[None,:]

    ∂L/∂a = ∂L/∂Z @ b
    ∂L/∂b = ∂L/∂Z^T @ a
    """
    d[a] = numpy.dot(d[z], b)
    d[b] = numpy.dot(d[z].T, a)


@adjoint(numpy.trace)
def trace(y, x):
    """Adjoint for numpy.trace: ∂L/∂X_ij = ∂L/∂y if i==j else 0"""
    # Gradient flows only to diagonal elements
    d[x] = d[y] * numpy.eye(x.shape[0], x.shape[1])


# ============================================================================
# Shape Manipulation Operations
# ============================================================================

@adjoint(numpy.squeeze)
def squeeze(y, x, axis=None):
    """Adjoint for numpy.squeeze: ∂L/∂x = reshape(∂L/∂z, original_shape)"""
    d[x] = numpy.reshape(d[y], x.shape)


@adjoint(numpy.expand_dims)
def expand_dims(y, x, axis):
    """Adjoint for numpy.expand_dims: ∂L/∂x = squeeze(∂L/∂z, axis)"""
    d[x] = numpy.squeeze(d[y], axis=axis)


@adjoint(numpy.concatenate)
def concatenate(z, arrays, axis=0):
    """Adjoint for numpy.concatenate: split gradient back to original arrays"""
    # Compute split indices based on array sizes
    sizes = [arr.shape[axis] for arr in arrays]
    split_indices = numpy.cumsum(sizes[:-1])

    # Split the gradient
    grads = numpy.split(d[z], split_indices, axis=axis)
    for i, arr in enumerate(arrays):
        d[arr] = grads[i]


@adjoint(numpy.stack)
def stack(z, arrays, axis=0):
    """Adjoint for numpy.stack: unstack gradient along the stacking axis"""
    # Move the stacked axis to the front, then unstack
    d_moved = numpy.moveaxis(d[z], axis, 0)
    # Split along first dimension (now the stacked axis)
    for i, arr in enumerate(arrays):
        d[arr] = d_moved[i]


# ============================================================================
# Element-wise Comparison and Selection
# ============================================================================

@adjoint(numpy.minimum)
def minimum(z, x, y):
    """Adjoint for numpy.minimum: gradient flows to the smaller argument"""
    # Gradient goes to x where x < y, to y where y <= x
    d[x] = tangent.unbroadcast(d[z] * (x <= y).astype(x.dtype), x)
    d[y] = tangent.unbroadcast(d[z] * (y < x).astype(y.dtype), y)


@adjoint(numpy.clip)
def clip(y, x, a_min, a_max):
    """Adjoint for numpy.clip: gradient flows only where x is not clipped

    Note: This implementation assumes both a_min and a_max are provided.
    For cases where one is None, the gradient may not be correct.
    """
    # Gradient is 1 where x was not clipped, 0 where it was clipped
    # x is clipped if x < a_min or x > a_max
    mask = numpy.logical_and(x >= a_min, x <= a_max).astype(x.dtype)
    d[x] = d[y] * mask


@adjoint(numpy.where)
def where(result, condition, x, y):
    """Adjoint for numpy.where: gradient goes to x if condition else y"""
    # Gradient for x: where condition is True
    d[x] = tangent.unbroadcast(numpy.where(condition, d[result], numpy.zeros_like(d[result])), x)
    # Gradient for y: where condition is False
    d[y] = tangent.unbroadcast(numpy.where(condition, numpy.zeros_like(d[result]), d[result]), y)


# ============================================================================
# Utility Functions
# ============================================================================

@adjoint(numpy.sign)
def sign(y, x):
    """Adjoint for numpy.sign: gradient is zero (discontinuous function)"""
    # Sign function has zero gradient almost everywhere
    # (discontinuous at 0, but we use zero gradient)
    d[x] = numpy.zeros_like(x)


@adjoint(numpy.floor)
def floor(y, x):
    """Adjoint for numpy.floor: gradient is zero (discontinuous function)"""
    d[x] = numpy.zeros_like(x)


@adjoint(numpy.ceil)
def ceil(y, x):
    """Adjoint for numpy.ceil: gradient is zero (discontinuous function)"""
    d[x] = numpy.zeros_like(x)


# ============================================================================
# Statistics Operations
# ============================================================================

@adjoint(numpy.var)
def var(y, x, axis=None, ddof=0, keepdims=False):
    """Adjoint for numpy.var (variance): ∂L/∂x_i = 2(x_i - mean(x))·∂L/∂z/(n-ddof)"""
    # Compute mean
    x_mean = numpy.mean(x, axis=axis, keepdims=True)

    # Number of elements
    if axis is None:
        n = x.size
    else:
        n = numpy.prod([x.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])

    # Gradient: 2 * (x - mean(x)) / (n - ddof)
    grad = 2.0 * (x - x_mean) / (n - ddof)

    # Unreduce and multiply by incoming gradient
    d[x] = tangent.unreduce(d[y], numpy.shape(x), axis, keepdims) * grad


@adjoint(numpy.std)
def std(y, x, axis=None, ddof=0, keepdims=False):
    """Adjoint for numpy.std (standard deviation): ∂L/∂x_i = (x_i - mean)·∂L/∂z/(n·std)"""
    # Compute mean
    x_mean = numpy.mean(x, axis=axis, keepdims=True)

    # Number of elements
    if axis is None:
        n = x.size
    else:
        n = numpy.prod([x.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])

    # Compute std (use the result we already have)
    if axis is None:
        std_val = y
    else:
        std_val = numpy.expand_dims(y, axis) if not keepdims else y

    # Gradient: (x - mean) / ((n - ddof) * std)
    grad = (x - x_mean) / ((n - ddof) * std_val)

    # Unreduce and multiply by incoming gradient
    d[x] = tangent.unreduce(d[y], numpy.shape(x), axis, keepdims) * grad


# Update UNIMPLEMENTED_ADJOINTS to remove our newly registered functions
# This is necessary because UNIMPLEMENTED_ADJOINTS is computed at grads.py load time
# before this module was imported
from tangent import grads as _grads_module

# List of functions we registered
_our_functions = [
    numpy.absolute, numpy.abs, numpy.square, numpy.reciprocal,
    numpy.log10, numpy.log2, numpy.log1p, numpy.expm1,
    numpy.min, numpy.max, numpy.prod,
    numpy.matmul, numpy.linalg.inv, numpy.outer, numpy.trace,
    numpy.squeeze, numpy.expand_dims, numpy.concatenate, numpy.stack,
    numpy.minimum, numpy.clip, numpy.where,
    numpy.sign, numpy.floor, numpy.ceil,
    numpy.var, numpy.std
]

# Remove from UNIMPLEMENTED_ADJOINTS
for func in _our_functions:
    _grads_module.UNIMPLEMENTED_ADJOINTS.discard(func)

print("✓ Extended NumPy gradients loaded successfully")
print(f"✓ Registered {len(_our_functions)} new gradient definitions")
