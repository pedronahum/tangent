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
from tangent import non_differentiable
from tangent.grads import adjoint
from tangent.tangents import tangent_

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
    d[x] = -d[y] / (x**2)


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


# numpy.concatenate / numpy.stack take a *list* of arrays, which Tangent
# cannot distribute gradients into. concat_desugar rewrites list-literal calls
# into the varargs helpers below (mirroring the JAX concat_seq/stack_seq
# machinery), whose varargs adjoints split the gradient back per input.


def np_concat_seq(axis, *arrays):
    """Runtime helper: concatenate a varargs sequence of arrays."""
    return numpy.concatenate(list(arrays), axis=axis)


def np_stack_seq(axis, *arrays):
    """Runtime helper: stack a varargs sequence of arrays."""
    return numpy.stack(list(arrays), axis=axis)


def np_concat_grads(dz, arrays, axis):
    """Split a concatenated gradient back into per-input gradients."""
    dz = numpy.asarray(dz)
    points = numpy.cumsum([a.shape[axis] for a in arrays[:-1]])
    return tuple(numpy.split(dz, points, axis=axis))


def np_stack_grads(dz, arrays, axis):
    """Unstack a stacked gradient along the stacking axis."""
    dz = numpy.asarray(dz)
    return tuple(numpy.moveaxis(dz, axis, 0))


non_differentiable.register_non_differentiable_functions(np_concat_grads, np_stack_grads)


@adjoint(np_concat_seq)
def adjoint_np_concat_seq(z, axis, *arrays):
    """Adjoint for np_concat_seq: split the gradient back per input."""
    d[arrays] = tangent.np_concat_grads(d[z], arrays, axis)


@adjoint(np_stack_seq)
def adjoint_np_stack_seq(z, axis, *arrays):
    """Adjoint for np_stack_seq: unstack the gradient."""
    d[arrays] = tangent.np_stack_grads(d[z], arrays, axis)


@tangent_(np_concat_seq)
def tangent_np_concat_seq(z, axis, *arrays):
    """Forward mode for np_concat_seq."""
    d[z] = tangent.np_concat_seq(axis, *d[arrays])


@tangent_(np_stack_seq)
def tangent_np_stack_seq(z, axis, *arrays):
    """Forward mode for np_stack_seq."""
    d[z] = tangent.np_stack_seq(axis, *d[arrays])


# The list-argument forms are reached when the desugar pass could not rewrite
# the call into the varargs helpers - i.e. the list is built dynamically
# (appends in a loop). List gradients are first-class now, so the adjoint
# hands back a *list* of per-element gradients, which flows through the
# list_append machinery like any other list.
@adjoint(numpy.concatenate)
def concatenate(z, arrays, axis=0):
    d[arrays] = tangent.unconcatenate(d[z], arrays, axis)


@adjoint(numpy.stack)
def stack(z, arrays, axis=0):
    d[arrays] = tangent.unstack_list(d[z], axis)


@tangent_(numpy.concatenate)
def tconcatenate(z, arrays, axis=0):
    d[z] = numpy.concatenate(d[arrays], axis)


@tangent_(numpy.stack)
def tstack(z, arrays, axis=0):
    d[z] = numpy.stack(d[arrays], axis)


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
# Forward-mode (tangent) definitions
#
# These mirror the adjoints above. Elementwise rules use the input tangent
# directly (multiplication broadcasts a scalar seed); shape/linear-algebra
# rules broadcast the tangent to the input's shape first, because
# forward-mode seeds arrive as scalars and those operations need a
# full-shape operand (same convention as tangents.py).
# ============================================================================


@tangent_(numpy.absolute)
def tabsolute(z, x):
    """Forward mode for numpy.absolute: dz = dx * sign(x)."""
    d[z] = d[x] * numpy.sign(x)


# Register alias (numpy.abs is numpy.absolute, but keep symmetry with the
# adjoint registrations above in case they ever diverge).
tangent_(numpy.abs)(tabsolute)


@tangent_(numpy.reciprocal)
def treciprocal(z, x):
    """Forward mode for numpy.reciprocal: dz = -dx / x**2."""
    d[z] = -d[x] / (x * x)


@tangent_(numpy.log10)
def tlog10(z, x):
    """Forward mode for numpy.log10: dz = dx / (x * ln(10))."""
    d[z] = d[x] / (x * numpy.log(10.0))


@tangent_(numpy.log2)
def tlog2(z, x):
    """Forward mode for numpy.log2: dz = dx / (x * ln(2))."""
    d[z] = d[x] / (x * numpy.log(2.0))


@tangent_(numpy.log1p)
def tlog1p(z, x):
    """Forward mode for numpy.log1p: dz = dx / (1 + x)."""
    d[z] = d[x] / (1.0 + x)


@tangent_(numpy.expm1)
def texpm1(z, x):
    """Forward mode for numpy.expm1: dz = dx * exp(x)."""
    d[z] = d[x] * numpy.exp(x)


@tangent_(numpy.min)
def tmin(z, x, axis=None, keepdims=False):
    """Forward mode for numpy.min: the tangent of the minimal element(s)."""
    if axis is None:
        min_val = z
    else:
        min_val = numpy.expand_dims(z, axis) if not keepdims else z
    mask = (x == min_val).astype(x.dtype)
    num_min = numpy.sum(mask, axis=axis, keepdims=keepdims)
    d[z] = numpy.sum(d[x] * mask, axis=axis, keepdims=keepdims) / num_min


@tangent_(numpy.max)
def tmax(z, x, axis=None, keepdims=False):
    """Forward mode for numpy.max: the tangent of the maximal element(s)."""
    if axis is None:
        max_val = z
    else:
        max_val = numpy.expand_dims(z, axis) if not keepdims else z
    mask = (x == max_val).astype(x.dtype)
    num_max = numpy.sum(mask, axis=axis, keepdims=keepdims)
    d[z] = numpy.sum(d[x] * mask, axis=axis, keepdims=keepdims) / num_max


@tangent_(numpy.prod)
def tprod(z, x, axis=None, keepdims=False):
    """Forward mode for numpy.prod: dz = sum(dx * prod(x) / x_i)."""
    d[z] = numpy.sum(
        d[x] * tangent.unreduce(z, numpy.shape(x), axis, keepdims) / x, axis=axis, keepdims=keepdims
    )


@tangent_(numpy.matmul)
def tmatmul(z, x, y):
    """Forward mode for numpy.matmul: dz = dx @ y + x @ dy."""
    d[z] = numpy.matmul(numpy.broadcast_to(d[x], numpy.shape(x)), y) + numpy.matmul(
        x, numpy.broadcast_to(d[y], numpy.shape(y))
    )


@tangent_(numpy.linalg.inv)
def tinv(z, x):
    """Forward mode for numpy.linalg.inv: dz = -z @ dx @ z."""
    d[z] = -numpy.matmul(numpy.matmul(z, numpy.broadcast_to(d[x], numpy.shape(x))), z)


@tangent_(numpy.outer)
def touter(z, a, b):
    """Forward mode for numpy.outer: dz = outer(da, b) + outer(a, db)."""
    d[z] = numpy.outer(numpy.broadcast_to(d[a], numpy.shape(a)), b) + numpy.outer(
        a, numpy.broadcast_to(d[b], numpy.shape(b))
    )


@tangent_(numpy.trace)
def ttrace(z, x):
    """Forward mode for numpy.trace: dz = trace(dx)."""
    d[z] = numpy.trace(numpy.broadcast_to(d[x], numpy.shape(x)))


@tangent_(numpy.squeeze)
def tsqueeze(z, x, axis=None):
    """Forward mode for numpy.squeeze."""
    d[z] = numpy.squeeze(numpy.broadcast_to(d[x], numpy.shape(x)), axis=axis)


@tangent_(numpy.expand_dims)
def texpand_dims(z, x, axis):
    """Forward mode for numpy.expand_dims."""
    d[z] = numpy.expand_dims(numpy.broadcast_to(d[x], numpy.shape(x)), axis)


@tangent_(numpy.clip)
def tclip(z, x, a_min, a_max):
    """Forward mode for numpy.clip: the tangent flows where x is unclipped."""
    d[z] = d[x] * numpy.logical_and(x >= a_min, x <= a_max).astype(x.dtype)


@tangent_(numpy.where)
def twhere(result, condition, x, y):
    """Forward mode for numpy.where: pick the tangent of the selected arm."""
    d[result] = numpy.where(condition, d[x], d[y])


@tangent_(numpy.sign)
def tsign(z, x):
    """Forward mode for numpy.sign: zero tangent (piecewise constant)."""
    d[z] = numpy.zeros_like(x)


@tangent_(numpy.floor)
def tfloor(z, x):
    """Forward mode for numpy.floor: zero tangent (piecewise constant)."""
    d[z] = numpy.zeros_like(x)


@tangent_(numpy.ceil)
def tceil(z, x):
    """Forward mode for numpy.ceil: zero tangent (piecewise constant)."""
    d[z] = numpy.zeros_like(x)


@tangent_(numpy.var)
def tvar(z, x, axis=None, ddof=0, keepdims=False):
    """Forward mode for numpy.var: dz = sum(2 (x - mean) dx) / (n - ddof)."""
    x_mean = numpy.mean(x, axis=axis, keepdims=True)
    if axis is None:
        n = x.size
    else:
        n = numpy.prod([x.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])
    d[z] = numpy.sum(2.0 * (x - x_mean) * d[x], axis=axis, keepdims=keepdims) / (n - ddof)


@tangent_(numpy.std)
def tstd(z, x, axis=None, ddof=0, keepdims=False):
    """Forward mode for numpy.std: dz = sum((x - mean) dx) / ((n - ddof) z)."""
    x_mean = numpy.mean(x, axis=axis, keepdims=True)
    if axis is None:
        n = x.size
    else:
        n = numpy.prod([x.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])
    d[z] = numpy.sum((x - x_mean) * d[x], axis=axis, keepdims=keepdims) / ((n - ddof) * z)


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
    numpy.absolute,
    numpy.abs,
    numpy.square,
    numpy.reciprocal,
    numpy.log10,
    numpy.log2,
    numpy.log1p,
    numpy.expm1,
    numpy.min,
    numpy.max,
    numpy.prod,
    numpy.matmul,
    numpy.linalg.inv,
    numpy.outer,
    numpy.trace,
    numpy.squeeze,
    numpy.expand_dims,
    numpy.concatenate,
    numpy.stack,
    numpy.minimum,
    numpy.clip,
    numpy.where,
    numpy.sign,
    numpy.floor,
    numpy.ceil,
    numpy.var,
    numpy.std,
]

# Remove from UNIMPLEMENTED_ADJOINTS
for func in _our_functions:
    _grads_module.UNIMPLEMENTED_ADJOINTS.discard(func)

# Same for the forward-mode registry: UNIMPLEMENTED_TANGENTS is computed at
# tangents.py load time, before this module registers its tangents.
from tangent import tangents as _tangents_module

_our_tangent_functions = [
    numpy.absolute,
    numpy.abs,
    numpy.reciprocal,
    numpy.log10,
    numpy.log2,
    numpy.log1p,
    numpy.expm1,
    numpy.min,
    numpy.max,
    numpy.prod,
    numpy.matmul,
    numpy.linalg.inv,
    numpy.outer,
    numpy.trace,
    numpy.squeeze,
    numpy.expand_dims,
    numpy.clip,
    numpy.where,
    numpy.sign,
    numpy.floor,
    numpy.ceil,
    numpy.var,
    numpy.std,
    # numpy.concatenate / numpy.stack deliberately stay unimplemented as
    # direct tangents: list-literal calls are desugared to np_concat_seq /
    # np_stack_seq (which have tangents), and anything else should raise a
    # clear forward-mode not-implemented error.
]

for func in _our_tangent_functions:
    _tangents_module.UNIMPLEMENTED_TANGENTS.discard(func)

import logging as _logging

_logging.getLogger('tangent').debug(
    'Extended NumPy gradients loaded successfully (%d new gradient definitions)',
    len(_our_functions),
)
