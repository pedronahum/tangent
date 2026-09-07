"""Extended TensorFlow gradient definitions for Tangent.

This module adds gradient definitions for commonly-used TensorFlow operations
that are not yet covered in tf_extensions.py. These gradients are inspired by
the NumPy and JAX implementations, adapted for TensorFlow 2.x.

The additions focus on:
1. Element-wise operations (abs, square, sqrt, sign, floor, ceil, reciprocal)
2. Logarithmic functions (log10, log2, log1p, expm1)
3. Reduction operations (reduce_min, reduce_prod)
4. Trigonometric functions (sin, cos, tan, asin, acos, atan)
5. Comparison operations (minimum, clip_by_value)
6. Neural network activations (relu, sigmoid, softmax)
"""
from __future__ import absolute_import


try:
    import tensorflow as tf
    import tangent
    from tangent import non_differentiable
    from tangent.grads import adjoint
    from tangent.tangents import tangent_
    # Side-effect import: registers the base TF adjoints/tangents that this
    # module extends.
    from tangent import tf_extensions  # noqa: F401
except ImportError:
    # Optional dependency; tangent/__init__.py reports the failure.
    raise

# ============================================================================
# Element-wise Operations
# ============================================================================

@adjoint(tf.abs)
def abs_(y, x):
    """Adjoint for tf.abs: ∂L/∂x = sign(x)·∂L/∂z"""
    d[x] = d[y] * tf.sign(x)


@adjoint(tf.square)
def square(y, x):
    """Adjoint for tf.square: ∂L/∂x = 2x·∂L/∂z"""
    d[x] = 2.0 * x * d[y]


@adjoint(tf.sqrt)
def sqrt(y, x):
    """Adjoint for tf.sqrt: ∂L/∂x = ∂L/∂z/(2√x) = ∂L/∂z/(2y)"""
    d[x] = d[y] / (2.0 * y)


@adjoint(tf.sign)
def sign(y, x):
    """Adjoint for tf.sign: gradient is zero (discontinuous function)"""
    d[x] = tf.zeros_like(x)


@adjoint(tf.floor)
def floor(y, x):
    """Adjoint for tf.floor: gradient is zero (discontinuous function)"""
    d[x] = tf.zeros_like(x)


# TF 2.x: ceil moved to tf.math.ceil
if hasattr(tf.math, 'ceil'):
    @adjoint(tf.math.ceil)
    def ceil_math(y, x):
        """Adjoint for tf.math.ceil: gradient is zero (discontinuous function)"""
        d[x] = tf.zeros_like(x)
elif hasattr(tf, 'ceil'):
    @adjoint(tf.ceil)
    def ceil_tf(y, x):
        """Adjoint for tf.ceil: gradient is zero (discontinuous function)"""
        d[x] = tf.zeros_like(x)


@adjoint(tf.round)
def round_(y, x):
    """Adjoint for tf.round: gradient is zero (discontinuous function)"""
    d[x] = tf.zeros_like(x)


# Try to register tf.reciprocal (may not exist in all TF versions)
try:
    @adjoint(tf.reciprocal)
    def reciprocal(y, x):
        """Adjoint for tf.reciprocal: ∂L/∂x = -∂L/∂z/x²"""
        d[x] = -d[y] / tf.square(x)
except AttributeError:
    pass  # tf.reciprocal not available


@adjoint(tf.minimum)
def minimum(z, x, y):
    """Adjoint for tf.minimum: gradient flows to the smaller argument"""
    # Gradient goes to x where x <= y, to y where y < x
    d[x] = tangent.unbroadcast_tensor(d[z] * tf.cast(x <= y, x.dtype), x)
    d[y] = tangent.unbroadcast_tensor(d[z] * tf.cast(y < x, y.dtype), y)


@adjoint(tf.clip_by_value)
def clip_by_value(y, x, clip_value_min, clip_value_max):
    """Adjoint for tf.clip_by_value: gradient flows only where x is not clipped"""
    # Gradient is 1 where x was not clipped, 0 where it was clipped
    mask = tf.logical_and(x >= clip_value_min, x <= clip_value_max)
    d[x] = d[y] * tf.cast(mask, x.dtype)


@adjoint(tf.where)
def where(z, condition, x, y):
    """Adjoint for tf.where(condition, x, y): gradient flows to x where the
    condition is true and to y where it is false."""
    d[x] = tangent.unbroadcast_tensor(
        tf.where(condition, d[z], tf.zeros_like(d[z])), x)
    d[y] = tangent.unbroadcast_tensor(
        tf.where(condition, tf.zeros_like(d[z]), d[z]), y)


# ============================================================================
# Logarithmic Functions
# ============================================================================

# Note: tf.math.log is already handled in tf_extensions.py as tf_log
# We register tf.math.log directly here for convenience

try:
    @adjoint(tf.math.log10)
    def log10(y, x):
        """Adjoint for tf.math.log10: ∂L/∂x = ∂L/∂z/(x·ln(10))"""
        d[x] = d[y] / (x * tf.math.log(10.0))
except AttributeError:
    pass  # log10 not available in this TF version


try:
    @adjoint(tf.math.log2)
    def log2(y, x):
        """Adjoint for tf.math.log2: ∂L/∂x = ∂L/∂z/(x·ln(2))"""
        d[x] = d[y] / (x * tf.math.log(2.0))
except AttributeError:
    pass  # log2 not available in this TF version


try:
    @adjoint(tf.math.log1p)
    def log1p(y, x):
        """Adjoint for tf.math.log1p (log(1+x)): ∂L/∂x = ∂L/∂z/(1+x)"""
        d[x] = d[y] / (1.0 + x)
except AttributeError:
    pass  # log1p not available


try:
    @adjoint(tf.math.expm1)
    def expm1(y, x):
        """Adjoint for tf.math.expm1 (exp(x)-1): ∂L/∂x = exp(x)·∂L/∂z"""
        d[x] = d[y] * tf.exp(x)
except AttributeError:
    pass  # expm1 not available


# ============================================================================
# Reduction Operations
# ============================================================================

@adjoint(tf.reduce_min)
def reduce_min(y, x, axis=None, keep_dims=False):
    """Adjoint for tf.reduce_min: gradient flows only to minimum element(s)"""
    # Find which elements equal the minimum
    # Unreduce y to match x's shape for comparison
    min_val_unreduced = tangent.unreduce(y, tangent.shape_as_list(x), axis, keep_dims)

    # Create mask for minimum values
    mask = tf.cast(tf.equal(x, min_val_unreduced), x.dtype)
    # Normalize if multiple minima (split gradient equally)
    num_min = tf.reduce_sum(mask, axis=axis, keepdims=True)

    # Unreduce gradient and apply mask
    grad_unreduced = tangent.unreduce(d[y], tangent.shape_as_list(x), axis, keep_dims)
    d[x] = grad_unreduced * mask / num_min


@adjoint(tf.reduce_prod)
def reduce_prod(y, x, axis=None, keep_dims=False):
    """Adjoint for tf.reduce_prod: ∂L/∂x_i = ∂L/∂z · prod(x) / x_i"""
    # Gradient is: d[y] * y / x
    # This works because d(∏x_i)/dx_j = (∏x_i) / x_j
    # Unreduce both y and d[y] to match x's shape
    y_unreduced = tangent.unreduce(y, tangent.shape_as_list(x), axis, keep_dims)
    grad_unreduced = tangent.unreduce(d[y], tangent.shape_as_list(x), axis, keep_dims)
    d[x] = grad_unreduced * y_unreduced / x


# ============================================================================
# Trigonometric Functions
# ============================================================================

@adjoint(tf.sin)
def sin(y, x):
    """Adjoint for tf.sin: ∂L/∂x = cos(x)·∂L/∂z"""
    d[x] = d[y] * tf.cos(x)


@adjoint(tf.cos)
def cos(y, x):
    """Adjoint for tf.cos: ∂L/∂x = -sin(x)·∂L/∂z"""
    d[x] = -d[y] * tf.sin(x)


@adjoint(tf.tan)
def tan(y, x):
    """Adjoint for tf.tan: ∂L/∂x = ∂L/∂z/cos²(x) = ∂L/∂z·(1 + tan²(x))"""
    d[x] = d[y] * (1.0 + tf.square(y))


try:
    @adjoint(tf.asin)
    def asin(y, x):
        """Adjoint for tf.asin: ∂L/∂x = ∂L/∂z/√(1-x²)"""
        d[x] = d[y] / tf.sqrt(1.0 - tf.square(x))
except AttributeError:
    pass  # asin not available


try:
    @adjoint(tf.acos)
    def acos(y, x):
        """Adjoint for tf.acos: ∂L/∂x = -∂L/∂z/√(1-x²)"""
        d[x] = -d[y] / tf.sqrt(1.0 - tf.square(x))
except AttributeError:
    pass  # acos not available


@adjoint(tf.atan)
def atan(y, x):
    """Adjoint for tf.atan: ∂L/∂x = ∂L/∂z/(1+x²)"""
    d[x] = d[y] / (1.0 + tf.square(x))


# ============================================================================
# Neural Network Activations
# ============================================================================

@adjoint(tf.nn.relu)
def relu(y, x):
    """Adjoint for tf.nn.relu: ∂L/∂x = ∂L/∂z where x > 0, else 0"""
    d[x] = d[y] * tf.cast(x > 0, x.dtype)


@adjoint(tf.nn.sigmoid)
def sigmoid(y, x):
    """Adjoint for tf.nn.sigmoid: ∂L/∂x = sigmoid(x)·(1-sigmoid(x))·∂L/∂z"""
    # y is already sigmoid(x), so gradient is y * (1 - y)
    d[x] = d[y] * y * (1.0 - y)


@adjoint(tf.nn.softmax)
def softmax(y, x, axis=-1):
    """Adjoint for tf.nn.softmax: ∂L/∂x_i = softmax(x)·(∂L/∂z - Σ(∂L/∂z·softmax(x)))"""
    # y is softmax(x)
    # Gradient: y * (dy - sum(dy * y))
    sum_term = tf.reduce_sum(d[y] * y, axis=axis, keepdims=True)
    d[x] = y * (d[y] - sum_term)


@adjoint(tf.nn.log_softmax)
def log_softmax(y, x, axis=-1):
    """Adjoint for tf.nn.log_softmax: d[x] = dy - exp(y) * sum(dy, axis)."""
    d[x] = d[y] - tf.exp(y) * tf.reduce_sum(d[y], axis=axis, keepdims=True)


@tangent_(tf.nn.softmax)
def tangent_softmax(y, x, axis=-1):
    """Forward mode for tf.nn.softmax."""
    d[y] = y * (d[x] - tf.reduce_sum(d[x] * y, axis=axis, keepdims=True))


@tangent_(tf.nn.log_softmax)
def tangent_log_softmax(y, x, axis=-1):
    """Forward mode for tf.nn.log_softmax: dy = dx - sum(softmax(x) * dx)."""
    d[y] = d[x] - tf.reduce_sum(tf.exp(y) * d[x], axis=axis, keepdims=True)


# ============================================================================
# Linear Algebra Operations
# ============================================================================

try:
    @adjoint(tf.linalg.inv)
    def linalg_inv(y, x):
        """Adjoint for tf.linalg.inv (matrix inverse).

        For Y = inv(X):
            ∂L/∂X = -Y^T @ ∂L/∂Y @ Y^T
        """
        # y = inv(x), so we use it directly
        y_t = tf.transpose(y)
        d[x] = -tf.matmul(tf.matmul(y_t, d[y]), y_t)
except AttributeError:
    pass  # linalg.inv not available


try:
    @adjoint(tf.linalg.trace)
    def linalg_trace(y, x):
        """Adjoint for tf.linalg.trace: ∂L/∂X_ij = ∂L/∂y if i==j else 0"""
        # Gradient flows only to diagonal elements
        shape = tf.shape(x)
        d[x] = d[y] * tf.eye(shape[0], shape[1], dtype=x.dtype)
except AttributeError:
    pass  # linalg.trace not available


@adjoint(tf.transpose)
def transpose(y, x, perm=None):
    """Adjoint for tf.transpose: ∂L/∂x = transpose(∂L/∂z, inverse_perm)"""
    if perm is None:
        # Default transpose (reverse all dimensions)
        d[x] = tf.transpose(d[y])
    else:
        # Compute inverse permutation
        inv_perm = tf.argsort(perm)
        d[x] = tf.transpose(d[y], inv_perm)


# ============================================================================
# Shape Operations
# ============================================================================

# tf.concat / tf.stack take a *list* of tensors, which Tangent cannot
# distribute gradients into. concat_desugar rewrites list-literal calls into
# the varargs helpers below (mirroring the JAX concat_seq/stack_seq
# machinery), whose varargs adjoints split the gradient back per input.

def tf_concat_seq(axis, *tensors):
    """Runtime helper: concatenate a varargs sequence of tensors."""
    return tf.concat(list(tensors), axis=axis)


def tf_stack_seq(axis, *tensors):
    """Runtime helper: stack a varargs sequence of tensors."""
    return tf.stack(list(tensors), axis=axis)


def tf_concat_grads(dz, tensors, axis):
    """Split a concatenated gradient back into per-input gradients."""
    sizes = [t.shape[axis] for t in tensors]
    return tuple(tf.split(dz, sizes, axis=axis))


def tf_stack_grads(dz, tensors, axis):
    """Unstack a stacked gradient along the stacking axis."""
    return tuple(tf.unstack(dz, axis=axis))


non_differentiable.register_non_differentiable_functions(
    tf_concat_grads, tf_stack_grads)


@adjoint(tf_concat_seq)
def adjoint_tf_concat_seq(z, axis, *tensors):
    """Adjoint for tf_concat_seq: split the gradient back per input."""
    d[tensors] = tangent.tf_concat_grads(d[z], tensors, axis)


@adjoint(tf_stack_seq)
def adjoint_tf_stack_seq(z, axis, *tensors):
    """Adjoint for tf_stack_seq: unstack the gradient."""
    d[tensors] = tangent.tf_stack_grads(d[z], tensors, axis)


@tangent_(tf_concat_seq)
def tangent_tf_concat_seq(z, axis, *tensors):
    """Forward mode for tf_concat_seq."""
    d[z] = tangent.tf_concat_seq(axis, *d[tensors])


@tangent_(tf_stack_seq)
def tangent_tf_stack_seq(z, axis, *tensors):
    """Forward mode for tf_stack_seq."""
    d[z] = tangent.tf_stack_seq(axis, *d[tensors])


# The list-argument forms are only reachable when the desugar pass could not
# rewrite the call (e.g. the list is built dynamically). Raise a clear error
# rather than silently producing zero gradients (which is what the previous
# loop-style adjoint did: gradient templates cannot assign through a loop
# variable).
@adjoint(tf.concat)
def concat(dz, values, axis):
    """Not differentiable: pass a list literal so it can be desugared."""
    raise NotImplementedError(
        'tangent can only differentiate tf.concat/tf.stack when the list of '
        'tensors is a literal. Bind the list to a variable assigned once '
        'from a literal, or pass a list literal directly.')


@adjoint(tf.stack)
def stack(dz, values, axis=0):
    """Not differentiable: pass a list literal so it can be desugared."""
    raise NotImplementedError(
        'tangent can only differentiate tf.concat/tf.stack when the list of '
        'tensors is a literal. Bind the list to a variable assigned once '
        'from a literal, or pass a list literal directly.')


# ============================================================================
# Forward-mode (tangent) definitions
#
# These mirror the adjoints above (this module historically registered
# adjoints only). The rules are direct translations of the NumPy/JAX
# tangents exercised by tests/test_forward_extended.py and the backend
# coverage suite.
# ============================================================================

@tangent_(tf.abs)
def tangent_abs(y, x):
    """Forward mode for tf.abs: dy = dx * sign(x)."""
    d[y] = d[x] * tf.sign(x)


@tangent_(tf.square)
def tangent_square(y, x):
    """Forward mode for tf.square: dy = 2 x dx."""
    d[y] = 2.0 * x * d[x]


@tangent_(tf.sqrt)
def tangent_sqrt(y, x):
    """Forward mode for tf.sqrt: dy = dx / (2 y)."""
    d[y] = d[x] / (2.0 * y)


@tangent_(tf.sign)
def tangent_sign(y, x):
    """Forward mode for tf.sign: zero tangent (piecewise constant)."""
    d[y] = tf.zeros_like(x)


@tangent_(tf.floor)
def tangent_floor(y, x):
    """Forward mode for tf.floor: zero tangent (piecewise constant)."""
    d[y] = tf.zeros_like(x)


@tangent_(tf.round)
def tangent_round(y, x):
    """Forward mode for tf.round: zero tangent (piecewise constant)."""
    d[y] = tf.zeros_like(x)


if hasattr(tf.math, 'ceil'):
    @tangent_(tf.math.ceil)
    def tangent_ceil(y, x):
        """Forward mode for tf.math.ceil: zero tangent."""
        d[y] = tf.zeros_like(x)


@tangent_(tf.minimum)
def tangent_minimum(z, x, y):
    """Forward mode for tf.minimum: the tangent of the smaller argument."""
    d[z] = tf.where(x <= y, d[x], d[y])


@tangent_(tf.clip_by_value)
def tangent_clip_by_value(y, x, clip_value_min, clip_value_max):
    """Forward mode for tf.clip_by_value: tangent flows where unclipped."""
    mask = tf.logical_and(x >= clip_value_min, x <= clip_value_max)
    d[y] = d[x] * tf.cast(mask, x.dtype)


@tangent_(tf.where)
def tangent_where(z, condition, x, y):
    """Forward mode for tf.where: pick the tangent of the selected arm."""
    d[z] = tf.where(condition, d[x], d[y])


try:
    @tangent_(tf.math.log10)
    def tangent_log10(y, x):
        """Forward mode for tf.math.log10: dy = dx / (x ln 10)."""
        d[y] = d[x] / (x * tf.math.log(10.0))

    @tangent_(tf.math.log2)
    def tangent_log2(y, x):
        """Forward mode for tf.math.log2: dy = dx / (x ln 2)."""
        d[y] = d[x] / (x * tf.math.log(2.0))

    @tangent_(tf.math.log1p)
    def tangent_log1p(y, x):
        """Forward mode for tf.math.log1p: dy = dx / (1 + x)."""
        d[y] = d[x] / (1.0 + x)

    @tangent_(tf.math.expm1)
    def tangent_expm1(y, x):
        """Forward mode for tf.math.expm1: dy = dx * exp(x)."""
        d[y] = d[x] * tf.exp(x)
except AttributeError:
    pass  # not available in this TF version


@tangent_(tf.reduce_min)
def tangent_reduce_min(y, x, axis=None, keep_dims=False):
    """Forward mode for tf.reduce_min: the tangent of the minimal
    element(s), split evenly across ties (matching the adjoint)."""
    min_val = tangent.unreduce(y, tangent.shape_as_list(x), axis, keep_dims)
    mask = tf.cast(tf.equal(x, min_val), x.dtype)
    num_min = tf.reduce_sum(mask, axis=axis, keepdims=keep_dims)
    d[y] = tf.reduce_sum(d[x] * mask, axis=axis, keepdims=keep_dims) / num_min


@tangent_(tf.reduce_prod)
def tangent_reduce_prod(y, x, axis=None, keep_dims=False):
    """Forward mode for tf.reduce_prod: dy = sum(dx * prod(x) / x_i)."""
    y_unreduced = tangent.unreduce(y, tangent.shape_as_list(x), axis,
                                   keep_dims)
    d[y] = tf.reduce_sum(d[x] * y_unreduced / x, axis=axis,
                         keepdims=keep_dims)


@tangent_(tf.sin)
def tangent_sin(y, x):
    """Forward mode for tf.sin: dy = dx * cos(x)."""
    d[y] = d[x] * tf.cos(x)


@tangent_(tf.cos)
def tangent_cos(y, x):
    """Forward mode for tf.cos: dy = -dx * sin(x)."""
    d[y] = -d[x] * tf.sin(x)


@tangent_(tf.tan)
def tangent_tan(y, x):
    """Forward mode for tf.tan: dy = dx * (1 + tan(x)^2)."""
    d[y] = d[x] * (1.0 + tf.square(y))


try:
    @tangent_(tf.asin)
    def tangent_asin(y, x):
        """Forward mode for tf.asin: dy = dx / sqrt(1 - x^2)."""
        d[y] = d[x] / tf.sqrt(1.0 - tf.square(x))

    @tangent_(tf.acos)
    def tangent_acos(y, x):
        """Forward mode for tf.acos: dy = -dx / sqrt(1 - x^2)."""
        d[y] = -d[x] / tf.sqrt(1.0 - tf.square(x))
except AttributeError:
    pass  # not available


@tangent_(tf.atan)
def tangent_atan(y, x):
    """Forward mode for tf.atan: dy = dx / (1 + x^2)."""
    d[y] = d[x] / (1.0 + tf.square(x))


@tangent_(tf.nn.relu)
def tangent_relu(y, x):
    """Forward mode for tf.nn.relu: dy = dx where x > 0, else 0."""
    d[y] = d[x] * tf.cast(x > 0, x.dtype)


@tangent_(tf.nn.sigmoid)
def tangent_sigmoid(y, x):
    """Forward mode for tf.nn.sigmoid: dy = dx * y * (1 - y)."""
    d[y] = d[x] * y * (1.0 - y)


try:
    @tangent_(tf.linalg.inv)
    def tangent_linalg_inv(y, x):
        """Forward mode for tf.linalg.inv: dy = -y @ dx @ y."""
        d[y] = -tf.matmul(tf.matmul(y, d[x]), y)

    @tangent_(tf.linalg.trace)
    def tangent_linalg_trace(y, x):
        """Forward mode for tf.linalg.trace: dy = trace(dx)."""
        d[y] = tf.linalg.trace(d[x])
except AttributeError:
    pass  # not available


@tangent_(tf.transpose)
def tangent_transpose(y, x, perm=None):
    """Forward mode for tf.transpose."""
    d[y] = tf.transpose(d[x], perm)


# ============================================================================
# Update UNIMPLEMENTED_ADJOINTS (if it exists in TF extensions)
# ============================================================================

# List of functions we registered
_our_functions = [
    tf.abs, tf.square, tf.sqrt, tf.sign, tf.floor, tf.round,
    tf.minimum, tf.clip_by_value, tf.where,
    tf.sin, tf.cos, tf.tan, tf.atan,
    tf.nn.relu, tf.nn.sigmoid, tf.nn.softmax, tf.nn.log_softmax,
    tf.transpose, tf.concat, tf.stack,
    tf.reduce_min, tf.reduce_prod,
]

# Add ceil based on TF version
if hasattr(tf.math, 'ceil'):
    _our_functions.append(tf.math.ceil)
elif hasattr(tf, 'ceil'):
    _our_functions.append(tf.ceil)

# Add optional functions if they exist
try:
    _our_functions.extend([tf.reciprocal, tf.math.log10, tf.math.log2,
                          tf.math.log1p, tf.math.expm1])
except AttributeError:
    pass

try:
    _our_functions.extend([tf.asin, tf.acos])
except AttributeError:
    pass

try:
    _our_functions.extend([tf.linalg.inv, tf.linalg.trace])
except AttributeError:
    pass

# Remove from UNIMPLEMENTED_ADJOINTS if it exists
try:
    from tangent import grads as _grads_module
    for func in _our_functions:
        _grads_module.UNIMPLEMENTED_ADJOINTS.discard(func)
except (ImportError, AttributeError):
    pass  # UNIMPLEMENTED_ADJOINTS doesn't exist or isn't relevant for TF

# Same for the forward-mode registry: tf_extensions blacklists every TF
# module function without a tangent at its import time, which runs before
# this module registers tangents for the ops above.
_our_tangent_functions = [
    tf.abs, tf.square, tf.sqrt, tf.sign, tf.floor, tf.round,
    tf.minimum, tf.clip_by_value, tf.where,
    tf.sin, tf.cos, tf.tan, tf.atan,
    tf.nn.relu, tf.nn.sigmoid, tf.nn.softmax, tf.nn.log_softmax,
    tf.transpose,
    tf.reduce_min, tf.reduce_prod,
]

if hasattr(tf.math, 'ceil'):
    _our_tangent_functions.append(tf.math.ceil)

try:
    _our_tangent_functions.extend([tf.math.log10, tf.math.log2,
                                   tf.math.log1p, tf.math.expm1])
except AttributeError:
    pass

try:
    _our_tangent_functions.extend([tf.asin, tf.acos])
except AttributeError:
    pass

try:
    _our_tangent_functions.extend([tf.linalg.inv, tf.linalg.trace])
except AttributeError:
    pass

try:
    from tangent import tangents as _tangents_module
    for func in _our_tangent_functions:
        _tangents_module.UNIMPLEMENTED_TANGENTS.discard(func)
except (ImportError, AttributeError):
    pass

import logging as _logging
_logging.getLogger('tangent').debug(
    'Extended TensorFlow gradients loaded successfully (%d new gradient '
    'definitions)', len(_our_functions))
