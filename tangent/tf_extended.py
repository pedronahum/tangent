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
    from tangent.elementwise_rules import prefix_vocab
    from tangent import op_catalog
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

# Unary elementwise ops: generated (adjoint AND tangent per op) from the
# unified op catalog. Binary ops (add/subtract/multiply/divide) stay
# hand-written for TF and are intentionally not routed through the catalog.
# Spellings that moved between TF versions (tf.math.ceil, tf.math.reciprocal,
# ...) are guarded with getattr so whichever exists is registered.
op_catalog.register(
    'tf',
    ops={
        'abs': tf.abs,
        'square': tf.square,
        'sqrt': tf.sqrt,
        'reciprocal': (getattr(tf, 'reciprocal', None), getattr(tf.math, 'reciprocal', None)),
        'expm1': getattr(tf.math, 'expm1', None),
        'log2': getattr(tf.math, 'log2', None),
        'log10': getattr(tf.math, 'log10', None),
        'log1p': getattr(tf.math, 'log1p', None),
        'sin': tf.sin,
        'cos': tf.cos,
        'tan': tf.tan,
        'arcsin': getattr(tf, 'asin', None),
        'arccos': getattr(tf, 'acos', None),
        'arctan': tf.atan,
        'relu': tf.nn.relu,
        'sigmoid': tf.nn.sigmoid,
        'sign': tf.sign,
        'floor': tf.floor,
        'ceil': (getattr(tf.math, 'ceil', None) or getattr(tf, 'ceil', None)),
        'round': tf.round,
    },
    vocab=prefix_vocab('tf', mask_pos='tf.cast(({arg}) > 0, x.dtype)'),
)


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
    d[x] = tangent.unbroadcast_tensor(tf.where(condition, d[z], tf.zeros_like(d[z])), x)
    d[y] = tangent.unbroadcast_tensor(tf.where(condition, tf.zeros_like(d[z]), d[z]), y)


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
# Neural Network Activations
# ============================================================================


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
    """Split a concatenated gradient back into per-input gradients.

    The incoming gradient can be a NumPy float64 array (the default seed
    expanded by the generic unreduce); elementwise TF ops auto-convert such
    operands to the tensor operand's dtype, but tf.split would mint a float64
    tensor that later poisons float32 arithmetic, so cast explicitly.
    """
    dz = tf.cast(dz, tensors[0].dtype)
    sizes = [t.shape[axis] for t in tensors]
    return tuple(tf.split(dz, sizes, axis=axis))


def tf_stack_grads(dz, tensors, axis):
    """Unstack a stacked gradient along the stacking axis (see tf_concat_grads
    for the cast)."""
    dz = tf.cast(dz, tensors[0].dtype)
    return tuple(tf.unstack(dz, axis=axis))


non_differentiable.register_non_differentiable_functions(tf_concat_grads, tf_stack_grads)


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
        'from a literal, or pass a list literal directly.'
    )


@adjoint(tf.stack)
def stack(dz, values, axis=0):
    """Not differentiable: pass a list literal so it can be desugared."""
    raise NotImplementedError(
        'tangent can only differentiate tf.concat/tf.stack when the list of '
        'tensors is a literal. Bind the list to a variable assigned once '
        'from a literal, or pass a list literal directly.'
    )


# ============================================================================
# Forward-mode (tangent) definitions
#
# These mirror the adjoints above (this module historically registered
# adjoints only). The rules are direct translations of the NumPy/JAX
# tangents exercised by tests/test_forward_extended.py and the backend
# coverage suite.
# ============================================================================


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
    y_unreduced = tangent.unreduce(y, tangent.shape_as_list(x), axis, keep_dims)
    d[y] = tf.reduce_sum(d[x] * y_unreduced / x, axis=axis, keepdims=keep_dims)


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
    tf.abs,
    tf.square,
    tf.sqrt,
    tf.sign,
    tf.floor,
    tf.round,
    tf.minimum,
    tf.clip_by_value,
    tf.where,
    tf.sin,
    tf.cos,
    tf.tan,
    tf.atan,
    tf.nn.relu,
    tf.nn.sigmoid,
    tf.nn.softmax,
    tf.nn.log_softmax,
    tf.transpose,
    tf.concat,
    tf.stack,
    tf.reduce_min,
    tf.reduce_prod,
]

# Add ceil based on TF version
if hasattr(tf.math, 'ceil'):
    _our_functions.append(tf.math.ceil)
elif hasattr(tf, 'ceil'):
    _our_functions.append(tf.ceil)

# Add optional functions if they exist
try:
    _our_functions.extend(
        [tf.reciprocal, tf.math.log10, tf.math.log2, tf.math.log1p, tf.math.expm1]
    )
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
    tf.abs,
    tf.square,
    tf.sqrt,
    tf.sign,
    tf.floor,
    tf.round,
    tf.minimum,
    tf.clip_by_value,
    tf.where,
    tf.sin,
    tf.cos,
    tf.tan,
    tf.atan,
    tf.nn.relu,
    tf.nn.sigmoid,
    tf.nn.softmax,
    tf.nn.log_softmax,
    tf.transpose,
    tf.reduce_min,
    tf.reduce_prod,
]

if hasattr(tf.math, 'ceil'):
    _our_tangent_functions.append(tf.math.ceil)

try:
    _our_tangent_functions.extend([tf.math.log10, tf.math.log2, tf.math.log1p, tf.math.expm1])
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
    'Extended TensorFlow gradients loaded successfully (%d new gradient definitions)',
    len(_our_functions),
)
