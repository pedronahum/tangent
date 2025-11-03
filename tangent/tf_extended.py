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

import warnings

try:
    import tensorflow as tf
    import tangent
    from tangent.grads import adjoint
    from tangent import tf_extensions
except ImportError as e:
    warnings.warn(f"TensorFlow extensions not available: {e}")
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

@adjoint(tf.concat)
def concat(z, values, axis):
    """Adjoint for tf.concat: split gradient back to original tensors"""
    # Compute split sizes based on input tensor sizes
    sizes = [tf.shape(v)[axis] for v in values]

    # Split the gradient
    grads = tf.split(d[z], sizes, axis=axis)
    for i, val in enumerate(values):
        d[val] = grads[i]


@adjoint(tf.stack)
def stack(z, values, axis=0):
    """Adjoint for tf.stack: unstack gradient along the stacking axis"""
    # Unstack the gradient
    grads = tf.unstack(d[z], axis=axis)
    for i, val in enumerate(values):
        d[val] = grads[i]


# ============================================================================
# Update UNIMPLEMENTED_ADJOINTS (if it exists in TF extensions)
# ============================================================================

# List of functions we registered
_our_functions = [
    tf.abs, tf.square, tf.sqrt, tf.sign, tf.floor, tf.round,
    tf.minimum, tf.clip_by_value,
    tf.sin, tf.cos, tf.tan, tf.atan,
    tf.nn.relu, tf.nn.sigmoid, tf.nn.softmax,
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

print("✓ Extended TensorFlow gradients loaded successfully")
print(f"✓ Registered {len(_our_functions)} new gradient definitions")
