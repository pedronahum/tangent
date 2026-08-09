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
"""Cross-backend gradient coverage for JAX, TensorFlow, PyTorch and Keras.

This module runs one shared catalog of operations through Tangent's
reverse-mode autodiff for every available backend and checks the result
against analytic gradients. It exists to keep the four backend extensions
at comparable coverage: when an op is added to one extension, add it to
the catalog here so all backends are held to the same bar.

Each test is skipped individually when its backend is not installed.
"""
import numpy as np
import pytest

import tangent

# ---------------------------------------------------------------------------
# Backend availability and helpers
# ---------------------------------------------------------------------------

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import keras.ops as kops
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False


BACKENDS = []
if HAS_JAX:
    BACKENDS.append(('jax', jnp))
if HAS_TF:
    BACKENDS.append(('tf', tf))
if HAS_TORCH:
    BACKENDS.append(('torch', torch))
if HAS_KERAS:
    BACKENDS.append(('keras', kops))

BACKEND_IDS = [b[0] for b in BACKENDS]


def requires_backend(name):
    available = {'jax': HAS_JAX, 'tf': HAS_TF, 'torch': HAS_TORCH,
                 'keras': HAS_KERAS}[name]
    return pytest.mark.skipif(not available,
                              reason='%s not installed' % name)


def to_backend(mod, arr):
    """Convert a numpy array to the backend's tensor type."""
    arr = np.asarray(arr, dtype='float32')
    if mod is jnp:
        return jnp.asarray(arr)
    if mod is tf:
        return tf.constant(arr)
    if HAS_TORCH and mod is torch:
        return torch.as_tensor(arr)
    return kops.convert_to_tensor(arr)  # keras


def from_backend(mod, t):
    """Convert a backend tensor back to numpy."""
    if isinstance(t, np.ndarray):
        return t
    if mod is jnp:
        return np.asarray(t)
    if mod is tf:
        return t.numpy()
    if HAS_TORCH and mod is torch:
        return t.detach().cpu().numpy()
    return np.asarray(kops.convert_to_numpy(t))


def allclose(a, b, tol=2e-4):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol, rtol=1e-3)


def backend_sum(backend, mod, x):
    """Sum that works across backends (TF names it reduce_sum)."""
    if backend == 'tf':
        return tf.reduce_sum(x)
    if backend == 'keras':
        return kops.sum(x)
    return mod.sum(x)


def backend_mean(backend, mod, x):
    """Mean that works across backends (TF names it reduce_mean)."""
    if backend == 'tf':
        return tf.reduce_mean(x)
    if backend == 'keras':
        return kops.mean(x)
    return mod.mean(x)


def grad_call(backend, df, *inputs):
    """Call a gradient function with a backend-appropriate seed.

    TF requires the seed dtype to match the primal dtype (a bare Python
    float seed is float64 and poisons float32 accumulation), so an explicit
    float32 seed is passed; the other backends coerce Python scalars fine.
    """
    if backend == 'tf':
        return df(*inputs, tf.constant(1.0, dtype='float32'))
    return df(*inputs)


# ---------------------------------------------------------------------------
# Unary op catalog: name -> (backend op name, analytic d/dx of sum(op(x)))
#
# Inputs are chosen from the safe domain of each op (positive for log/sqrt,
# inside [-1, 1] for the inverse trig functions).
# ---------------------------------------------------------------------------

X_POS = np.array([0.3, 0.7, 1.2], dtype='float32')
X_UNIT = np.array([-0.4, 0.1, 0.6], dtype='float32')
X_ANY = np.array([-0.8, 0.2, 1.1], dtype='float32')

UNARY_OPS = {
    'negative': ('negative', lambda x: -np.ones_like(x), X_ANY),
    'exp': ('exp', lambda x: np.exp(x), X_ANY),
    'log': ('log', lambda x: 1.0 / x, X_POS),
    'sqrt': ('sqrt', lambda x: 1.0 / (2.0 * np.sqrt(x)), X_POS),
    'square': ('square', lambda x: 2.0 * x, X_ANY),
    'sin': ('sin', lambda x: np.cos(x), X_ANY),
    'cos': ('cos', lambda x: -np.sin(x), X_ANY),
    'tan': ('tan', lambda x: 1.0 / np.cos(x) ** 2, X_UNIT),
    'arcsin': ('arcsin', lambda x: 1.0 / np.sqrt(1.0 - x ** 2), X_UNIT),
    'arccos': ('arccos', lambda x: -1.0 / np.sqrt(1.0 - x ** 2), X_UNIT),
    'arctan': ('arctan', lambda x: 1.0 / (1.0 + x ** 2), X_ANY),
    'sinh': ('sinh', lambda x: np.cosh(x), X_ANY),
    'cosh': ('cosh', lambda x: np.sinh(x), X_ANY),
    'tanh': ('tanh', lambda x: 1.0 - np.tanh(x) ** 2, X_ANY),
    'abs': ('abs', lambda x: np.sign(x), X_ANY),
    'relu': ('relu', lambda x: (x > 0).astype(x.dtype), X_ANY),
    'sigmoid': ('sigmoid',
                lambda x: 1.0 / (1.0 + np.exp(-x)) *
                (1.0 - 1.0 / (1.0 + np.exp(-x))), X_ANY),
    # Piecewise-constant ops have zero gradient everywhere (they are
    # discontinuous); Tangent must return zeros rather than fail.
    'floor': ('floor', lambda x: np.zeros_like(x), X_ANY),
    'ceil': ('ceil', lambda x: np.zeros_like(x), X_ANY),
    'round': ('round', lambda x: np.zeros_like(x), X_ANY),
    'sign': ('sign', lambda x: np.zeros_like(x), X_ANY),
}


def _backend_op(mod, op_name):
    """Fetch an op from a backend, tolerating small naming differences."""
    return getattr(mod, op_name)


# TF exposes some ops under tf.math / tf.nn rather than top-level.
_TF_OP_MAP = {'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan',
              'floor': 'floor', 'ceil': 'ceil', 'round': 'round',
              'sign': 'sign'}


def _resolve_unary_op(backend, mod, op_name):
  """Return the backend callable for a catalog unary op, or None if the
  backend does not expose it (caller skips)."""
  if backend == 'tf':
    if op_name == 'relu':
      return tf.nn.relu
    if op_name == 'sigmoid':
      return tf.math.sigmoid
    if op_name in _TF_OP_MAP:
      return getattr(tf.math, _TF_OP_MAP[op_name])
  if backend == 'keras' and op_name in ('relu', 'sigmoid'):
    return getattr(kops, op_name)
  try:
    return _backend_op(mod, UNARY_OPS[op_name][0])
  except AttributeError:
    return None


def _unary_tangent_grad(backend, mod, op, x_np):
  """Tangent's reverse-mode gradient of sum(op(x)) at x_np, as numpy."""
  def f(x):
    return backend_sum(backend, mod, op(x))

  df = tangent.grad(f)
  return from_backend(mod, grad_call(backend, df, to_backend(mod, x_np)))


@pytest.mark.parametrize('backend', BACKEND_IDS)
@pytest.mark.parametrize('op_name', sorted(UNARY_OPS))
def test_unary_gradients(backend, op_name):
    mod = dict(BACKENDS)[backend]
    op = _resolve_unary_op(backend, mod, op_name)
    if op is None:
        pytest.skip('op %s not exposed by backend %s' % (op_name, backend))

    _, analytic, domain = UNARY_OPS[op_name]
    x_np = np.array(domain, dtype='float32')
    got = _unary_tangent_grad(backend, mod, op, x_np)
    assert allclose(got, analytic(x_np)), \
        '%s.%s: got %s expected %s' % (backend, op_name, got, analytic(x_np))


# ---------------------------------------------------------------------------
# Finite-difference cross-check: an independent numerical oracle.
#
# The tests above compare against hand-derived analytic gradients. This one
# compares against central finite differences computed in NumPy, so it
# cross-validates the analytic expressions and, importantly, auto-verifies the
# correctness of any newly added op/adjoint without a manual derivation -
# which is what keeps the per-backend adjoint surface scalable.
# ---------------------------------------------------------------------------

# NumPy has no relu/sigmoid; provide equivalents for the FD oracle.
_FD_OP_OVERRIDE = {
    'relu': lambda x: np.maximum(x, 0.0),
    'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-x)),
}


def _fd_grad_unary(op_name, x_np, h=1e-4):
    """Central finite-difference gradient of sum(op(x)), in float64."""
    op = _FD_OP_OVERRIDE.get(op_name) or getattr(np, op_name)
    x = np.asarray(x_np, dtype='float64')

    def loss(v):
        return float(np.sum(op(v)))

    grad = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        grad.flat[i] = (loss(x + e) - loss(x - e)) / (2.0 * h)
    return grad


@pytest.mark.parametrize('backend', BACKEND_IDS)
@pytest.mark.parametrize('op_name', sorted(UNARY_OPS))
def test_unary_grads_match_finite_differences(backend, op_name):
    mod = dict(BACKENDS)[backend]
    op = _resolve_unary_op(backend, mod, op_name)
    if op is None:
        pytest.skip('op %s not exposed by backend %s' % (op_name, backend))

    _, _, domain = UNARY_OPS[op_name]
    x_np = np.array(domain, dtype='float32')
    got = _unary_tangent_grad(backend, mod, op, x_np)
    expected = _fd_grad_unary(op_name, x_np)
    # FD is only ~1e-3 accurate for the discontinuity-free points used here;
    # use a looser tolerance than the analytic comparison.
    assert np.allclose(np.asarray(got), expected, atol=1e-2, rtol=1e-2), \
        '%s.%s: got %s expected ~%s' % (backend, op_name, got, expected)


# ---------------------------------------------------------------------------
# Binary op catalog: name -> (op name, analytic d/dx1 of sum(op(x1, x2)))
# ---------------------------------------------------------------------------

X1 = np.array([1.0, 2.0, 3.0], dtype='float32')
X2 = np.array([0.5, 1.5, 2.5], dtype='float32')

BINARY_OPS = {
    'add': ('add', lambda x1, x2: np.ones_like(x1)),
    'subtract': ('subtract', lambda x1, x2: np.ones_like(x1)),
    'multiply': ('multiply', lambda x1, x2: x2),
    'divide': ('divide', lambda x1, x2: 1.0 / x2),
    'power': ('power', lambda x1, x2: x2 * x1 ** (x2 - 1)),
    'maximum': ('maximum', lambda x1, x2: (x1 >= x2).astype('float32')),
    'minimum': ('minimum', lambda x1, x2: (x1 <= x2).astype('float32')),
}

# power needs positive bases for clean gradients
X1_POS = np.array([1.0, 2.0, 3.0], dtype='float32')
X2_POW = np.array([2.0, 2.0, 2.0], dtype='float32')


@pytest.mark.parametrize('backend', BACKEND_IDS)
@pytest.mark.parametrize('op_name', sorted(BINARY_OPS))
def test_binary_gradients(backend, op_name):
    mod = dict(BACKENDS)[backend]
    name = BINARY_OPS[op_name][0]
    if backend == 'tf':
        tf_map = {'subtract': tf.subtract, 'multiply': tf.multiply,
                  'divide': tf.divide, 'power': tf.math.pow,
                  'maximum': tf.maximum, 'minimum': tf.minimum,
                  'add': tf.add}
        op = tf_map[op_name]
    else:
        try:
            op = _backend_op(mod, name)
        except AttributeError:
            pytest.skip('op %s not exposed by backend %s' % (op_name, backend))

    analytic = BINARY_OPS[op_name][1]
    if op_name == 'power':
        x1_np, x2_np = np.array(X1_POS), np.array(X2_POW)
    else:
        x1_np, x2_np = np.array(X1), np.array(X2)

    def f(x1, x2):
        return backend_sum(backend, mod, op(x1, x2))

    df = tangent.grad(f, wrt=(0,))
    got = from_backend(mod, grad_call(backend, df, to_backend(mod, x1_np), to_backend(mod, x2_np)))
    assert allclose(got, analytic(x1_np, x2_np)), \
        '%s.%s: got %s expected %s' % (backend, op_name, got,
                                       analytic(x1_np, x2_np))


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKEND_IDS)
def test_sum_gradient(backend):
    mod = dict(BACKENDS)[backend]
    x_np = np.array([1.0, 2.0, 3.0], dtype='float32')

    def f(x):
        return backend_sum(backend, mod, x)

    got = from_backend(mod, grad_call(backend, tangent.grad(f), to_backend(mod, x_np)))
    assert allclose(got, np.ones(3))


@pytest.mark.parametrize('backend', BACKEND_IDS)
def test_mean_gradient(backend):
    mod = dict(BACKENDS)[backend]
    x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype='float32')

    def f(x):
        return backend_mean(backend, mod, x)

    got = from_backend(mod, grad_call(backend, tangent.grad(f), to_backend(mod, x_np)))
    assert allclose(got, np.full(4, 0.25))


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKEND_IDS)
def test_matmul_gradient(backend):
    mod = dict(BACKENDS)[backend]
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
    w_np = np.array([[0.5, -0.5], [1.0, 0.0]], dtype='float32')

    if backend == 'keras':
        def f(x, w):
            return kops.sum(kops.matmul(x, w))
    else:
        def f(x, w):
            return backend_sum(backend, mod, mod.matmul(x, w))

    # dL/dW = X^T @ ones
    got = from_backend(mod, grad_call(backend, tangent.grad(f, wrt=(1,)),
                                        to_backend(mod, x_np), to_backend(mod, w_np)))
    assert allclose(got, x_np.T @ np.ones((2, 2)))


@pytest.mark.parametrize('backend', BACKEND_IDS)
def test_vector_dot_gradient(backend):
    if backend == 'tf':
        pytest.skip('tf.matmul requires rank >= 2; no 1-D dot adjoint is '
                    'registered for TensorFlow')
    mod = dict(BACKENDS)[backend]
    x_np = np.array([1.0, 2.0, 3.0], dtype='float32')
    y_np = np.array([4.0, 5.0, 6.0], dtype='float32')

    def f(x, y):
        return mod.matmul(x, y) if backend != 'keras' else kops.matmul(x, y)

    got = from_backend(mod, grad_call(backend, tangent.grad(f, wrt=(0,)),
                                        to_backend(mod, x_np), to_backend(mod, y_np)))
    assert allclose(got, y_np)


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKEND_IDS)
def test_reshape_gradient(backend):
    mod = dict(BACKENDS)[backend]
    x_np = np.arange(6.0, dtype='float32')

    if backend == 'keras':
        def f(x):
            return kops.sum(kops.reshape(x, (2, 3)))
    else:
        def f(x):
            return backend_sum(backend, mod, mod.reshape(x, (2, 3)))

    got = from_backend(mod, grad_call(backend, tangent.grad(f), to_backend(mod, x_np)))
    assert allclose(got, np.ones(6))


@pytest.mark.parametrize('backend', BACKEND_IDS)
def test_transpose_gradient(backend):
    mod = dict(BACKENDS)[backend]
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32')

    # Each backend gets its own function definition: Tangent resolves every
    # call in the source, including dead branches, so a shared if/elif body
    # would bind other backends' calls against the active closure.
    if backend == 'tf':
        def f(x):
            return backend_sum(backend, mod, tf.transpose(x))
    elif backend == 'keras':
        def f(x):
            return backend_sum(backend, mod, kops.transpose(x))
    elif backend == 'torch':
        def f(x):
            # torch.transpose requires explicit dimensions
            return backend_sum(backend, mod, mod.transpose(x, 0, 1))
    else:
        def f(x):
            return backend_sum(backend, mod, mod.transpose(x))

    got = from_backend(mod, grad_call(backend, tangent.grad(f), to_backend(mod, x_np)))
    assert allclose(got, np.ones((2, 3)))


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKEND_IDS)
def test_clip_gradient(backend):
    mod = dict(BACKENDS)[backend]
    x_np = np.array([-1.0, 0.5, 2.0], dtype='float32')

    # The clipping op and the reduction are both backend-specific, so each
    # backend gets its own function definition.
    if backend == 'jax':
        def f(x):
            return jnp.sum(jnp.clip(x, 0.0, 1.0))
    elif backend == 'tf':
        def f(x):
            return tf.reduce_sum(tf.clip_by_value(x, 0.0, 1.0))
    elif backend == 'torch':
        def f(x):
            return torch.sum(torch.clamp(x, 0.0, 1.0))
    else:
        def f(x):
            return kops.sum(kops.clip(x, 0.0, 1.0))

    got = from_backend(mod, grad_call(backend, tangent.grad(f), to_backend(mod, x_np)))
    # Gradient is 1 inside the clip range and 0 where the value was clipped.
    assert allclose(got, np.array([0.0, 1.0, 0.0], dtype='float32'))


# ---------------------------------------------------------------------------
# Dimension manipulation: squeeze / expand_dims
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKEND_IDS)
def test_squeeze_gradient(backend):
    mod = dict(BACKENDS)[backend]
    x_np = np.arange(3.0, dtype='float32').reshape(1, 3, 1)

    if backend == 'jax':
        def f(x):
            return jnp.sum(jnp.squeeze(x))
    elif backend == 'tf':
        def f(x):
            return tf.reduce_sum(tf.squeeze(x))
    elif backend == 'torch':
        def f(x):
            return torch.sum(torch.squeeze(x))
    else:
        def f(x):
            return kops.sum(kops.squeeze(x))

    got = from_backend(mod, grad_call(backend, tangent.grad(f), to_backend(mod, x_np)))
    # The gradient is broadcast back to the original (un-squeezed) shape.
    assert allclose(got, np.ones((1, 3, 1), dtype='float32'))


@pytest.mark.parametrize('backend', BACKEND_IDS)
def test_expand_dims_gradient(backend):
    mod = dict(BACKENDS)[backend]
    x_np = np.array([1.0, 2.0, 3.0], dtype='float32')

    if backend == 'jax':
        def f(x):
            return jnp.sum(jnp.expand_dims(x, 0))
    elif backend == 'tf':
        def f(x):
            return tf.reduce_sum(tf.expand_dims(x, 0))
    elif backend == 'torch':
        def f(x):
            return torch.sum(torch.unsqueeze(x, 0))
    else:
        def f(x):
            return kops.sum(kops.expand_dims(x, 0))

    got = from_backend(mod, grad_call(backend, tangent.grad(f), to_backend(mod, x_np)))
    assert allclose(got, np.ones(3, dtype='float32'))


# ---------------------------------------------------------------------------
# Reduction max / min (gradient flows to the extremal element)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKEND_IDS)
def test_reduce_max_gradient(backend):
    mod = dict(BACKENDS)[backend]
    x_np = np.array([1.0, 3.0, 2.0], dtype='float32')

    if backend == 'jax':
        def f(x):
            return jnp.max(x)
    elif backend == 'tf':
        def f(x):
            return tf.reduce_max(x)
    elif backend == 'torch':
        def f(x):
            return torch.max(x)
    else:
        def f(x):
            return kops.max(x)

    got = from_backend(mod, grad_call(backend, tangent.grad(f), to_backend(mod, x_np)))
    assert allclose(got, np.array([0.0, 1.0, 0.0], dtype='float32'))


@pytest.mark.parametrize('backend', BACKEND_IDS)
def test_reduce_min_gradient(backend):
    mod = dict(BACKENDS)[backend]
    x_np = np.array([3.0, 1.0, 2.0], dtype='float32')

    if backend == 'jax':
        def f(x):
            return jnp.min(x)
    elif backend == 'tf':
        def f(x):
            return tf.reduce_min(x)
    elif backend == 'torch':
        def f(x):
            return torch.min(x)
    else:
        def f(x):
            return kops.min(x)

    got = from_backend(mod, grad_call(backend, tangent.grad(f), to_backend(mod, x_np)))
    assert allclose(got, np.array([0.0, 1.0, 0.0], dtype='float32'))


# ---------------------------------------------------------------------------
# Conditional selection: where(condition, x, y)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKEND_IDS)
def test_where_gradient(backend):
    mod = dict(BACKENDS)[backend]
    x_np = np.array([-1.0, 2.0, -3.0], dtype='float32')
    y_np = np.array([4.0, 5.0, 6.0], dtype='float32')

    # d/dx of sum(where(x>0, x, y)) is the mask (x>0); d/dy is its complement.
    if backend == 'jax':
        def f(x, y):
            return jnp.sum(jnp.where(x > 0, x, y))
    elif backend == 'tf':
        def f(x, y):
            return tf.reduce_sum(tf.where(x > 0, x, y))
    elif backend == 'torch':
        def f(x, y):
            return torch.sum(torch.where(x > 0, x, y))
    else:
        def f(x, y):
            return kops.sum(kops.where(x > 0, x, y))

    df = tangent.grad(f, wrt=(0, 1))
    gx, gy = df(to_backend(mod, x_np), to_backend(mod, y_np))
    assert allclose(from_backend(mod, gx), np.array([0.0, 1.0, 0.0]))
    assert allclose(from_backend(mod, gy), np.array([1.0, 0.0, 1.0]))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
