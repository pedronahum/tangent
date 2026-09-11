# Backends

The same `tangent.grad` call differentiates code written against any
supported array library. Backends are optional: each extension loads when its
library is importable and stays silent otherwise (`tangent.backend_status()`
reports what loaded).

```python
import tangent

def f_np(x):       return np.sum(x ** 2)          # NumPy
def f_jax(x):      return jnp.sum(jnp.tanh(x))    # JAX
def f_tf(x):       return tf.reduce_sum(tf.tanh(x))   # TensorFlow eager
def f_torch(x):    return torch.sum(torch.tanh(x))    # PyTorch
def f_keras(x):    return kops.sum(kops.tanh(x))      # Keras 3, any backend
def f_tinygrad(x): return x.tanh().sum()              # tinygrad methods

for f in (f_np, f_jax, f_tf, f_torch, f_keras, f_tinygrad):
    df = tangent.grad(f)
```

## Coverage

| Backend | Module | Reverse adjoints | Forward tangents | Notes |
|---|---|---|---|---|
| NumPy | `grads.py` + `numpy_extended.py` | 90+ | 55+ | Core + einsum, sort, linalg (inv/det/solve/norm/cholesky/eigvalsh), reductions, shape ops |
| JAX | `jax_extensions.py` | 50+ | 45+ | `jax.numpy` and `jax.nn`; reference backend for second-order tests |
| TensorFlow 2.x | `tf_extensions.py` + `tf_extended.py` | 45+ | 20+ | Eager mode; conv/pooling, linalg, reductions |
| PyTorch | `torch_extensions.py` | 45+ | 30+ | Functional `torch.*` API, verified against `torch.autograd` |
| Keras 3 | `keras_extensions.py` | 35+ | ~20 | Backend-agnostic `keras.ops` (TF, JAX, or torch underneath) |
| tinygrad | `tinygrad_extensions.py` | 65+ | ~20 | Method API incl. conv2d/pooling/layernorm/batchnorm; the generated gradient is itself a tinygrad graph that tinygrad's compiler fuses |

Cross-backend parity is enforced by a shared op catalog
(`tests/test_backend_coverage.py`) checked against analytic *and*
finite-difference oracles for every installed backend, and the elementwise
rules are generated from one backend-neutral table
(`tangent/elementwise_rules.py`) so definitions cannot drift.

## SciPy

With SciPy installed, `scipy.special` (erf/erfc, gammaln/gamma/digamma,
expit/logit, xlogy, logsumexp) and `scipy.linalg` (solve, inv) differentiate
too - the scientific/statistical function set. See the changelog for the full
list.

## Backend notes

- **The `@` operator** works for NumPy, JAX, TensorFlow, PyTorch and
  tinygrad (dispatched via `tangent.utils.register_matmul_grad`).
- **TensorFlow seeds**: pass an explicit seed matching your tensor dtype,
  e.g. `df(x, tf.constant(1.0, dtype=x.dtype))`.
- **Keras 3** runs on whichever backend Keras is configured with
  (`KERAS_BACKEND=tensorflow|jax|torch`).
- **tinygrad method calls** (`x.sum()`) on computed values are resolved by a
  registered method resolver, scoped to modules that import tinygrad.
- **tinygrad conv/pool limits**: `conv2d` gradients need `groups=1`;
  `max_pool2d` needs `dilation=1`, no `ceil_mode`/`return_indices`,
  symmetric padding. Unsupported configurations raise `NotImplementedError`.

## Extending a backend

Registering a missing gradient is one template:

```python
from tangent.grads import adjoint
import mybackend

@adjoint(mybackend.softplus)
def softplus(y, x):
    d[x] = d[y] / (1.0 + mybackend.exp(-x))
```

The runtime dispatch registries (`register_add_grad`,
`register_unbroadcast`, `register_matmul_grad`, ...) in `tangent.utils`
let new tensor types participate in gradient accumulation — see the
[API Reference](api.md).
