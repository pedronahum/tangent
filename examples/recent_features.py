#!/usr/bin/env python3
"""Showcase of Tangent's recent capabilities.

This script demonstrates features added on top of Tangent's core
source-to-source autodiff:

  1. Straight-line coarsening   - a symbolic optimizer that emits one compact
                                  vector-Jacobian product instead of one adjoint
                                  statement per op.
  2. Pytree (container) args    - tuples, lists and dicts of arrays as
                                  differentiable function arguments.
  3. Multi-backend              - the same tangent.grad across NumPy, PyTorch
                                  and Keras 3.
  4. concat/stack               - differentiable jnp.concatenate / jnp.stack,
                                  including variable-bound lists.
  5. Second derivatives         - grad(grad(f)), including through loops.

Run it:
    python examples/recent_features.py

Optional backends (torch, keras, jax) are used when installed and skipped
gracefully otherwise; NumPy always runs.
"""
import inspect
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import tangent

# Optional backends.
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

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Straight-line coarsening
# ---------------------------------------------------------------------------
def demo_coarsening():
    _hr("1. Straight-line coarsening: one symbolic VJP instead of per-op adjoints")

    import gast
    import textwrap
    from tangent.optimizations.coarsening import apply_coarsening

    def kernel(a, b, c):
        t1 = a * b
        t2 = np.sin(t1)
        t3 = np.exp(t2) + c
        return t3 * a

    df_std = tangent.grad(kernel, wrt=(0, 1, 2), optimized=False)
    df_co = tangent.grad(kernel, wrt=(0, 1, 2), optimized=True,
                         optimizations={'coarsening': True})

    std_src = inspect.getsource(df_std)
    # The coarsened adjoint itself (a single symbolic VJP), not the wrapper.
    kernel_src = textwrap.dedent(inspect.getsource(kernel))
    adj_ast = apply_coarsening(gast.parse(kernel_src).body[0])
    co_src = gast.unparse(adj_ast)

    print("Standard (unoptimized) gradient: %d lines" % len(std_src.splitlines()))
    print("Coarsened adjoint:               %d lines" % len(co_src.splitlines()))
    print("\nCoarsened adjoint (one symbolic vector-Jacobian product):")
    print(co_src)

    # Both agree numerically.
    a, b, c = 0.5, 1.2, 0.7
    g_std = df_std(a, b, c)
    g_co = df_co(a, b, c)
    match = all(np.allclose(s, co) for s, co in zip(g_std, g_co))
    print("Coarsened matches standard:", "PASS" if match else "FAIL")


# ---------------------------------------------------------------------------
# 2. Pytree (container) arguments
# ---------------------------------------------------------------------------
def demo_pytree_args():
    _hr("2. Pytree (container) arguments: dicts and tuples of arrays")

    # Dict of arrays.
    def loss_dict(params):
        return np.sum(params['w'] * params['x']) + np.sum(params['b'])

    dloss = tangent.grad(loss_dict)
    params = {'w': np.array([1.0, 2.0]), 'x': np.array([3.0, 4.0]),
              'b': np.array([0.5, 0.5])}
    grads = dloss(params)
    print("Dict-arg gradients:")
    print("  d/dw =", grads['w'], "(expected x = [3. 4.])")
    print("  d/dx =", grads['x'], "(expected w = [1. 2.])")
    print("  d/db =", grads['b'], "(expected ones)")

    # Tuple of arrays.
    def loss_tuple(params):
        return np.sum(params[0] ** 2) + np.sum(params[1])

    dloss_t = tangent.grad(loss_tuple)
    g0, g1 = dloss_t((np.array([1.0, 2.0]), np.array([5.0, 6.0])))
    print("\nTuple-arg gradients:")
    print("  d/d(params[0]) =", g0, "(expected 2*params[0])")
    print("  d/d(params[1]) =", g1, "(expected ones)")


# ---------------------------------------------------------------------------
# 3. Multi-backend: one API across NumPy, PyTorch, Keras
# ---------------------------------------------------------------------------
def demo_multibackend():
    _hr("3. Multi-backend: the same tangent.grad across NumPy, PyTorch, Keras")

    x_np = np.array([1.0, 2.0, 3.0], dtype='float32')

    def f_np(x):
        return np.sum(np.tanh(x))

    g = tangent.grad(f_np)(x_np)
    print("NumPy   grad of sum(tanh(x)):", g)

    if HAS_TORCH:
        def f_torch(x):
            return torch.sum(torch.tanh(x))

        g = tangent.grad(f_torch)(torch.as_tensor(x_np))
        print("PyTorch grad of sum(tanh(x)):", np.asarray(g))
    else:
        print("PyTorch: not installed, skipped")

    if HAS_KERAS:
        def f_keras(x):
            return kops.sum(kops.tanh(x))

        g = tangent.grad(f_keras)(kops.convert_to_tensor(x_np))
        print("Keras   grad of sum(tanh(x)):", np.asarray(g))
    else:
        print("Keras  : not installed, skipped")


# ---------------------------------------------------------------------------
# 4. Differentiable concat/stack (including variable-bound lists)
# ---------------------------------------------------------------------------
def demo_concat_stack():
    _hr("4. Differentiable jnp.concatenate / jnp.stack (JAX)")
    if not HAS_JAX:
        print("JAX not installed, skipped")
        return

    # Literal list.
    def f_lit(a, b):
        return jnp.sum(jnp.concatenate([a, b], axis=0))

    ga, gb = tangent.grad(f_lit, wrt=(0, 1))(jnp.array([1.0, 2.0]),
                                              jnp.array([3.0]))
    print("concatenate literal  -> d/da =", np.asarray(ga), ", d/db =", np.asarray(gb))

    # Variable-bound list (assigned once, never mutated).
    def f_var(a, b):
        parts = [a, b]
        return jnp.sum(jnp.stack(parts, axis=0))

    ga, gb = tangent.grad(f_var, wrt=(0, 1))(jnp.array([1.0, 2.0]),
                                              jnp.array([3.0, 4.0]))
    print("stack (variable list) -> d/da =", np.asarray(ga), ", d/db =", np.asarray(gb))


# ---------------------------------------------------------------------------
# 5. Second derivatives
# ---------------------------------------------------------------------------
def demo_second_derivatives():
    _hr("5. Second derivatives: grad(grad(f)), including through a loop")

    def cubic(x):
        return x ** 3

    ddf = tangent.grad(tangent.grad(cubic))
    print("d^2/dx^2 x^3 at x=2 =", ddf(2.0), "(expected 12.0)")

    def power_loop(x, n):
        acc = x
        for _ in range(n):
            acc = acc * x
        return acc  # x^(n+1)

    d2 = tangent.grad(tangent.grad(power_loop))
    # d^2/dx^2 x^4 = 12 x^2 ; at x=2 -> 48
    print("d^2/dx^2 x^4 at x=2 (via loop) =", d2(2.0, 3), "(expected 48.0)")


if __name__ == '__main__':
    print("Tangent - recent features showcase")
    demo_coarsening()
    demo_pytree_args()
    demo_multibackend()
    demo_concat_stack()
    demo_second_derivatives()
    print("\n" + "=" * 70)
    print("Showcase complete.")
    print("=" * 70)
